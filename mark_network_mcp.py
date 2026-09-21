#!/usr/bin/env python3
"""
mark_network_mcp.py -- MCP (Model Context Protocol) stdio server exposing
the mark_network/context_probe memory engine as a standard, zero-dependency
tool interface.

Why this exists (2026-09-21): GSTR pointed out that "uploaded to GitHub"
is not the same thing as "can be dropped into another system in 5
minutes." Yuki's Context Builder and mark extraction are still
tightly-coupled script logic with no carved-out module boundary, so it
can't define an "interface template" to integrate against. Vivienne's
suggestion: don't wait for that boundary to exist on the other side --
ship this side as a black-boxed, zero-dependency engine with the
simplest possible data contract instead. MCP is the standard,
already-documented way to do that.

No official `mcp` SDK dependency: attempting to install it on this Mac
failed (the `cryptography` dependency needs a working Rust/cargo
toolchain to build a native wheel, and this machine doesn't have one).
The MCP protocol itself is just JSON-RPC 2.0 over newline-delimited
stdio, which is simple and well-documented enough to implement directly
against the standard library -- this also means whoever runs this later
doesn't need Rust installed either.

Exposes exactly the two atomic operations Vivienne proposed:
  - observe: this round's activated marks -> record a co-activation event
             (feeds the dynamic Hebbian shadow layer). No return value of
             substance; just a status.
  - query:   a natural-language message -> the same formatted reference
             text context_probe.probe() would inject into a prompt.

Usage (as a subprocess, talking JSON-RPC over stdin/stdout):
    python3 mark_network_mcp.py

Every line printed to stdout is one JSON-RPC message (request or
response); nothing else is ever written to stdout, so a client reading
line-by-line never has to worry about interleaved debug output.
"""
import json
import sys
import traceback

# These already exist in this project; no reimplementation here, this
# file is purely a protocol wrapper around them.
import context_probe
import mark_network

PROTOCOL_VERSION = "2024-11-05"  # the MCP spec version this speaks
SERVER_NAME = "mark-network"
SERVER_VERSION = "0.1.0"

TOOLS = [
    {
        "name": "observe",
        "description": (
            "Record which marks (topic tags) were activated together in "
            "this round of conversation, feeding the dynamic Hebbian "
            "shadow layer. Call this once per turn with whatever marks "
            "actually fired -- it only logs, it never affects retrieval "
            "until a human has reviewed the shadow-mode data and decided "
            "to wire it in."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "marks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The mark labels activated this round, e.g. [\"文明6\", \"神难度\"]",
                }
            },
            "required": ["marks"],
        },
    },
    {
        "name": "query",
        "description": (
            "Given a raw message, return formatted reference text pulled "
            "from historical records if (and only if) the message's "
            "content matches known marks strongly enough to be worth "
            "surfacing. Returns an empty string, not an error, when "
            "nothing relevant is found -- callers should treat that as "
            "'nothing to add', not a failure."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The raw message text to check for mark matches.",
                }
            },
            "required": ["text"],
        },
    },
]


def handle_observe(args):
    marks = args.get("marks", [])
    if not isinstance(marks, list) or not all(isinstance(m, str) for m in marks):
        raise ValueError("`marks` must be a list of strings")
    mark_network.record_coactivation(marks)
    return {"status": "ok", "recorded": len(marks) >= 2}


def handle_query(args):
    text = args.get("text", "")
    if not isinstance(text, str):
        raise ValueError("`text` must be a string")
    result = context_probe.probe(text)
    return {"reference_text": result}


DISPATCH = {"observe": handle_observe, "query": handle_query}


def _send(msg):
    """Write one JSON-RPC message as a single line to stdout, then flush.
    Nothing else in this process is allowed to write to stdout -- a
    stray print() anywhere would corrupt the message stream for whatever
    is reading it."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _error(req_id, code, message):
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _result(req_id, result):
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {}) or {}

    if method == "initialize":
        _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {}},
        })
    elif method == "notifications/initialized":
        pass  # no response required for a notification
    elif method == "tools/list":
        _result(req_id, {"tools": TOOLS})
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        handler = DISPATCH.get(name)
        if handler is None:
            _error(req_id, -32601, f"unknown tool: {name}")
            return
        try:
            output = handler(args)
            _result(req_id, {
                "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}],
                "isError": False,
            })
        except Exception as e:
            _result(req_id, {
                "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                "isError": True,
            })
    else:
        if req_id is not None:  # only requests need an error response, not notifications
            _error(req_id, -32601, f"unknown method: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue  # malformed line -- nothing sane to respond with, drop it
        try:
            handle_request(req)
        except Exception:
            # Never let one bad request kill the whole server -- log the
            # traceback to stderr (not stdout, which is the protocol
            # channel) and keep reading.
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
