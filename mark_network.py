#!/usr/bin/env python3
"""
mark_network.py -- connects the marks in marks.json into a weighted graph,
based on which marks co-occur in the same diary entry. This is the "memory
neural algorithm" pilot Richard and Bota designed on 2026-09-17.

STANDALONE ON PURPOSE, same philosophy as context_probe.py: build it, test
it with --selftest / --demo, and only wire it into context_probe.py's live
retrieval path once the edges it produces look sane by eye.

Scope of this v1 (deliberately narrow):
  - Co-occurrence unit is the WHOLE diary entry (one file = one day), not
    the paragraph. Paragraph-level would need a segmenter we don't have;
    entry-level is a real simplification, not a hidden shortcut -- it will
    connect marks that showed up on the same busy day even if they were in
    unrelated parts of that entry. Worth knowing before trusting an edge.
  - Edges are STATIC: computed once from the 24 existing diary files via
    NPMI (normalized pointwise mutual information). No model inference is
    used to decide relatedness -- only counting, which is exactly what
    resolves Yuki's original "where do edges come from" objection.
  - The Hebbian/dynamic-reinforcement layer (weights that grow with reuse,
    decay when unused) is implemented but in SHADOW MODE ONLY: it logs
    co-activations and computes what the dynamic weight WOULD become in a
    separate file, and never touches the static edges that anything real
    would read. Nothing in this file is wired into context_probe.py yet.

Usage:
  python3 mark_network.py --build            # (re)build static edges from diaries
  python3 mark_network.py --related 坐标      # show top related marks
  python3 mark_network.py --selftest         # sanity checks + prints edge stats
"""

import json
import math
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
MARKS_PATH = os.path.join(BASE, "marks.json")
DIARY_DIR = os.path.join(BASE, "bota_diary")
EDGES_PATH = os.path.join(BASE, "mark_edges.json")
COACTIVATION_LOG = os.path.join(BASE, "mark_coactivation_log.jsonl")
SHADOW_DYNAMIC_PATH = os.path.join(BASE, "mark_edges_dynamic_shadow.json")

ETA = 1.0        # Hebbian reinforcement step per logged co-activation
DECAY = 0.99     # per-cycle multiplicative decay (shadow mode only)

# Kept in sync with context_probe.py's SELF_REFERENCE_MARKS by hand (can't
# import it directly -- context_probe.py imports THIS module, so importing
# back would be circular). Added 2026-09-19 after the real "bot"<->"bota"
# incident: these marks are substrings of each other by construction, so
# every mention of "bota" also mechanically hits "bot" -- their NPMI/
# Hebbian "relatedness" is a pure artifact of the mark-matching system,
# not a real signal. Their retrieval already goes through the separate
# self-reference short-circuit in context_probe.py, so they get nothing
# from being in this graph and only pollute it (the ballooning weight was
# found in exactly this pair: Hebbian pushed it to ~33x NPMI's [-1,1]
# bound because they co-fire on nearly every message).
SELF_REFERENCE_MARKS = {"bota", "bot", "BOT"}


def load_marks():
    if not os.path.exists(MARKS_PATH):
        return []
    try:
        with open(MARKS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [m for m in data.keys() if m not in SELF_REFERENCE_MARKS]
    except Exception as e:
        print(f"  [warn] could not load {MARKS_PATH}: {e}")
        return []


def diary_files():
    if not os.path.isdir(DIARY_DIR):
        return []
    return sorted(
        os.path.join(DIARY_DIR, fn)
        for fn in os.listdir(DIARY_DIR)
        if fn.endswith(".txt")
    )


def marks_in_entry(text, marks):
    """Which marks appear as a literal substring anywhere in this diary
    entry. Entry-level, not paragraph-level -- see module docstring."""
    return {m for m in marks if m in text}


def build_cooccurrence():
    """Scan every diary entry once; return (doc_freq, co_freq, co_context, N).
    doc_freq[mark] = number of diary entries containing that mark.
    co_freq[(a, b)] = number of diary entries containing BOTH (a < b,
    sorted, so each pair is counted once regardless of order).
    co_context[(a, b)] = set of every OTHER mark that appeared alongside
    both a and b in any entry where they co-occurred -- this is the
    corroboration signature used later to gate expansion (added 2026-09-17
    after the 检索 -> Broker/开源 false-expansion incident: an edge should
    only be trusted for a given message if that message shares some of the
    vocabulary the edge actually came from)."""
    marks = load_marks()
    files = diary_files()
    doc_freq = {m: 0 for m in marks}
    co_freq = {}
    co_context = {}
    n = 0
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"  [skip] could not read {path}: {e}")
            continue
        n += 1
        present = sorted(marks_in_entry(text, marks))
        present_set = set(present)
        for m in present:
            doc_freq[m] += 1
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair = (present[i], present[j])
                co_freq[pair] = co_freq.get(pair, 0) + 1
                others = present_set - {present[i], present[j]}
                co_context.setdefault(pair, set()).update(others)
    return doc_freq, co_freq, co_context, n


def compute_npmi(doc_freq, co_freq, co_context, n):
    """Return edges as {mark_a: {mark_b: {"npmi": val, "context": [...]}}},
    symmetric (both directions stored), keeping only edges with NPMI > 0 --
    i.e. pairs that co-occur MORE than chance would predict, given how
    common each mark already is on its own. NPMI is bounded in [-1, 1]:
    1 = always together, 0 = independent, negative = avoid each other.
    Only positive edges are kept since this network is for finding real
    relatedness, not for modeling mutual exclusion. "context" is the
    corroboration vocabulary from build_cooccurrence -- see its docstring."""
    if n == 0:
        return {}
    edges = {}
    for (a, b), co in co_freq.items():
        if doc_freq.get(a, 0) == 0 or doc_freq.get(b, 0) == 0 or co == 0:
            continue
        p_a = doc_freq[a] / n
        p_b = doc_freq[b] / n
        p_ab = co / n
        pmi = math.log(p_ab / (p_a * p_b))
        if p_ab >= 1.0:
            npmi = 1.0   # co-occur in literally every entry -- the limit case
        else:
            npmi = pmi / -math.log(p_ab)
        if npmi > 0:
            ctx = sorted(co_context.get((a, b), set()))
            edges.setdefault(a, {})[b] = {"npmi": round(npmi, 4), "context": ctx}
            edges.setdefault(b, {})[a] = {"npmi": round(npmi, 4), "context": ctx}
    return edges


def save_edges(edges, path=EDGES_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=1, sort_keys=True)


def load_edges(path=EDGES_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def related_marks(mark, edges=None, top_n=3):
    """Top N one-hop neighbors of a mark by NPMI, highest first. Returns
    (neighbor, npmi, context) triples. Used to expand a direct hit with
    weaker, explicitly-labeled related hits -- NOT to replace the
    direct-hit mechanism context_probe.py already has."""
    if edges is None:
        edges = load_edges()
    neighbors = edges.get(mark, {})
    ranked = sorted(neighbors.items(), key=lambda kv: -kv[1]["npmi"])
    return [(nb, v["npmi"], v.get("context", [])) for nb, v in ranked[:top_n]]


def expand_hits(direct_hits, message_text="", edges=None, max_extra=2,
                 require_corroboration=True):
    """Given marks that were DIRECTLY found in an incoming message, return
    up to max_extra additional (neighbor_mark, via_mark, npmi) triples for
    one-hop related marks.

    CORROBORATION GATE (added 2026-09-17, after the 检索 -> Broker/开源
    false-expansion incident): an edge's "context" is the set of other
    marks that were actually present in the diary entries that produced
    it. If require_corroboration is True (the default), a neighbor is only
    included when message_text contains at least one word from that
    edge's context -- i.e. some sign the CURRENT message is actually in
    the same topic area the edge came from, not just that the direct-hit
    word happens to be polysemous. An edge with empty context (e.g. one
    seeded by live coactivation rather than diary co-occurrence) has
    nothing to corroborate against and is allowed through unfiltered."""
    if edges is None:
        edges = load_edges()
    seen = set(direct_hits)
    candidates = []
    for hit in direct_hits:
        for neighbor, npmi, context in related_marks(hit, edges, top_n=5):
            if neighbor in seen:
                continue
            if require_corroboration and context:
                if not any(word in message_text for word in context):
                    continue  # no corroborating vocabulary -- don't trust this expansion
            candidates.append((neighbor, hit, npmi))
            seen.add(neighbor)
    candidates.sort(key=lambda t: -t[2])
    return candidates[:max_extra]


# --- Shadow-mode Hebbian layer -----------------------------------------
# Everything below this line computes a DYNAMIC weight that mirrors what
# the network would become if retrieval-time co-activation reinforced
# edges and idle edges decayed. It reads and writes ONLY the shadow file
# (SHADOW_DYNAMIC_PATH) and the log -- it never touches EDGES_PATH, which
# is the only file anything live would actually read from. That is the
# whole point of shadow mode: safe to run for real, on real traffic,
# with zero chance of it changing what gets retrieved today.

def record_coactivation(hits):
    """Call this once per real retrieval event with the list of marks
    that were triggered together (direct hits only, for now). Appends one
    line to the log; does not compute anything itself.

    Self-reference marks are dropped before logging (2026-09-19) -- they
    co-fire with whatever real mark happens to be in the same message
    purely because "bota"/"bot" show up in nearly every message, not
    because of any genuine relationship to that mark."""
    real_hits = [h for h in hits if h not in SELF_REFERENCE_MARKS]
    if len(real_hits) < 2:
        return  # nothing co-occurred if only one (or zero) marks fired
    entry = {"ts": time.time(), "hits": sorted(set(real_hits))}
    with open(COACTIVATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_coactivation_log():
    if not os.path.exists(COACTIVATION_LOG):
        return []
    events = []
    with open(COACTIVATION_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def run_shadow_cycle():
    """One 'tick' of the shadow-mode Hebbian update: decay everything once,
    then replay whatever co-activation events were logged since the last
    tick, reinforcing every pair that fired together. Meant to be called
    periodically (once a day, hung off the existing midnight trigger --
    not wired up yet, this just needs to be callable). Idempotent-safe:
    tracks its own last-run timestamp inside the shadow file so re-running
    it doesn't double-count events already replayed."""
    if os.path.exists(SHADOW_DYNAMIC_PATH):
        with open(SHADOW_DYNAMIC_PATH, encoding="utf-8") as f:
            shadow = json.load(f)
    else:
        shadow = {"_meta": {"last_run": 0.0}, "edges": load_edges()}

    last_run = shadow["_meta"].get("last_run", 0.0)
    edges = shadow["edges"]

    # decay every existing edge's numeric weight once per cycle (context
    # vocabulary is static, inherited from the diary build, and untouched)
    for a in edges:
        for b in edges[a]:
            edges[a][b]["npmi"] = round(edges[a][b]["npmi"] * DECAY, 6)

    # reinforce every pair that co-fired since the last cycle
    new_events = [e for e in _read_coactivation_log() if e["ts"] > last_run]
    for event in new_events:
        hits = event["hits"]
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                a, b = hits[i], hits[j]
                edges.setdefault(a, {})
                edges.setdefault(b, {})
                a_entry = edges[a].setdefault(b, {"npmi": 0.0, "context": []})
                b_entry = edges[b].setdefault(a, {"npmi": 0.0, "context": []})
                a_entry["npmi"] = round(a_entry["npmi"] + ETA, 6)
                b_entry["npmi"] = round(b_entry["npmi"] + ETA, 6)

    shadow["_meta"]["last_run"] = time.time()
    shadow["edges"] = edges
    with open(SHADOW_DYNAMIC_PATH, "w", encoding="utf-8") as f:
        json.dump(shadow, f, ensure_ascii=False, indent=1, sort_keys=True)
    return len(new_events), shadow


def _print_stats(edges):
    n_nodes = len(edges)
    n_edges = sum(len(v) for v in edges.values()) // 2
    print(f"{n_nodes} marks have at least one edge; {n_edges} undirected edges total.")
    top = []
    for a, neighbors in edges.items():
        for b, v in neighbors.items():
            if a < b:
                top.append((v["npmi"], a, b))
    top.sort(key=lambda t: -t[0])
    print("\nTop 15 strongest edges:")
    for w, a, b in top[:15]:
        print(f"  {a}  <->  {b}   NPMI={w}")


def selftest():
    print("Building static edges from diary co-occurrence ...")
    doc_freq, co_freq, co_context, n = build_cooccurrence()
    print(f"Scanned {n} diary entries, {len(doc_freq)} known marks, "
          f"{len(co_freq)} raw co-occurring pairs.")
    edges = compute_npmi(doc_freq, co_freq, co_context, n)
    _print_stats(edges)
    print(f"\n(not saved -- run --build to write {EDGES_PATH})")

    print("\n--- expand_hits demo (gate OFF, to show raw relatedness) ---")
    sample_hits = [m for m in ["坐标", "窗口", "记分"] if m in doc_freq]
    if sample_hits:
        extra = expand_hits(sample_hits, "", edges, max_extra=2,
                             require_corroboration=False)
        print(f"direct hits: {sample_hits}")
        print(f"one-hop expansion (ungated): {extra}")
    else:
        print("(sample marks not present, skipping)")

    print("\n--- regression check: 检索 -> Broker/开源 false expansion ---")
    if "检索" in doc_freq:
        bad_msg = "那你这回是怎么检索到的呢"
        result = expand_hits(["检索"], bad_msg, edges, max_extra=3)
        print(f"message: {bad_msg!r}")
        print(f"expansion with corroboration gate ON: {result} "
              f"(should be empty or near-empty)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--selftest":
        selftest()
    elif args[0] == "--build":
        doc_freq, co_freq, co_context, n = build_cooccurrence()
        edges = compute_npmi(doc_freq, co_freq, co_context, n)
        save_edges(edges)
        _print_stats(edges)
        print(f"\nSaved to {EDGES_PATH}")
    elif args[0] == "--related" and len(args) > 1:
        edges = load_edges()
        if not edges:
            print("No edges file yet -- run --build first.")
        else:
            for neighbor, npmi, context in related_marks(args[1], edges, top_n=8):
                print(f"  {neighbor}   NPMI={npmi}   context={context}")
    elif args[0] == "--shadow-cycle":
        n_events, shadow = run_shadow_cycle()
        print(f"Replayed {n_events} new co-activation events.")
        print(f"Shadow dynamic weights written to {SHADOW_DYNAMIC_PATH} "
              f"(static {EDGES_PATH} untouched).")
    else:
        print("Usage: python3 mark_network.py [--selftest | --build | "
              "--related <mark> | --shadow-cycle]")
