# Mark Network: Lightweight Long-Term Memory Retrieval for Conversational Agents

## What is this

A memory retrieval system for a chat agent that talks to people over long periods of time — no vector database, no embedding model required. The agent reads a history of past logs (diary entries, chat transcripts, anything text-based), tags recurring topics as they come up, and automatically computes which topics are related to each other using pure statistics (NPMI + Hebbian reinforcement). The result behaves like long-term memory: when a topic resurfaces, the agent can recall related things it discussed before — but every number in the system is auditable by hand, nothing is a black box.

## Why this exists

The mainstream approach to agent long-term memory is a vector database: embed every chunk of text, retrieve by semantic similarity. This project goes the opposite direction — build a "topic relationship graph" using the simplest possible statistics (pointwise mutual information, Hebbian learning). You trade away some semantic generalization for something that costs zero extra model calls and where any retrieved result can be manually traced back to exactly why it fired. Best suited for situations where explainability matters more than semantic precision.

## How it works

**Two layers, kept separate:**
- **Static layer (NPMI)** — a relatively stable graph computed once from historical text.
- **Dynamic layer (Hebbian, shadow mode)** — strengthens/decays as real usage happens; starts in log-only mode with zero effect on live retrieval.

The two layers are never allowed to contaminate each other — historical text should never be backfilled as "dynamic co-activation events," or the same evidence gets double-counted in both layers.


### Nodes and edges (static layer)

**Nodes (marks)** — Chinese topic words must be tagged manually; there's no shortcut. Fully-automatic keyword extraction from raw text has already been tried and failed here (a sliding-window approach produced almost entirely grammatical fragments, no real topics). Automatic extraction only works reliably for two narrow categories: text inside bracket/quote delimiters, and Latin-alphabet tokens (3-20 characters) — both are mechanical, no understanding required.

**Edges (relatedness)** — Never use raw co-occurrence counts as the weight; that biases toward high-frequency words looking related to everything. Use NPMI (normalized pointwise mutual information) instead:
- `PMI(A,B) = log( P(A,B) / (P(A)×P(B)) )`
- `NPMI(A,B) = PMI(A,B) / -log(P(A,B))`, range `[-1, 1]`; 1 = always co-occur, 0 = independent, negative = mutually exclusive
- When `P(A,B) = 1` (co-occur every time), NPMI is defined as 1 directly, avoiding division by zero
- Only edges with NPMI > 0 are kept; negative values aren't useful signal
- Zero model calls — this entire step is counting and `log()`, no inference needed anywhere

### Disambiguation: what does this word mean *right now*

Pure co-occurrence statistics run into a real problem: if a word gets heavily used recently to discuss "the system we're currently building," its strongest NPMI neighbors become whatever that design conversation was about — completely disconnected from the word's normal meaning (a real incident here: the word "retrieval" got polluted by a multi-day conversation about building a *retrieval system*, so an unrelated question containing that word started expanding into irrelevant architecture-discussion terms).

**Fix: store each edge's source context.** Every edge keeps not just its NPMI value but also which *other* marks co-occurred in the historical entries that produced it (reusing the existing mark vocabulary, no new lookup table). When expanding one hop, only trust the expansion if the current input shares vocabulary with that edge's source context. Edges with empty context (e.g. freshly created by the dynamic layer) pass through unfiltered by default, since there's no evidence yet to check against.

### Self-reference short-circuit

An agent's own name/address gets mentioned constantly (pure greetings), and treating it like a normal mark means near-constant, meaningless retrieval triggers.

**Fix:** collect the agent's own name/aliases into one small set. When checking a hit:
- If **all** matched marks belong to this set, skip the normal retrieval pipeline entirely and return one fixed line ("this message is probably just addressing me, not asking about my history") — cleaner than returning irrelevant search results.
- If a **real** mark is also present alongside a self-reference mark, drop the self-reference term from the search terms and let the real mark drive retrieval as usual — a self-reference hit shouldn't suppress a genuinely informative neighboring mark.

### Dynamic layer: Hebbian reinforcement + decay, shadow mode first

**Trigger:** whenever 2+ marks are hit together in a real retrieval event, log one "co-activation" entry — no judgment about whether that trigger was actually useful, pure statistics, no reward signal needed.

**Update rule (leaky Hebbian rule):**
- On each co-activation: `w(A,B) += η` (a small fixed constant, e.g. 1)
- Once per cycle: `w(A,B) *= λ` (decay factor < 1, e.g. 0.99)
- This has a clean convergence property: if A and B co-occur at a steady rate `r`, the weight converges to `w* = η×r / (1-λ)` — it won't grow unbounded, and edges that only fired once or twice and never again decay back toward zero.

**Shadow mode is the whole point at first:** this layer writes to its own separate file (never overwrites the static layer's edges), only logs and computes weights, and is completely disconnected from live retrieval ranking. Let it run against real usage for a while, review where the weights land, before ever deciding whether to wire it into retrieval.

## Known pitfalls — don't re-learn these the hard way

1. **NPMI is naturally inflated for low-frequency marks.** Two marks that each appear only once, happening to co-occur that one time, get NPMI = 1.0 immediately — this isn't a bug, it's a known weakness of the whole PMI family (rare events statistically inflate easily). If too many of these "perfect but n=1" edges show up, add a minimum co-occurrence threshold.
2. **Never backfill historical data as dynamic-layer co-activation events.** Covered above, but worth repeating — this is the easiest mistake to make and the easiest to overlook the consequences of.
3. **Off-by-one dates around midnight triggers.** When tagging happens right after some "write today's summary" trigger fires, "today" from the trigger's perspective is usually the day that just *ended*, not the day the trigger fired on. Get this backwards and every check built on top of mark dates will silently fail.
4. **One-hop expansion only — resist adding multi-hop early.** Multi-hop turns "how many hops" into a new, hard-to-tune hyperparameter, and wider spread means more noise. Validate that one hop is useful before considering more.
5. **Rich-get-richer risk in the dynamic layer.** Hebbian reinforcement + global decay means frequently-co-firing pairs keep compounding while rare-but-real associations can get crowded to the margins. If the system starts "only remembering whatever was discussed most recently," a floor weight for low-frequency-but-real marks may be needed. Not yet implemented here — flagged for whoever picks this up next.
6. **Substring-related marks pollute the graph for free.** If one mark is literally a substring of another (e.g. an agent's short name inside its longer name), they will co-fire on nearly every message mechanically, not because they're actually related — the observed real-world case here pushed a Hebbian weight to ~33x NPMI's theoretical [-1, 1] bound. Either exclude such pairs from the graph entirely, or require word-boundary matching for Latin-script marks (check *both* the character before and after a match — checking only one side still lets substrings at the other edge of a word slip through).

## Files in this repo

- `context_probe.py` — main entry point; decides whether a message should trigger memory retrieval and how
- `mark_network.py` — builds the mark relationship graph (static NPMI layer + dynamic Hebbian shadow layer)
- `recall.py` — low-level text search that scans the historical log files

## Prerequisites

**1. `marks.json`** — the mark vocabulary, JSON format:
```json
{
  "SomeTopicWord": {"first_seen": "2026-09-18", "diaries": 0, "count": 0, "source": "manual"},
  "SomeEnglishMark": {"first_seen": "2026-09-19", "diaries": 0, "count": 0, "source": "auto"}
}
```
`source` distinguishes `"manual"` (hand-tagged; required for topic words with no natural delimiters) from `"auto"` (extracted automatically from bracketed spans or Latin tokens).

**2. A directory of historical text files** (one file per day works well) — both `recall.py` and `mark_network.py` scan this directory to compute co-occurrence.

## CLI reference

```bash
# Add a new manual mark (required for topic words that can't be auto-extracted)
python3 build_marks.py --add "some new topic"
# When backfilling a mark for a *previous* day (see pitfall #3 above), always pass --date explicitly
python3 build_marks.py --add "some topic" --date 2026-09-18

# Rescan all historical text, refresh auto-marks, rebuild the co-occurrence graph
# (run this any time marks.json changes)
python3 mark_network.py --build

# Inspect a mark's one-hop neighbors (debugging)
python3 mark_network.py --related "some mark"

# Shadow mode: replay new co-activation log entries into Hebbian weights
# (does not touch live retrieval)
python3 mark_network.py --shadow-cycle

# Run the built-in recall/precision self-test — always run this after any code change
python3 context_probe.py --selftest
```

## Integrating into your own agent

There is exactly one function you need to call, right before generating a reply:

```python
import context_probe

reference_text = context_probe.probe(the_users_raw_message)
if reference_text:
    # Treat this as reference material, not a binding instruction
    prompt = reference_text + "\n\n" + original_prompt
```

`probe()` internally handles: deciding whether to trigger, how many results to return, whether to return the self-reference short-circuit note, and the total character budget. You should not call `detect_trigger()` or `fetch_references()` directly — those are internal implementation details.

The return value is always one of two things:
- `""` — nothing triggered, or it triggered but found nothing useful; do nothing
- A non-empty string — already formatted with a header, ready to append to your prompt as-is

## Configuration (top of `context_probe.py`)

- `ENABLED` — master switch; set `False` to disable globally (useful while debugging)
- `MAX_ENTRIES` — max reference entries per response (a ceiling, not a target — don't pad to reach it)
- `MAX_CHARS` — total character budget across all returned entries
- `SELF_REFERENCE_MARKS` — set this to your own agent's name/aliases before deploying

## Before you deploy

1. Every hardcoded path in these files (e.g. `marks.json`, the log directory) needs to point at your own data — search and replace before running.
2. Run `python3 context_probe.py --selftest` and sanity-check the recall/precision numbers *before* wiring this into a live conversation flow. Don't skip this step.

## License

MIT — see LICENSE.

---
Originally designed and validated inside a Discord relay bot project, September 2026.
