#!/usr/bin/env python3
"""
recall.py -- on-demand search over bota's own written record.

Why this exists
---------------
The diaries are written every night but were effectively write-only: nothing
read them again except at session start, or when Richard said "go look it up".
That gap caused a repeated, specific error -- attributing a past fact to the
wrong person (four times in three days) -- even though the right answer was
already sitting in bota_diary/.

Deliberately NOT auto-injected into every message. Pushing everyone's
attributes into every message would burn context on the many messages that
never reference the past, and would plausibly *increase* cross-wiring by
keeping confusable material permanently in view. Retrieval here is pull,
not push.

Usage
-----
  python3 recall.py "文明6"              # keyword
  python3 recall.py "健身房" --context 2  # more surrounding lines
  python3 recall.py --person gstr        # everything about one person
  python3 recall.py "MCP" --days 7       # only recent diaries
  python3 recall.py "闸门" --max 5       # cap hits

Output gives file + line number so results can be cited rather than
paraphrased -- paraphrase is where drift enters.
"""

import argparse
import datetime
import os
import re
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
DIARY_DIR = os.path.join(BASE, "bota_diary")

EXTRA_DOCS = [
    "Discord_Relay_Bot_Project_Memory.docx",
    "Debate_Decision_System_Charter.txt",
    "Yuki_Architecture_Memo.md",
    "yuki_project_watchlist.md",
    "Board_Games_Rules.txt",
    "BOTA_MANUAL.txt",
    "bota_style_notes.md",
]

PERSON_ALIASES = {
    "gstr": ["GSTR", "坤子", "gstr0686"],
    "richard": ["Richard", "richardliu"],
    "hui": ["Hui", "old_grinder_hui"],
    "yuki": ["Yuki"],
    "qaq": ["QAQ", "wyf20001"],
    "vivienne": ["Vivienne", "薇薇安", "Vivian"],
}


def read_docx(path):
    """Extract paragraph text from .docx without external deps."""
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
    except Exception:
        return []
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml).splitlines()


def load_sources(days=None):
    """Return [(label, [lines...])] for every searchable document."""
    sources = []

    if os.path.isdir(DIARY_DIR):
        names = sorted(f for f in os.listdir(DIARY_DIR) if f.endswith(".txt"))
        if days:
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            keep = []
            for n in names:
                try:
                    d = datetime.date.fromisoformat(n[:10])
                except ValueError:
                    keep.append(n)
                    continue
                if d >= cutoff:
                    keep.append(n)
            names = keep
        for n in names:
            try:
                with open(os.path.join(DIARY_DIR, n), encoding="utf-8",
                          errors="ignore") as f:
                    sources.append((f"diary/{n[:-4]}", f.read().splitlines()))
            except OSError:
                pass

    for doc in EXTRA_DOCS:
        p = os.path.join(BASE, doc)
        if not os.path.exists(p):
            continue
        if doc.endswith(".docx"):
            lines = read_docx(p)
        else:
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
            except OSError:
                continue
        sources.append((doc, lines))

    return sources


def search(sources, patterns, context=1, max_hits=20):
    """Find matching lines. Most recent diaries first, docs last."""
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    results = []

    diaries = [s for s in sources if s[0].startswith("diary/")]
    others = [s for s in sources if not s[0].startswith("diary/")]
    ordered = list(reversed(diaries)) + others

    for label, lines in ordered:
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            if any(r.search(line) for r in regexes):
                lo = max(0, i - context)
                hi = min(len(lines), i + context + 1)
                block = [(lines[j], j == i) for j in range(lo, hi)]
                results.append((label, i + 1, block))
                if len(results) >= max_hits:
                    return results
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Search bota diaries and project documents.")
    ap.add_argument("query", nargs="?", help="keyword or regex")
    ap.add_argument("--person",
                    help="gstr/richard/hui/yuki/qaq/vivienne")
    ap.add_argument("--days", type=int, help="only last N days of diaries")
    ap.add_argument("--context", type=int, default=1, help="context lines")
    ap.add_argument("--max", type=int, default=20, help="max hits")
    args = ap.parse_args()

    if args.person:
        key = args.person.strip().lower()
        pats = PERSON_ALIASES.get(key, [args.person])
        if args.query:
            joined = "|".join(re.escape(p) for p in pats)
            pats = [rf"(?=.*(?:{joined}))(?=.*{re.escape(args.query)})"]
    elif args.query:
        pats = [args.query]
    else:
        ap.error("need a query or --person")

    sources = load_sources(days=args.days)
    if not sources:
        print("no sources found -- is bota_diary/ present?")
        return

    hits = search(sources, pats, context=args.context, max_hits=args.max)

    if not hits:
        print(f"no matches. searched {len(sources)} documents.")
        print("(absence is weak evidence -- the record is lossy by design)")
        return

    print(f"{len(hits)} hit(s) across {len(sources)} documents:\n")
    for label, lineno, block in hits:
        print(f"--- {label}  (line {lineno}) ---")
        for text, is_match in block:
            print(f"{'>>' if is_match else '  '} {text}")
        print()


if __name__ == "__main__":
    main()
