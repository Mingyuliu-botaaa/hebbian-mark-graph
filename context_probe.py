#!/usr/bin/env python3
"""
context_probe.py -- decide whether an incoming message references the past,
and if so, pull a few dated references out of the diaries.

Standalone on purpose. NOT wired into mention_listener.py yet.
Run it against sample messages first; only if the noise is acceptable
does it get attached to the live relay.

Design comes from the 2026-09-16 debate (Bota vs Yuki, see
Debate_Decision_System_Charter.txt). Binding constraints from that debate:

  1. Fail toward silence. No trigger, or weak evidence -> attach nothing.
     The default is NOT "inject something anyway".
  2. 3 entries is a CEILING, not a quota. Fewer is fine. Zero is fine.
  3. Every entry carries date + source, so a retrieved line can never be
     mistaken for something remembered.
  4. Zero model calls. This is a cheap pre-filter; if it ever needs
     inference it has lost its reason to exist.
  5. One switch turns the whole thing off.

Usage:
  python3 context_probe.py "那个健身房的事"      # single message
  python3 context_probe.py --selftest            # run the built-in suite
"""

import os
import re
import sys

ENABLED = True          # constraint 5: the off switch
MAX_ENTRIES = 3         # constraint 2: ceiling, not target
MAX_CHARS = 400
EXCLUDE_RECENT_DAYS = 1  # today's diary is still in live context         # shared budget across all entries

# Marks that are just how people address the bot, not a topic -- see
# probe()'s self-reference short-circuit, added 2026-09-18.
SELF_REFERENCE_MARKS = {"bota", "bot", "BOT"}

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# --- Tier 1: proper nouns (highest precision) -------------------------
# In this channel these are almost never ambiguous. A message containing
# "浮士德" is not plausibly about something else.
PROPER_NOUNS = [
    "坤子", "GSTR", "Hui", "QAQ", "薇薇安", "Vivienne", "Yuki",
    "VASP", "MCP", "Codex", "stockbot", "STOCK BOT",
    "文明6", "浮士德", "五子棋", "过河棋", "生长棋", "重构棋", "井字棋",
    "荒城之月", "辩论决策", "黑曜石", "中子星", "SHA-256",
]

# --- Tier 2: distal demonstrative + measure word / noun ---------------
# 那 is referential ONLY when followed by a measure word or noun.
# Bare 那 is usually a discourse connective and carries no reference:
#   那我们开始吧 / 那么 / 那就 / 那样  <- all noise
DEMONSTRATIVE_RE = re.compile(
    r"那(?:个|件|次|条|种|份|张|台|本|篇|场|局|段|套|批|回|天|年|阵)"
)
# Explicit blocklist, checked first -- cheaper than tuning the regex.
DEMONSTRATIVE_BLOCK = ["那么", "那就", "那样", "那边", "那里", "那时"]

# --- Tier 3: explicit recall phrases (supplementary) ------------------
RECALL_PHRASES = [
    "还记得", "记不记得", "记得吗", "上次", "上回", "之前", "当时",
    "前几天", "那天", "以前", "原来说", "你说过", "我说过",
]


def load_marks():
    """Marks are the trigger vocabulary. A term is only here if it has
    diary content behind it, so a hit can never produce an empty result."""
    path = os.path.join(BASE, "marks.json")
    if not os.path.exists(path):
        return {}
    try:
        import json
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


_LATIN_MARK_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _is_latin_mark(mark):
    """True for marks made entirely of Latin letters/digits/._- (the same
    class build_marks.py's auto-extraction pulls out). Chinese has no
    space-based word segmentation, so substring hits there are intentional
    -- boundary-checking only applies to this Latin subset."""
    return bool(_LATIN_MARK_RE.match(mark))


def _word_boundary_hit(mark, text):
    """Richard's fix (2026-09-19) for 'bot' matching inside 'bota': a
    Latin-script mark only counts as hit if BOTH the character right
    before and right after the match are non-letters (or the match is at
    the very start/end of the string). Checking only one side misses half
    the problem -- 'bot' at the END of 'robot' has a clean trailing
    boundary but a dirty leading one, so both sides must be checked."""
    for m in re.finditer(re.escape(mark), text, flags=re.IGNORECASE):
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        before_is_letter = bool(re.match(r"[A-Za-z]", before)) if before else False
        after_is_letter = bool(re.match(r"[A-Za-z]", after)) if after else False
        if not before_is_letter and not after_is_letter:
            return True
    return False


def _mark_hit(mark, text):
    """Dispatch: word-boundary check for Latin marks, plain substring
    check for everything else (Chinese)."""
    if _is_latin_mark(mark):
        return _word_boundary_hit(mark, text)
    return mark in text


def detect_trigger(text: str):
    """Return (triggered, reason, matched_marks)."""
    if not ENABLED or not text or not text.strip():
        return False, "disabled or empty", []
    marks = load_marks()
    hits = [m for m in marks if _mark_hit(m, text)]
    if not hits:
        return False, "no mark", []
    hits.sort(key=len, reverse=True)          # prefer the specific ones
    top_hits = hits[:4]
    # Shadow-mode data collection only (2026-09-17): log which marks fired
    # together so mark_network's Hebbian layer has real usage data to learn
    # from later. This does NOT change what gets retrieved or returned --
    # it only ever writes to the shadow log, never to anything read here.
    try:
        import mark_network
        mark_network.record_coactivation(top_hits)
    except Exception:
        pass  # logging must never be able to break the actual probe
    return True, f"mark:{','.join(top_hits)}", top_hits


STOPWORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它", "们", "这", "那",
    "有", "在", "和", "跟", "就", "都", "也", "不", "没", "很", "么",
    "呢", "吗", "啊", "吧", "把", "被", "让", "给", "对", "从", "到",
    "个", "会", "要", "过", "还", "又", "再", "去", "来", "说", "做",
}


def _content_terms(text: str):
    """Pull candidate search terms out of the message.

    No segmenter available, so: strip punctuation, drop single-char
    stopwords, and take 2-char sliding windows. Junk windows are harmless
    because we SCORE rather than require -- noise scores 0 and sinks.
    """
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", text)
    terms = set()
    for chunk in cleaned.split():
        if re.match(r"^[A-Za-z0-9]+$", chunk):      # latin/number token
            if len(chunk) >= 2:
                terms.add(chunk)
            continue
        chars = [c for c in chunk if c not in STOPWORDS]
        for i in range(len(chars) - 1):             # 2-char windows
            terms.add(chars[i] + chars[i + 1])
    return terms


def fetch_references(terms, message="", max_entries=MAX_ENTRIES,
                     max_chars=MAX_CHARS, min_score=4.0):
    """Score diary lines against the message; return the best few.

    Rarity weighting is the important part. Without it, common bigrams
    like 结果 / 怎样 appear everywhere and inflate the score of lines that
    have nothing to do with the question -- which is exactly how the
    first version returned three unrelated Yuki lines when asked about a
    chess game. A term that occurs in many lines carries almost no
    information, so its weight decays toward zero.
    """
    try:
        import recall
    except Exception as e:
        return [f"[probe] recall.py unavailable: {e}"]

    import math
    content = _content_terms(message)
    if not terms and not content:
        return []

    sources = recall.load_sources()
    if EXCLUDE_RECENT_DAYS:
        import datetime
        cutoff = (datetime.date.today()
                  - datetime.timedelta(days=EXCLUDE_RECENT_DAYS)).isoformat()
        sources = [(l, ls) for l, ls in sources
                   if not (l.startswith('diary/') and l[6:16] > cutoff)]
    all_lines = []
    for label, lines in sources:
        for lineno, line in enumerate(lines, 1):
            if len(line.strip()) >= 12:
                all_lines.append((label, lineno, line))
    if not all_lines:
        return []

    # document frequency for every candidate term
    vocab = set(terms) | content
    df = {t: 0 for t in vocab}
    for _, _, line in all_lines:
        for t in vocab:
            if t in line:
                df[t] += 1

    total = len(all_lines)
    weight = {}
    for t in vocab:
        if df[t] == 0:
            weight[t] = 0.0
        else:
            idf = math.log(total / df[t])          # rare term -> big idf
            base = 2.0 if t in terms else 1.0      # proper nouns count double
            weight[t] = base * idf

    scored = []
    for label, lineno, line in all_lines:
        score = sum(weight[t] for t in vocab if weight[t] > 0 and t in line)
        if score >= min_score:
            scored.append((score, label, lineno, line.strip()))

    scored.sort(key=lambda x: -x[0])

    out, used = [], 0
    for score, label, lineno, text in scored:
        if len(out) >= max_entries:
            break
        text = re.sub(r"\*\*", "", text)
        if len(text) > 110:
            text = text[:110] + "..."
        date = label.replace("diary/", "")
        entry = f"〔检索〕{date}：{text}"
        if used + len(entry) > max_chars:
            break
        if entry not in out:
            out.append(entry)
            used += len(entry)
    return out


def probe(text: str):
    """Main entry point. Returns '' when nothing should be attached."""
    triggered, reason, terms = detect_trigger(text)
    if not triggered:
        return ""

    # Self-reference short-circuit (added 2026-09-18, Richard's fix for the
    # "botaaaa" false positive): "bota"/"bot"/"BOT" are just how people
    # address me, so they're extremely common and carry no topical
    # information -- running the normal scored search on them pulls back
    # essentially random diary lines. If they're the ONLY thing that
    # matched, skip the search entirely and say so in one line instead of
    # attaching noise. If a genuine mark matched alongside them, drop only
    # the self-reference marks from the search terms and let the real
    # mark(s) drive retrieval as usual.
    real_terms = [t for t in terms if t not in SELF_REFERENCE_MARKS]
    if not real_terms:
        return ("〔提示〕这条消息里唯一命中的标注词是我自己的称呼（bota/bot），"
                "大概率只是在叫我，不代表消息在问关于我的历史内容，未做检索。")

    refs = fetch_references(real_terms, message=text)
    if not refs:
        return ""          # constraint 1: triggered but no evidence -> silence
    header = ("〔以下为脚本从旧日记检索的参考，可能与当前问题无关；"
              "当前对话内容优先，冲突时以眼前的消息为准〕")
    return header + "\n" + "\n".join(refs)


# --- self test -------------------------------------------------------
# SHOULD_FIRE: genuinely references something in the past.
# SHOULD_NOT:  ordinary chatter, incl. the 那-as-connective traps.
SHOULD_FIRE = [
    "那个健身房的事是谁说的来着",
    "你还记得浮士德吗",
    "上次跟Yuki下的那局棋结果怎样",
    "坤子之前提的那个MCP方案",
    "文明6那次神难度你用的哪个文明",
    "我们之前聊过的中子星",
]
SHOULD_NOT = [
    "botaaaa",
    "什么事！",
    "那我们开始吧",
    "那么先这样",
    "那就这么定了",
    "那样不行",
    "今天吃饭了吗",
    "我有点困了",
    "帮我写个脚本",
    "🤖✨",
]


def selftest():
    print("=" * 56)
    print("SHOULD FIRE")
    print("=" * 56)
    tp = 0
    for m in SHOULD_FIRE:
        fired, reason, terms = detect_trigger(m)
        tp += fired
        print(f"{'✅' if fired else '❌ MISS'}  {m}")
        print(f"      → {reason}")
    print()
    print("=" * 56)
    print("SHOULD NOT FIRE")
    print("=" * 56)
    tn = 0
    for m in SHOULD_NOT:
        fired, reason, terms = detect_trigger(m)
        tn += (not fired)
        print(f"{'✅' if not fired else '❌ FALSE POSITIVE'}  {m}")
        if fired:
            print(f"      → {reason}")
    print()
    print(f"recall  : {tp}/{len(SHOULD_FIRE)} fired when they should")
    print(f"precision: {tn}/{len(SHOULD_NOT)} stayed silent when they should")
    print()
    print("=" * 56)
    print("END-TO-END (trigger + search + format)")
    print("=" * 56)
    for m in SHOULD_FIRE[:3]:
        print(f"\n>>> {m}")
        out = probe(m)
        print(out if out else "  (triggered but no evidence -> attached nothing)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--selftest":
        selftest()
    else:
        result = probe(args[0])
        print(result if result else "(nothing attached)")
