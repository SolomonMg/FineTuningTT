#!/usr/bin/env python3
"""
LLM rewrite for policy-safe transcripts while preserving task signal.

- Uses flagged.csv (from check_moderation.py) to target only flagged rows.
- Rewrites ONLY Transcript/Description in the USER message:
    * Remove explicit threats/violence instructions, slurs, graphic sexual detail.
    * Keep stance cues (re China/CCP/Xi/HK/Tibet/Xinjiang/Taiwan) and organizing cues
      (protest, boycott, strike, rally, petition, vote) in safe, non-violent language.
- Drops rows flagged for sexual content involving minors (cannot safely rewrite).
- Optionally post-checks rewritten rows with moderation (--recheck).

Usage:
  export OPENAI_API_KEY="sk-..."
  python llm_rewrite_flagged.py \
    --in data/train_BAL.jsonl \
    --flagged data/flagged.csv \
    --out data/train_BAL_rewritten.jsonl \
    --dropped data/train_BAL_dropped.jsonl \
    --model gpt-4.1-mini-2025-04-14 \
    --recheck
"""
import argparse, csv, json, os, re, sys, time, random
from typing import Dict, Set, Tuple, Optional
from openai import OpenAI

TRANSCRIPT_RE  = re.compile(r"(Transcript:\s*)(.*?)(\nDescription:)", re.DOTALL)
DESCRIPTION_RE = re.compile(r"(Description:\s*)(.*?)(\nVerified:|\Z)", re.DOTALL)

URL_RE   = re.compile(r'https?://\S+', re.I)
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(?:(?:(?:\+?\d{1,3})[ .-]?)?(?:\(?\d{3}\)?)[ .-]?\d{3}[ .-]?\d{4})')

BANNED_TERMS = [
    # violence/harassment/graphic/terror words; keep broad to be safe
    "violence","violent","kill","murder","shoot","stab","attack","assault","lynch",
    "bomb","grenade","explosive","molotov","behead","rape","porn","nude","explicit sex",
    "slur","hate","harass","harassment","threat","threaten","genocide","ethnic cleansing"
]

# Words that are safe and still carry signal
ALLOWED_SIGNAL = [
    # entities/topics
    "China","PRC","CCP","Xi Jinping","Hong Kong","Tibet","Xinjiang","Uyghurs","Taiwan","DPP",
    "Tiananmen","June 4","Beijing","Shanghai","Shenzhen","censorship","corruption","sanctions",
    # lawful collective action
    "protest","boycott","strike","march","rally","petition","organize","demonstration","vote",
    # neutral evaluation words
    "criticizes","supports","praises","condemns","disputes","denies","alleges","claims",
    "policy","law","leader","government","party","election","speech","media","platform"
]

def strip_banned_terms(text: str) -> str:
    # remove banned terms case-insensitively; replace with neutral placeholders
    for w in BANNED_TERMS:
        text = re.sub(rf"(?i)\b{re.escape(w)}\b", "content removed", text)
    return text

def load_flagged(path:str) -> Dict[int, Set[str]]:
    flagged = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ln = int(row["line_number"])
            except Exception:
                continue
            cats = set(c.strip().lower() for c in (row.get("categories","") or "").split(";") if c.strip())
            flagged[ln] = cats
    return flagged

def find_user_idx(msgs):
    for i,m in enumerate(msgs):
        if m.get("role")=="user":
            return i
    return None

def should_drop_for_minors(cats:Set[str], user_text:str) -> bool:
    # If moderation categories included sexual/minors, drop.
    if any("sexual/minors" in c for c in cats):
        return True
    # Heuristic fallback:
    if re.search(r'\b(child|children|kid|kids|teen|teens|teenager|minor|underage)\b', user_text, re.I) \
       and re.search(r'\b(sex|sexual|explicit|rape|porn)\b', user_text, re.I):
        return True
    return False

def extract_td(user_text:str) -> Tuple[str,str]:
    """Extract transcript & description; return (transcript, description). Empty if missing."""
    tr = ""
    ds = ""
    m1 = TRANSCRIPT_RE.search(user_text)
    if m1:
        tr = m1.group(2)
    m2 = DESCRIPTION_RE.search(user_text)
    if m2:
        ds = m2.group(2)
    return tr, ds

def replace_td(user_text:str, new_tr:str, new_ds:str) -> str:
    def _repl_tr(m):
        return m.group(1) + new_tr + m.group(3)
    def _repl_desc(m):
        tail = m.group(3) if m.group(3) else ""
        return m.group(1) + new_ds + tail
    out = TRANSCRIPT_RE.sub(_repl_tr, user_text, count=1)
    out = DESCRIPTION_RE.sub(_repl_desc, out, count=1)
    return out

def SYSTEM_PROMPT(strict: bool) -> str:
    base = (
    "You are a safety editor. Rewrite the TikTok transcript and description so they are POLICY-COMPLIANT, "
    "while preserving only the meaning needed to classify: (a) stance toward China/its leaders, "
    "(b) whether it relates to China-sensitive topics (Taiwan, Hong Kong NSL, Tibet, Xinjiang/Uyghurs, "
    "Tiananmen/June 4, criticism of CCP/Xi, corruption), and (c) whether it encourages or portrays collective action.\n\n"
    "HARD REQUIREMENTS\n"
    "• Do NOT output any of these words or close variants: "
    "  kill, killed, killing, shoot, shot, shooting, stab, stabbing, attack, attacked, attacking, "
    "  assault, lynch, behead, bomb, grenade, explosive, molotov, rape, porn, nude, explicit sex, "
    "  violence, violent, threat, threaten, harass, harassment, slur, genocide, ethnic cleansing.\n"
    "• Remove identity-based insults/slurs and graphic sexual detail. Replace with neutral descriptors.\n"
    "• Never instruct or encourage illegal or unsafe actions. Avoid second-person imperatives (e.g., “you should…”).\n"
    "• Keep mentions of China/PRC/CCP/Xi/Hong Kong/Tibet/Xinjiang/Uyghurs/Taiwan/DPP/Tiananmen when present. "
    "  Keep lawful civic cues when present using safe language (e.g., “mentions a peaceful protest/boycott/strike/march/"
    "  rally/petition/vote”).\n"
    "• If content suggests sexual content involving minors, return {\"drop\": true} (no rewrite).\n"
    "• Keep outputs concise and neutral: each section ≤ 50 words. No added facts. No quotations.\n\n"
    "SAFE PHRASES TO USE (when applicable)\n"
    "• For negative stance: “criticizes” / “condemns” / “disputes”.\n"
    "• For positive stance: “supports” / “praises”.\n"
    "• For sensitive topics: “references politically sensitive topics”.\n"
    "• For risky/removed content: “demeaning language (removed)”, “graphic details (removed)”, "
    "  “reference to wrongdoing (removed)”, “weapon reference (removed)”.\n"
    "• For collective action: “mentions peaceful civic participation”.\n\n"
    "OUTPUT FORMAT\n"
    "Return ONLY minified JSON with these keys: "
    "{\"transcript\":\"...\",\"description\":\"...\"}. "
    "If dropping, return {\"drop\": true}. No extra text."
)
    if not strict:
        return base + 'Return ONLY JSON: {"transcript": "...","description":"..."} or {"drop": true}.'
    # STRICT mode: constrained vocabulary and short outputs
    allowed = ", ".join(ALLOWED_SIGNAL)
    banned = ", ".join(BANNED_TERMS)
    return base + (
        "STRICT MODE:\n"
        f"• Do NOT use any of these words: {banned}.\n"
        f"• Prefer these safe terms when applicable: {allowed}.\n"
        "• Summarize concisely (<= 60 words per section). Keep lawful organizing cues if present.\n"
        'Return ONLY JSON: {"transcript": "...","description":"..."} or {"drop": true}.'
    )


USER_TPL = (
    "Rewrite the following sections in a policy-compliant way while preserving stance/sensitivity/organizing cues.\n"
    "Do not include any commentary, only the JSON.\n\n"
    "TRANSCRIPT_START\n{transcript}\nTRANSCRIPT_END\n\n"
    "DESCRIPTION_START\n{description}\nDESCRIPTION_END"
)

def mask_pii(s:str) -> str:
    s = URL_RE.sub("[URL]", s)
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    return s

def call_llm(client: OpenAI, model: str, transcript: str, description: str, strict = False, max_retries=6, rpm=30.0):
    # Basic throttle across calls
    interval = 60.0 / max(1.0, rpm)
    if not hasattr(call_llm, "_last"):
        call_llm._last = 0.0
    import time
    now = time.time()
    wait = interval - (now - call_llm._last)
    if wait > 0: time.sleep(wait)

    prompt = USER_TPL.format(transcript=transcript, description=description)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content": SYSTEM_PROMPT(strict=True) if strict else SYSTEM_PROMPT(False)},
                    {"role":"user","content": prompt}
                ],
                temperature=0.1
            )
            call_llm._last = time.time()
            txt = resp.choices[0].message.content.strip()
            return txt
        except Exception:
            # simple backoff
            time.sleep(1.0 * (2 ** attempt) + random.uniform(0,0.25))
            if attempt == max_retries - 1:
                raise

def maybe_parse_json(s:str) -> Optional[dict]:
    s = s.strip()
    # allow fenced code blocks
    if s.startswith("```"):
        s = s.strip("`")
        # drop possible "json" first word
        s = re.sub(r"^json\s*", "", s)
    try:
        return json.loads(s)
    except Exception:
        # try to find the first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m: return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--flagged", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--dropped", default=None)
    ap.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--rpm", type=float, default=30.0)
    ap.add_argument("--max-rows", type=int, default=None)
    # add to argparse near other flags
    ap.add_argument("--recheck-out-csv", default=None,
                    help="If set, write flagged rows (after rewrite) to this CSV")
    ap.add_argument("--recheck-batch-size", type=int, default=8,
                    help="Batch size for moderation recheck")
    ap.add_argument("--strict", action="store_true",
                help="Use controlled-vocabulary rewrite for flagged rows")


    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    flagged = load_flagged(args.flagged)
    if not flagged:
        print("No flagged rows found; copying input unchanged.")
        os.makedirs(os.path.dirname(args.outp) or ".", exist_ok=True)
        with open(args.inp,"r",encoding="utf-8") as fi, open(args.outp,"w",encoding="utf-8") as fo:
            for line in fi: fo.write(line)
        return

    client = OpenAI()
    os.makedirs(os.path.dirname(args.outp) or ".", exist_ok=True)
    drop_fh = open(args.dropped, "w", encoding="utf-8") if args.dropped else None

    kept = changed = dropped = 0
    rewrites_for_check = []

    with open(args.inp,"r",encoding="utf-8") as fi, open(args.outp,"w",encoding="utf-8") as fo:
        for lineno, line in enumerate(fi, start=1):
            if args.max_rows and lineno > args.max_rows: break
            s = line.strip()
            if not s: continue

            # Not flagged → copy through
            if lineno not in flagged:
                fo.write(line); kept += 1
                continue

            try:
                obj = json.loads(s)
                msgs = obj.get("messages", [])
                ui = find_user_idx(msgs)
                if ui is None:
                    # no user → keep (rare)
                    fo.write(line); kept += 1
                    continue
                user_txt = msgs[ui].get("content","")
                if should_drop_for_minors(flagged[lineno], user_txt):
                    dropped += 1
                    if drop_fh: drop_fh.write(line)
                    continue

                tr, ds = extract_td(user_txt)
                tr, ds = mask_pii(tr), mask_pii(ds)

                llm_out = call_llm(client, args.model, tr, ds, strict=args.strict, rpm=args.rpm)
                data = maybe_parse_json(llm_out)
                if not data or (("drop" in data) and data.get("drop") is True):
                    # drop if model asked to drop
                    dropped += 1
                    if drop_fh: drop_fh.write(line)
                    continue

                new_tr = (data.get("transcript") or "").strip()
                new_ds = (data.get("description") or "").strip()
                if args.strict:
                    new_tr = strip_banned_terms(new_tr or "")
                    new_ds = strip_banned_terms(new_ds or "")

                if not new_tr and not new_ds:
                    # fallback: keep original (safer for recall) or drop?
                    # Here we drop to avoid another moderation failure
                    dropped += 1
                    if drop_fh: drop_fh.write(line)
                    continue

                new_user = replace_td(user_txt, new_tr or tr, new_ds or ds)
                msgs[ui]["content"] = new_user
                obj["messages"] = msgs
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1; changed += 1
                rewrites_for_check.append((lineno, new_user))

            except Exception:
                dropped += 1
                if drop_fh: drop_fh.write(line)

    if drop_fh: drop_fh.close()
    print(f"Rewrite complete. Kept={kept} Rephrased={changed} Dropped={dropped}")
    print(f"Output → {args.outp}")
    if args.dropped:
        print(f"Dropped → {args.dropped}")

    if args.recheck and rewrites_for_check:
        try:
            from math import ceil
            interval = 60.0 / max(1.0, args.rpm)
            last = 0.0
            flagged_after = 0

            # optional CSV
            csvw = None
            if args.recheck_out_csv:
                os.makedirs(os.path.dirname(args.recheck_out_csv) or ".", exist_ok=True)
                csvf = open(args.recheck_out_csv, "w", newline="", encoding="utf-8")
                import csv as _csv
                csvw = _csv.writer(csvf)
                csvw.writerow(["line_number", "categories", "user_excerpt"])

            for i in range(0, len(rewrites_for_check), args.recheck_batch_size):
                chunk = rewrites_for_check[i:i+args.recheck_batch_size]
                texts = [t for (_, t) in chunk]

                # throttle
                now = time.time()
                wait = interval - (now - last)
                if wait > 0:
                    time.sleep(wait)

                # call moderation (with a simple retry)
                for attempt in range(6):
                    try:
                        resp = client.moderations.create(
                            model="omni-moderation-latest",
                            input=texts
                        )
                        last = time.time()
                        break
                    except Exception:
                        time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.25))
                        if attempt == 5:
                            raise

                # handle results
                for (ln, txt), res in zip(chunk, resp.results):
                    if bool(getattr(res, "flagged", False)):
                        flagged_after += 1
                        if csvw:
                            cats = []
                            cats_dict = getattr(res, "categories", None)
                            if isinstance(cats_dict, dict):
                                cats = sorted([k for k, v in cats_dict.items() if v])
                            excerpt = (txt[:200] + "…") if len(txt) > 200 else txt
                            csvw.writerow([ln, ";".join(cats), excerpt])

            if csvw:
                csvf.close()
                print(f"Recheck CSV → {args.recheck_out_csv}")

            print(f"Recheck on rewritten rows: {flagged_after} still flagged out of {len(rewrites_for_check)}")

        except Exception as e:
            print(f"[warn] recheck failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
