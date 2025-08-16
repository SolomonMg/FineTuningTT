
"""
Usage: 

Toy run 

python build_finetune_jsonl.py \
    --labels-csv data/china_labeling_sample_all_Jul30_proc.csv \
    --meta-parquet data/china_labeling_sample_all_with_caption.parquet \
    --out-jsonl train_SMALL.jsonl \
    --val-jsonl val_SMALL.jsonl \
    --val-size 5 \
    --seed 7 \
    --limit 10


Actual run: 

python build_finetune_jsonl.py \
    --labels-csv data/china_labeling_sample_all_Jul30_proc.csv \
    --meta-parquet data/china_labeling_sample_all_with_caption.parquet \
    --out-jsonl train_SMALL.jsonl \
    --val-jsonl val_SMALL.jsonl \
    --val-size 100 \
    --seed 7

"""

#!/usr/bin/env python3
# DuckDB → JSONL (adds train/val split, tiny validator)
import argparse, json, duckdb, random

SYSTEM_MSG = (
  "You are a meticulous labeling assistant for TikTok videos. "
  "Follow the CODEBOOK and output format exactly.\n\n"
  "CODEBOOK — DEFINITIONS & TASKS\n"
  "CHINA-SENSITIVE ISSUES include: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
  "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
  "of Xi Jinping or the CCP; corruption in China.\n\n"
  "LABEL THE VIDEO ON FOUR DIMENSIONS:\n"
  "1) china_stance — 'good' | 'bad' | 'neutral' | 'not_related' | 'cannot_determine'\n"
  "2) china_sensitive — 'yes' | 'no' | 'cannot_determine'\n"
  "3) collective_action — 'yes' | 'no' | 'cannot_determine'\n"
  "4) languages — ['english','mandarin','spanish','other','no_language']\n\n"
  "FORMAT RULES\n"
  "• Output ONLY a minified JSON object with keys: china_stance, china_sensitive, collective_action, languages.\n"
  "• If 'no_language' is present, it MUST be the only item in languages.\n"
  "• Use 'cannot_determine' when unsure. Do not add extra keys or prose."
)

USER_TPL = (
  "Transcript: {transcript}\n"
  "Description: {description}\n"
  "Verified: {verified}\n"
  "Followers: {followers}\n"
  "Hearts: {hearts}\n"
  "Likes: {likes}\n"
  "Country: {country}\n"
  "Music Title: {music_title}\n"
  "Music Author: {music_author}\n"
  "POL: {pol}\n"
  "Created At: {create_time}\n"
  "Location Created: {loc_created}\n\n"
  "Return JSON only."
)

# --- tiny helpers ---
def truthy(v):
    s = str(v).strip().lower()
    return s in {"1","true","t","yes","y"} or (s.isdigit() and int(s) != 0)

def norm_stance(x):
    if x is None: return "cannot_determine"
    s = str(x).strip().lower()
    if s in {"good","bad","neutral","not_related","cannot_determine"}: return s
    if s in {"not related","not related to china"}: return "not_related"
    if s in {"could not determine","cnd","unknown","n/a"}: return "cannot_determine"
    return "cannot_determine"

def norm_yn(x):
    if x is None: return "cannot_determine"
    s = str(x).strip().lower()
    if s in {"yes","no","cannot_determine"}: return s
    if s in {"y","1","true"}: return "yes"
    if s in {"n","0","false"}: return "no"
    if s in {"cnd","unknown","could not determine","n/a"}: return "cannot_determine"
    return "cannot_determine"

def build_languages(row):
    langs=[]
    for col,outn in [("english","english"),("mandarin","mandarin"),
                     ("spanish","spanish"),("other_lang","other"),("no_language","no_language")]:
        if truthy(row.get(col)): langs.append(outn)
    if not langs: langs=["no_language"]
    if "no_language" in langs: return ["no_language"]
    order={"english":0,"mandarin":1,"spanish":2,"other":3,"no_language":4}
    return sorted(langs, key=lambda x: order.get(x,99))

# --- tiny validator (cheap but useful) ---
ALLOWED_STANCE = {"good","bad","neutral","not_related","cannot_determine"}
ALLOWED_YNCD   = {"yes","no","cannot_determine"}
ALLOWED_LANG   = {"english","mandarin","spanish","other","no_language"}
def validate_labels(obj):
    if set(obj.keys()) != {"china_stance","china_sensitive","collective_action","languages"}:
        return "unexpected keys"
    if obj["china_stance"] not in ALLOWED_STANCE: return "bad china_stance"
    if obj["china_sensitive"] not in ALLOWED_YNCD: return "bad china_sensitive"
    if obj["collective_action"] not in ALLOWED_YNCD: return "bad collective_action"
    langs = obj["languages"]
    if not isinstance(langs, list): return "languages not a list"
    if any(l not in ALLOWED_LANG for l in langs): return "unknown language"
    if len(set(langs)) != len(langs): return "duplicate languages"
    if "no_language" in langs and len(langs) != 1: return "no_language must be only item"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--meta-parquet", required=True)
    # train/val outputs
    ap.add_argument("--out-jsonl", required=True, help="train output (or overall if no val)")
    ap.add_argument("--val-jsonl", default=None, help="optional validation output")
    # split controls
    ap.add_argument("--train-ratio", type=float, default=1.0, help="ignored if --val-jsonl not set; else 0<r<=1")
    ap.add_argument("--val-size", type=int, default=None, help="take exactly N for val (overrides train-ratio)")
    ap.add_argument("--seed", type=int, default=42)
    # data controls
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--labels-video-col", default="video")
    args = ap.parse_args()

    con = duckdb.connect()

    sql = f"""
    WITH L AS (
      SELECT
        regexp_extract(l.{args.labels_video_col}, '([0-9]{{10,}})\\.mp4', 1) AS meta_id,
        l.china_stance_score, l.sensitive, l.collective_action,
        l.english, l.mandarin, l.spanish, l.other_lang, l.no_language
      FROM read_csv_auto(?, HEADER=TRUE, ALL_VARCHAR=TRUE, IGNORE_ERRORS=TRUE, SAMPLE_SIZE=-1) AS l
      WHERE {args.labels_video_col} IS NOT NULL
    ),
    M AS (
      SELECT
        CAST(m.meta_id AS VARCHAR) AS meta_id,
        m.meta_locationCreated, m.meta_createTime,
        m.author_verified, m.authorstats_followerCount,
        m.authorstats_heartCount, m.authorstats_diggCount,
        m.country, m.music_title, m.music_authorName, m.pol,
        m.meta_desc, m.subtitle
      FROM read_parquet(?) AS m
    )
    SELECT
      L.meta_id,
      L.china_stance_score, L.sensitive, L.collective_action,
      L.english, L.mandarin, L.spanish, L.other_lang, L.no_language,
      M.meta_locationCreated, M.meta_createTime,
      M.author_verified, M.authorstats_followerCount,
      M.authorstats_heartCount, M.authorstats_diggCount,
      M.country, M.music_title, M.music_authorName, M.pol,
      M.meta_desc, M.subtitle
    FROM L JOIN M USING (meta_id)
    """
    if args.limit: sql += f" LIMIT {int(args.limit)}"

    cur = con.execute(sql, [args.labels_csv, args.meta_parquet])
    data = cur.fetchnumpy()
    cols = list(data.keys())
    n = len(data[cols[0]]) if cols else 0

    # build all rows once (keeps code simple; fine for small/medium sets)
    rows = []
    skipped = 0
    for i in range(n):
        row = {c: data[c][i] for c in cols}
        labels = {
            "china_stance":      norm_stance(row.get("china_stance_score")),
            "china_sensitive":   norm_yn(row.get("sensitive")),
            "collective_action": norm_yn(row.get("collective_action")),
            "languages":         build_languages(row),
        }
        if (err := validate_labels(labels)) is not None:
            skipped += 1
            continue
        assistant = json.dumps(labels, separators=(",",":"), ensure_ascii=False)
        user_msg = USER_TPL.format(
            transcript    = (row.get("subtitle") or ""),
            description   = (row.get("meta_desc") or ""),
            verified      = ("Yes" if truthy(row.get("author_verified")) else "No"),
            followers     = str(int(float(row.get("authorstats_followerCount") or 0))),
            hearts        = str(int(float(row.get("authorstats_heartCount") or 0))),
            likes         = str(int(float(row.get("authorstats_diggCount") or 0))),
            country       = (row.get("country") or ""),
            music_title   = (row.get("music_title") or ""),
            music_author  = (row.get("music_authorName") or ""),
            pol           = (row.get("pol") or ""),
            create_time   = (row.get("meta_createTime") or ""),
            loc_created   = (row.get("meta_locationCreated") or "")
        )
        rows.append({
            "messages":[
                {"role":"system","content": SYSTEM_MSG},
                {"role":"user","content": user_msg},
                {"role":"assistant","content": assistant}
            ]
        })

    if not rows:
        raise SystemExit("No valid rows to write (all skipped?).")

    # split
    if args.val_jsonl:
        rng = random.Random(args.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        if args.val_size and args.val_size > 0:
            v = min(args.val_size, len(rows))
            val_idx = set(idx[:v])
            train_idx = [i for i in idx if i not in val_idx]
        else:
            cut = int(len(rows) * float(args.train_ratio))
            train_idx, val_idx = idx[:cut], set(idx[cut:])
        with open(args.out_jsonl, "w", encoding="utf-8") as ft:
            for i in train_idx: ft.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        with open(args.val_jsonl, "w", encoding="utf-8") as fv:
            for i in val_idx: fv.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print(f"Wrote {len(train_idx)} → {args.out_jsonl}; {len(val_idx)} → {args.val_jsonl}; skipped {skipped}")
    else:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for obj in rows: f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} → {args.out_jsonl}; skipped {skipped}")

if __name__ == "__main__":
    main()

