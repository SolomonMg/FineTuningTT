#!/usr/bin/env python3
"""
Pre-split/balancer for labeled TikTok data (labels only; no metadata join).
Creates TRAIN and VAL that are more balanced per joint label cell:
(china_stance_score × sensitive × collective_action).

Author: Sol Messing

What this script does
  - Reads labels CSV with DuckDB (no pandas).
  - Derives meta_id from the video filepath (.../<digits>.mp4 → <digits>).
  - Computes china_stance_score via argmax over {china_good, china_bad, china_neutral, china_not_related, china_cnd}.
  - Normalizes binary labels to {yes, no, cannot_determine}: sensitive, collective_action.
  - Passes language flags through: english, mandarin, spanish, other_lang, no_language.
  - Deterministically ranks within each joint label cell using hash(meta_id || seed).
  - Allocates rows to TRAIN and VAL by ratio + per-cell caps, with guarantees:
      - Per-cell caps applied independently to TRAIN and VAL.
      - If a cell has c ≥ 2, both splits receive ≥ 1 item (subject to VAL cap).
  - Writes two CSVs you can feed directly to build_finetune_jsonl.py.

Inputs
  - labels CSV (e.g., data/china_labeling_sample_all_Jul30.csv)
    Required columns: video (or pass --labels-video-col), china_good, china_bad,
                      china_neutral, china_not_related, china_cnd, sensitive, collective_action
    Optional columns: english, mandarin, spanish, other_lang, no_language

Outputs (builder-ready columns)
  - TRAIN/VAL CSVs with columns:
      meta_id, china_stance_score, sensitive, collective_action,
      english, mandarin, spanish, other_lang, no_language

Requirements
  pip install duckdb

Usage 
python sample_labels_duckdb.py \
  --labels-csv data/china_labeling_sample_all_Jul30.csv \
  --out-train  data/labels_bal_train.csv \
  --out-val    data/labels_bal_val.csv \
  --train-cap-per-cell 200 \
  --val-cap-per-cell 100 \
  --seed 7

Notes
  - Set --labels-video-col if your filepath column isn't named 'video'.
  - For a simpler mode (balanced VAL + remainder TRAIN or downsampling majority classes),
    use the earlier simplified sampler; this script focuses on *balanced splits for both sets*.
"""

import argparse
import os
import duckdb

OUT_COLS = [
    "meta_id", "china_stance_score", "sensitive", "collective_action",
    "english", "mandarin", "spanish", "other_lang", "no_language"
]

# ---------- helpers ----------

def sql_q(s: str) -> str:
    """Escape single quotes for SQL string literal."""
    return s.replace("'", "''")

def connect_db():
    return duckdb.connect()

def build_L0(con, labels_csv: str, labels_video_col: str):
    """Raw parse + lowercase strings; keep only rows with meta_id extracted."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE L0 AS
    SELECT
      regexp_extract({labels_video_col}, '([0-9]{{10,}})\\.mp4', 1) AS meta_id,
      lower(trim(china_good))        AS g,
      lower(trim(china_bad))         AS b,
      lower(trim(china_neutral))     AS n,
      lower(trim(china_not_related)) AS u,
      lower(trim(china_cnd))         AS cnd,
      lower(trim(sensitive))         AS sensitive_raw,
      lower(trim(collective_action)) AS collective_raw,
      lower(trim(english))     AS english,
      lower(trim(mandarin))    AS mandarin,
      lower(trim(spanish))     AS spanish,
      lower(trim(other_lang))  AS other_lang,
      lower(trim(no_language)) AS no_language
    FROM read_csv_auto('{sql_q(labels_csv)}',
                       HEADER=TRUE, ALL_VARCHAR=TRUE, IGNORE_ERRORS=TRUE, SAMPLE_SIZE=-1)
    WHERE {labels_video_col} IS NOT NULL
      AND regexp_extract({labels_video_col}, '([0-9]{{10,}})\\.mp4', 1) IS NOT NULL;
    """)

def build_LN(con):
    """Normalize: stance via argmax; booleanize sensitive/collective; pass language flags."""
    con.execute("""
    CREATE OR REPLACE TEMP TABLE LN AS
    WITH V AS (
      SELECT
        meta_id,
        COALESCE(TRY_CAST(g  AS DOUBLE), CASE WHEN g  IN ('true','t','yes','y') THEN 1 ELSE 0 END, 0) AS v_good,
        COALESCE(TRY_CAST(b  AS DOUBLE), CASE WHEN b  IN ('true','t','yes','y') THEN 1 ELSE 0 END, 0) AS v_bad,
        COALESCE(TRY_CAST(n  AS DOUBLE), CASE WHEN n  IN ('true','t','yes','y') THEN 1 ELSE 0 END, 0) AS v_neutral,
        COALESCE(TRY_CAST(u  AS DOUBLE), CASE WHEN u  IN ('true','t','yes','y') THEN 1 ELSE 0 END, 0) AS v_notrel,
        COALESCE(TRY_CAST(cnd AS DOUBLE), CASE WHEN cnd IN ('true','t','yes','y') THEN 1 ELSE 0 END, 0) AS v_cnd,
        sensitive_raw, collective_raw, english, mandarin, spanish, other_lang, no_language
      FROM L0
    ), S AS (
      SELECT
        meta_id, v_good, v_bad, v_neutral, v_notrel, v_cnd,
        GREATEST(v_good, v_bad, v_neutral, v_notrel, v_cnd) AS vmax,
        (CASE WHEN v_good    = GREATEST(v_good, v_bad, v_neutral, v_notrel, v_cnd) THEN 1 ELSE 0 END +
         CASE WHEN v_bad     = GREATEST(v_good, v_bad, v_neutral, v_notrel, v_cnd) THEN 1 ELSE 0 END +
         CASE WHEN v_neutral = GREATEST(v_good, v_bad, v_neutral, v_notrel, v_cnd) THEN 1 ELSE 0 END +
         CASE WHEN v_notrel  = GREATEST(v_good, v_bad, v_neutral, v_notrel, v_cnd) THEN 1 ELSE 0 END +
         CASE WHEN v_cnd     = GREATEST(v_good, v_bad, v_neutral, v_notrel, v_cnd) THEN 1 ELSE 0 END) AS nmax,
        sensitive_raw, collective_raw, english, mandarin, spanish, other_lang, no_language
      FROM V
    )
    SELECT
      meta_id,
      CASE
        WHEN vmax = 0 OR nmax > 1 THEN 'cannot_determine'
        WHEN v_good    = vmax THEN 'good'
        WHEN v_bad     = vmax THEN 'bad'
        WHEN v_neutral = vmax THEN 'neutral'
        WHEN v_notrel  = vmax THEN 'not_related'
        WHEN v_cnd     = vmax THEN 'cannot_determine'
        ELSE 'cannot_determine'
      END AS china_stance_score,
      CASE
        WHEN sensitive_raw IN ('yes','y','1','true','t') THEN 'yes'
        WHEN sensitive_raw IN ('no','n','0','false','f') THEN 'no'
        WHEN sensitive_raw IN ('could not determine','cannot determine','cnd','unknown','n/a','') THEN 'cannot_determine'
        ELSE 'cannot_determine'
      END AS sensitive,
      CASE
        WHEN collective_raw IN ('yes','y','1','true','t') THEN 'yes'
        WHEN collective_raw IN ('no','n','0','false','f') THEN 'no'
        WHEN collective_raw IN ('could not determine','cannot determine','cnd','unknown','n/a','') THEN 'cannot_determine'
        ELSE 'cannot_determine'
      END AS collective_action,
      english, mandarin, spanish, other_lang, no_language
    FROM S
    WHERE meta_id IS NOT NULL;
    """)

def rank_cells(con, seed: int):
    """Rank once per joint cell deterministically and store counts."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE RANKED AS
    SELECT
      *,
      row_number() OVER (
        PARTITION BY china_stance_score, sensitive, collective_action
        ORDER BY hash(meta_id || '-SPLIT-{int(seed)}')
      ) AS rn,
      count(*) OVER (
        PARTITION BY china_stance_score, sensitive, collective_action
      ) AS c
    FROM LN;
    """)

def allocate_counts(con, train_cap_per_cell: int, val_cap_per_cell: int, train_frac: float):
    """Compute per-row train_n and val_n per cell with ratio + caps and room for both splits."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ALLOC AS
    SELECT
      *,
      LEAST(
        {int(train_cap_per_cell)},
        CASE
          WHEN c = 1 THEN 1
          WHEN c = 2 THEN 1
          ELSE GREATEST(1, CAST(ROUND(c * {float(train_frac)}) AS INTEGER))
        END,
        CASE WHEN c >= 2 THEN c - 1 ELSE 1 END
      ) AS train_n
    FROM RANKED;
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ALLOC2 AS
    SELECT
      *,
      LEAST({int(val_cap_per_cell)}, GREATEST(0, c - train_n)) AS val_n
    FROM ALLOC;
    """)

def slice_splits(con):
    """Materialize TRAIN and VAL tables (non-overlapping) with builder-ready columns."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE TRAIN AS
    SELECT {", ".join(OUT_COLS)}
    FROM ALLOC2
    WHERE rn <= train_n;
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE VAL AS
    SELECT {", ".join(OUT_COLS)}
    FROM ALLOC2
    WHERE rn > train_n AND rn <= train_n + val_n;
    """)

def write_csv(con, table: str, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    con.execute(f"COPY (SELECT {', '.join(OUT_COLS)} FROM {table}) TO '{sql_q(out_path)}' (FORMAT CSV, HEADER);")

def summarize(con, table: str, title: str):
    print(f"\n== {title} ==")
    rows = con.execute(f"""
      SELECT china_stance_score, sensitive, collective_action, COUNT(*) AS n
      FROM {table}
      GROUP BY 1,2,3
      ORDER BY n DESC
    """).fetchall()
    for r in rows:
        print(f"{r[0]:>16} | {r[1]:>17} | {r[2]:>16} | {r[3]:>6}")
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print("Total rows:", total)

def audit_starved(con):
    """Report cells with c>=2 but VAL got 0 rows."""
    res = con.execute("""
      WITH C AS (
        SELECT china_stance_score, sensitive, collective_action, COUNT(*) AS c
        FROM LN GROUP BY 1,2,3
      ),
      V AS (
        SELECT china_stance_score, sensitive, collective_action, COUNT(*) AS v
        FROM VAL GROUP BY 1,2,3
      )
      SELECT C.china_stance_score, C.sensitive, C.collective_action, C.c, COALESCE(V.v,0) AS val_got
      FROM C LEFT JOIN V USING (china_stance_score, sensitive, collective_action)
      WHERE C.c >= 2 AND COALESCE(V.v,0) = 0
      ORDER BY C.c DESC;
    """).fetchall()
    if res:
        print("\nCells with c>=2 but VAL received 0 (lower --train-frac or raise --val-cap-per-cell):")
        for r in res:
            print(f"{r[0]:>16} | {r[1]:>17} | {r[2]:>16}  c={r[3]}  val=0")

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--labels-video-col", default="video")
    ap.add_argument("--train-cap-per-cell", type=int, default=200)
    ap.add_argument("--val-cap-per-cell", type=int, default=100)
    ap.add_argument("--train-frac", type=float, default=0.7, help="fraction per cell to TRAIN (0..1)")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()

def main():
    args = parse_args()
    con = connect_db()
    build_L0(con, args.labels_csv, args.labels_video_col)
    build_LN(con)
    rank_cells(con, args.seed)
    allocate_counts(con, args.train_cap_per_cell, args.val_cap_per_cell, args.train_frac)
    slice_splits(con)
    write_csv(con, "TRAIN", args.out_train)
    write_csv(con, "VAL",   args.out_val)
    summarize(con, "TRAIN", "TRAIN (balanced per cell)")
    summarize(con, "VAL",   "VAL   (balanced per cell)")
    audit_starved(con)

if __name__ == "__main__":
    main()
