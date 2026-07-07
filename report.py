"""Generate an interactive lead-time chart from normalized lead-time data.

Reads the normalized.csv produced by normalize.py and writes a fully
self-contained HTML file (no CDN, works offline): a scatter of lead time in
weeks vs order date, colored by order location, with a metric switch
(to LA port / to location), location toggles, and a date-range filter.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Categorical palettes hold 8 slots; the highest-volume locations get named
# slots and the tail folds into "Other".
TOP_SLOTS = 7

PAYLOAD_MARKER = "/*__PAYLOAD__*/null"


def build_payload(df: pd.DataFrame, source: str) -> dict:
    short = df["location"].fillna("Unknown").str.replace(r",\s*\w\w$", "", regex=True)
    counts = short.value_counts()
    top = counts.head(TOP_SLOTS).index.tolist()
    bucket = short.where(short.isin(top), "Other")
    buckets = top + (["Other"] if bucket.eq("Other").any() else [])
    index = {b: i for i, b in enumerate(buckets)}

    points = []
    for row, b in zip(df.itertuples(index=False), bucket):
        if pd.isna(row.order_date):
            continue
        points.append({
            "n": row.name if isinstance(row.name, str) else str(row.name),
            "b": index[b],
            "d": row.order_date.date().isoformat(),
            "p": None if pd.isna(row.weeks_to_port) else round(row.weeks_to_port, 1),
            "c": None if pd.isna(row.weeks_to_loc) else round(row.weeks_to_loc, 1),
        })
    points.sort(key=lambda p: p["d"])
    return {
        "buckets": buckets,
        "points": points,
        "source": source,
        "generated": date.today().isoformat(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", default="normalized.csv",
                    help="normalized csv from normalize.py (default normalized.csv)")
    ap.add_argument("-o", "--output", default="lead_times.html")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input, parse_dates=["order_date"])
    payload = build_payload(df, Path(args.input).name)

    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    if PAYLOAD_MARKER not in template:
        raise ValueError("payload marker not found in report_template.html")
    html = template.replace(PAYLOAD_MARKER, json.dumps(payload, ensure_ascii=False))
    Path(args.output).write_text(html, encoding="utf-8")

    print(f"{len(payload['points'])} jobs across {len(payload['buckets'])} "
          f"location groups -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
