"""Ingest a Monday.com Corp Sales board export and produce normalized,
job-level lead-time data.

The export interleaves group-title rows, repeated header rows, and group
summary rows. Container date columns (ETA Departure, ETA Port, ETA Load,
Actual to LOC) are comma-separated strings mirrored from the linked Corp
Containers board — one date per container, so a job may have several.

Output: one row per job with order date, location, and lead times in weeks
  weeks_to_port  = last ETA Port date    - Order Date
  weeks_to_loc   = last Actual to LOC    - Order Date
(ETA Port is the only port date available; there is no actual-to-port date.)
Rows that fail validation are written to a separate rejects file with a
reason, so nothing disappears silently.
"""

import argparse
import sys

import pandas as pd

HEADER_NAME = "Name"

# Groups that represent real orders. Everything else (quotes, cold
# proposals, new requests) never shipped and has no lead time.
DEFAULT_GROUPS = ("Ordered", "Archived Shipped Complete")

MULTI_DATE_COLS = ["ETA Departure", "ETA Port", "ETA Load", "Actual to LOC"]

# Plausible bounds for order -> location lead time. Anything outside is bad
# data (typically containers from an earlier job linked onto a reorder item,
# or year typos).
DEFAULT_MIN_WEEKS = 8
DEFAULT_MAX_WEEKS = 24


def parse_export(path: str) -> pd.DataFrame:
    """Read a Monday export and return item rows tagged with their group."""
    raw = pd.read_excel(path, header=None)

    header_rows = raw.index[raw[0].eq(HEADER_NAME)].tolist()
    if not header_rows:
        raise ValueError(f"No header row (first cell {HEADER_NAME!r}) found in {path}")
    columns = raw.iloc[header_rows[0]].tolist()

    # A group title row has a name in col 0 and nothing else except the
    # boilerplate on the first two rows of the file.
    nonnull = raw.notna().sum(axis=1)
    group_rows = [
        i for i in raw.index
        if pd.notna(raw.iat[i, 0]) and nonnull[i] == 1 and i not in header_rows
    ]

    frames = []
    for start in group_rows:
        group = str(raw.iat[start, 0])
        # Data runs from the header row after the title to the next title.
        next_title = min((g for g in group_rows if g > start), default=len(raw))
        block = raw.iloc[start + 1 : next_title].copy()
        block.columns = columns
        block = block[block[HEADER_NAME].notna() & block[HEADER_NAME].ne(HEADER_NAME)]
        block["Group"] = group
        frames.append(block)

    return pd.concat(frames, ignore_index=True)


def parse_date_list(value) -> list[pd.Timestamp]:
    """Parse a comma-separated container date string into timestamps."""
    if pd.isna(value):
        return []
    dates = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            dates.append(pd.Timestamp(token))
    return dates


def normalize(
    items: pd.DataFrame,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    min_weeks: float = DEFAULT_MIN_WEEKS,
    max_weeks: float = DEFAULT_MAX_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean, rejects). Rejects carry a `reject_reason` column."""
    df = items[items["Group"].isin(groups)].copy()

    out = pd.DataFrame({
        "name": df[HEADER_NAME],
        "group": df["Group"],
        "location": df["Order Location"],
        "order_date": pd.to_datetime(df["Order Date"], errors="coerce"),
    })

    for col in MULTI_DATE_COLS:
        out[col.lower().replace(" ", "_")] = df[col].map(parse_date_list)

    out["n_containers"] = out["eta_port"].str.len().clip(lower=1)
    out["last_port_date"] = out["eta_port"].map(lambda ds: max(ds) if ds else pd.NaT)
    out["last_loc_date"] = out["actual_to_loc"].map(lambda ds: max(ds) if ds else pd.NaT)

    out["weeks_to_port"] = (out["last_port_date"] - out["order_date"]).dt.days / 7
    out["weeks_to_loc"] = (out["last_loc_date"] - out["order_date"]).dt.days / 7

    reasons = pd.Series("", index=out.index)
    reasons[out["order_date"].isna()] = "missing order date"
    no_dates = out["order_date"].notna() & out["last_port_date"].isna() & out["last_loc_date"].isna()
    reasons[no_dates] = "no container dates"
    out_of_bounds = out["weeks_to_loc"].notna() & (
        (out["weeks_to_loc"] < min_weeks) | (out["weeks_to_loc"] > max_weeks)
    )
    reasons[out_of_bounds & reasons.eq("")] = (
        f"location lead time outside {min_weeks}-{max_weeks} weeks"
    )
    neg_port = out["weeks_to_port"].notna() & (out["weeks_to_port"] < 0)
    reasons[neg_port & reasons.eq("")] = "port date before order date"

    rejects = out[reasons.ne("")].copy()
    rejects["reject_reason"] = reasons[reasons.ne("")]
    clean = out[reasons.eq("")].copy()
    return clean, rejects


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("excel_file", help="Monday.com Corp Sales export (.xlsx)")
    ap.add_argument("-o", "--output", default="normalized.csv")
    ap.add_argument("--rejects", default="rejects.csv")
    ap.add_argument("--min-weeks", type=float, default=DEFAULT_MIN_WEEKS,
                    help=f"reject location lead times under this (default {DEFAULT_MIN_WEEKS})")
    ap.add_argument("--max-weeks", type=float, default=DEFAULT_MAX_WEEKS,
                    help=f"reject location lead times over this (default {DEFAULT_MAX_WEEKS})")
    args = ap.parse_args(argv)

    items = parse_export(args.excel_file)
    clean, rejects = normalize(items, min_weeks=args.min_weeks, max_weeks=args.max_weeks)

    list_cols = [c.lower().replace(" ", "_") for c in MULTI_DATE_COLS]
    for frame in (clean, rejects):
        for c in list_cols:
            frame[c] = frame[c].map(lambda ds: ", ".join(d.date().isoformat() for d in ds))
    clean.to_csv(args.output, index=False)
    rejects.to_csv(args.rejects, index=False)

    print(f"{len(items)} items read; {len(clean)} clean jobs -> {args.output}; "
          f"{len(rejects)} rejected -> {args.rejects}")
    print("\nReject reasons:")
    print(rejects["reject_reason"].value_counts().to_string())
    for metric in ("weeks_to_port", "weeks_to_loc"):
        v = clean[metric].dropna()
        print(f"\n{metric}: n={len(v)}  median={v.median():.1f}  "
              f"p5={v.quantile(.05):.1f}  p95={v.quantile(.95):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
