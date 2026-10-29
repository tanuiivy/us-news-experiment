import csv, io, sys
from collections import Counter

# Article text fields are huge, so raise csv's per-field limit or rows will crash
csv.field_size_limit(2**31 - 1)

out_path, counts_path = sys.argv[1], sys.argv[2]
START, END = "2019-01-01", "2023-12-31"

# Wrap stdin so quoted multi-line article text is read correctly on Windows
stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="")
reader = csv.DictReader(stdin)
counts = Counter()
seen = kept = 0
lo, hi = "9999", ""   # earliest and latest date seen, to verify the date format

with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Stock_symbol", "Article_title", "Publisher"])
    for row in reader:
        seen += 1
        day = (row.get("Date") or "")[:10]
        if day:
            lo, hi = min(lo, day), max(hi, day)
        if START <= day <= END:
            sym = row.get("Stock_symbol") or ""
            w.writerow([row.get("Date"), sym, row.get("Article_title"), row.get("Publisher")])
            counts[(sym, day[:4])] += 1
            kept += 1
        if seen % 500_000 == 0:
            print(f"{seen:,} rows read, {kept:,} kept", file=sys.stderr, flush=True)

with open(counts_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Stock_symbol", "year", "n_articles"])
    for (sym, yr), n in sorted(counts.items()):
        w.writerow([sym, yr, n])

print(f"done: {seen:,} read, {kept:,} kept, dates {lo} to {hi}", file=sys.stderr)
