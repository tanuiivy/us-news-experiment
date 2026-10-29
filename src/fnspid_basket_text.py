import csv, io, sys

csv.field_size_limit(2**31 - 1)
out_path = sys.argv[1]
START, END = "2019-01-01", "2023-12-31"
BASKET = set("GOOG INTC AMD SBUX AAL QCOM MU CMCSA PYPL BIIB PEP COST ADBE GILD BIDU".split())
CUT = 4000   # characters kept per article (enough for 512 tokens)

stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="")
reader = csv.DictReader(stdin)
seen = kept = usable = 0
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Stock_symbol", "Article_title", "Article"])
    for row in reader:
        seen += 1
        day = (row.get("Date") or "")[:10]
        if START <= day <= END and row.get("Stock_symbol") in BASKET:
            art = (row.get("Article") or "")[:CUT]
            w.writerow([row.get("Date"), row.get("Stock_symbol"), row.get("Article_title"), art])
            kept += 1
            if len(art.strip()) >= 200:
                usable += 1
            if kept == 71778:   # all matching rows found, no need to read the rest
                break
        if seen % 1_000_000 == 0:
            print(f"{seen:,} rows read, {kept:,} kept, {usable:,} with usable text", file=sys.stderr, flush=True)
print(f"done: {seen:,} read, {kept:,} kept (expect 71,778), usable text: {usable:,} ({usable / max(kept, 1):.1%})", file=sys.stderr)
