"""Stream FNSPID, resuming from the last byte received whenever the connection drops or stalls.

Usage: python src/fnspid_basket_text_resumable.py OUT.csv [URL] [EXPECT]
Keeps the same rows, in the same order, as the headlines extraction, with title and article text.
"""
import csv, http.client, io, os, sys, time, urllib.request

DEFAULT_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
out_path = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URL
EXPECT = int(sys.argv[3]) if len(sys.argv) > 3 else 71778
START, END = "2019-01-01", "2023-12-31"
BASKET = set("GOOG INTC AMD SBUX AAL QCOM MU CMCSA PYPL BIIB PEP COST ADBE GILD BIDU".split())
CUT = 4000   # characters kept per article (enough for 512 tokens)
TIMEOUT, MAX_TRIES = float(os.environ.get("FNSPID_TIMEOUT", 30)), 60

state = {"pos": 0}

def chunks():
    tries = 0
    while True:
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={state['pos']}-"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                cr = r.headers.get("Content-Range")           # e.g. "bytes 0-999/5000"
                if cr and "/" in cr and cr.split("/")[1].isdigit():
                    total = int(cr.split("/")[1])
                else:
                    total = state["pos"] + int(r.headers.get("Content-Length", 0))
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        if state["pos"] >= total:
                            return                  # every byte has arrived: a real end of file
                        raise ConnectionError(f"short read: got {state['pos']:,} of {total:,} bytes")
                    state["pos"] += len(b)
                    tries = 0
                    yield b
        except (OSError, http.client.HTTPException) as e:
            tries += 1
            print(f"connection problem at byte {state['pos']:,}: {e!r}; retry {tries}/{MAX_TRIES}",
                  file=sys.stderr, flush=True)
            if tries >= MAX_TRIES:
                raise
            time.sleep(5)

class Raw(io.RawIOBase):
    def __init__(self, gen):
        self.gen, self.buf = gen, b""
    def readable(self):
        return True
    def readinto(self, b):
        try:
            while not self.buf:
                self.buf = next(self.gen)
        except StopIteration:
            return 0
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n

csv.field_size_limit(2**31 - 1)
text = io.TextIOWrapper(io.BufferedReader(Raw(chunks()), 1 << 20), encoding="utf-8", errors="replace", newline="")
reader = csv.DictReader(text)
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
            if kept == EXPECT:       # all matching rows found, no need to read the rest
                break
        if seen % 1_000_000 == 0:
            print(f"{seen:,} rows read, {kept:,} kept, {usable:,} with usable text, {state['pos'] / 1e9:.2f} GB",
                  file=sys.stderr, flush=True)
print(f"done: {seen:,} read, {kept:,} kept (expect {EXPECT:,}), usable text: {usable:,} ({usable / max(kept, 1):.1%})",
      file=sys.stderr)
if kept != EXPECT:
    print("WARNING: kept count differs from expected, do not use this file", file=sys.stderr)
