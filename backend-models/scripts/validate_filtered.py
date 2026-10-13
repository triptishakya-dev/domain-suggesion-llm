import argparse,json
from pathlib import Path
FIELDS={"input","domain","profession","reason","skills"}
ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); a=ap.parse_args(); data=json.loads(Path(a.input).read_text(encoding="utf-8-sig"))
if not isinstance(data,list) or not data: raise SystemExit("Dataset must be a non-empty array")
for i,r in enumerate(data,1):
 if set(r)!=FIELDS or any(not isinstance(r[k],str) or not r[k].strip() for k in FIELDS-{"skills"}) or not isinstance(r["skills"],list) or not r["skills"]: raise SystemExit(f"Invalid record at {i}")
print(f"valid: {len(data)} records")
