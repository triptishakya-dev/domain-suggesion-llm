"""Validate raw domain suggestion records and write filtered outputs."""
import argparse, hashlib, json, re, tempfile
from pathlib import Path
FIELDS={"input","domain","profession","reason","skills"}; TEXT_FIELDS=FIELDS-{"skills"}; CONTROL_RE=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as f: json.dump(value,f,ensure_ascii=False,indent=2); f.write("\n"); temp=Path(f.name)
 temp.replace(path)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output-dir",required=True); a=ap.parse_args(); p=Path(a.input); data=json.loads(p.read_text(encoding="utf-8-sig"))
 if not isinstance(data,list) or not data: raise SystemExit("Input must be a non-empty JSON array")
 accepted=[]; rejected=[]
 for pos,r in enumerate(data,1):
  why=[]
  if not isinstance(r,dict): why=["invalid_record_type"]
  else:
   for k in sorted(FIELDS-r.keys()): why.append("missing_field:"+k)
   for k in sorted(r.keys()-FIELDS): why.append("unexpected_field:"+k)
   for k in sorted(TEXT_FIELDS & r.keys()):
    if not isinstance(r[k],str) or not r[k].strip(): why.append("invalid_text_field:"+k)
    elif CONTROL_RE.search(r[k]): why.append("invalid_control_character:"+k)
   if not isinstance(r.get("skills"),list) or not r["skills"] or any(not isinstance(s,str) or not s.strip() for s in r["skills"]): why.append("invalid_skills")
  if why: rejected.append({"source_position":pos,"record":r,"reasons":why})
  else: accepted.append(r)
 digest=hashlib.sha256(p.read_bytes()).hexdigest(); out=Path(a.output_dir); write_json(out/"row.json",accepted); write_json(out/"rejected.json",rejected); write_json(out/"filter_report.json",{"source_sha256":digest,"rule_version":1,"source_count":len(data),"accepted_count":len(accepted),"rejected_count":len(rejected)})
 print(f"filtered {len(data)} records: {len(accepted)} accepted, {len(rejected)} rejected")
if __name__=="__main__": main()
