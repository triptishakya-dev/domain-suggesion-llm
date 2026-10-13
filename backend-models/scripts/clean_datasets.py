"""Normalize and deduplicate filtered records."""
import argparse,hashlib,json,re,tempfile,unicodedata
from pathlib import Path
SPACE=re.compile(r"\s+")
def norm(v): return SPACE.sub(" ",unicodedata.normalize("NFC",v).replace("\u00a0"," ")).strip()
def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as f: json.dump(value,f,ensure_ascii=False,indent=2); f.write("\n"); temp=Path(f.name)
 temp.replace(path)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output-dir",required=True); a=ap.parse_args(); p=Path(a.input); data=json.loads(p.read_text(encoding="utf-8-sig")); out=Path(a.output_dir); cleaned=[]; changes=[]; seen={}
 for pos,r in enumerate(data,1):
  item={k:norm(r[k]) for k in ("input","domain","profession","reason")}; skills=[]; keys=set()
  for s in r["skills"]:
   s=norm(s)
   if s.casefold() not in keys: keys.add(s.casefold()); skills.append(s)
  item["skills"]=skills; sig=json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(",",":"))
  if sig in seen: changes.append({"source_position":pos,"changes":["removed_exact_duplicate"],"duplicate_of":seen[sig]}); continue
  seen[sig]=pos; cleaned.append(item)
  changed=[k for k in item if item[k]!=r[k]]
  if changed: changes.append({"source_position":pos,"changes":changed})
 write_json(out/"row.json",cleaned); write_json(out/"change_log.json",changes); write_json(out/"review_queue.json",[]); write_json(out/"clean_report.json",{"input_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"rule_version":1,"input_count":len(data),"cleaned_count":len(cleaned),"removed_duplicate_count":len(data)-len(cleaned),"change_count":len(changes)})
 print(f"cleaned {len(data)} records: {len(cleaned)} retained")
if __name__=="__main__": main()
