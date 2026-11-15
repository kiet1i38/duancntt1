"""Verify immutable results and regenerate the analysis without a GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def verify():
    expected=json.loads((ROOT/"sha256.json").read_text())
    for name,digest in expected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    plan=json.loads((ROOT/"experiment_plan.json").read_text())
    total=0
    for model in plan["models"]:
        folder=ROOT/"results"/model
        mf=json.loads((folder/"manifest.json").read_text())
        assert mf["status"]=="complete"
        for name,key in (("runner.py","runner_sha256"),("experiment_plan.json","experiment_plan_sha256")):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==mf["contract"][key]
        routes={r["source_row_index"]:r for r in json.loads((folder/"routing.json").read_text())}
        archive={r["source_row_index"]:r for r in map(json.loads,(ROOT/"archived_routes"/f"{model}.jsonl").read_text().splitlines())}
        metrics=json.loads((folder/"metrics.json").read_text())
        for method in plan["method_order"][model]:
            rows=[json.loads(x) for x in (folder/(method+".jsonl")).read_text(encoding="utf-8").splitlines()]
            assert sorted(r["source_row_index"] for r in rows)==list(range(1319))
            for r in rows:
                i=r["source_row_index"]
                assert r["question_sha256"]==routes[i]["question_sha256"]==archive[i]["question_sha256"]
                assert r["gamma"]==(routes[i]["gamma"] if method=="sd_frozen" else float(method[6:]))
            assert sum(bool(r["accuracy"]) for r in rows)==metrics[method]["correct"]
            assert abs(sum(r["cot_length"] for r in rows)/1319-metrics[method]["mean_cot_tokens"])<1e-9
            total+=len(rows)
    assert total==18466
    print(f"Verified {len(expected)} file hashes and {total} aligned generations",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"analysis")
    args=parser.parse_args()
    verify()
    if not args.verify_only:
        subprocess.run([sys.executable,"-B",str(ROOT/"analyze_results.py"),"--results",str(ROOT/"results"),"--output",str(args.output)],check=True)
        subprocess.run([sys.executable,"-B",str(ROOT/"plot_results.py"),"--analysis",str(args.output/"analysis.json"),"--output",str(args.output)],check=True)
