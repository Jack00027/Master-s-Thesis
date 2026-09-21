import json, glob, os, csv

def rows_for(path, arm):
    h = json.load(open(path))
    run = os.path.basename(os.path.dirname(path))
    q = run.split("__")[0].replace("q_", "")
    out = []
    for rec in h.get("pretrain_log", []):
        if "eval_loss" in rec:
            out.append([arm, q, "mlm_eval", rec["epoch"], rec["eval_loss"]])
        elif "loss" in rec:
            out.append([arm, q, "mlm_train", rec["epoch"], rec["loss"]])
    ft = h.get("finetune_log") or h.get("finetune") or {}
    losses = ft.get("loss", ft) if isinstance(ft, dict) else ft
    if isinstance(losses, list):
        for i, x in enumerate(losses, 1):
            out.append([arm, q, "nce", i, x if not isinstance(x, dict) else x.get("loss")])
    return out

rows = []
for arm, pat in [("weighted", "models_weighted/*__weighted_paper/history.json"),
                 ("baseline", "models_v3/*/history.json")]:
    for f in sorted(glob.glob(pat)):
        rows += rows_for(f, arm)

with open("history_long.csv", "w", newline="") as o:
    w = csv.writer(o); w.writerow(["arm","quarter","metric","epoch","value"]); w.writerows(rows)
print(len(rows), "rows,", len({r[1] for r in rows}), "quarters")