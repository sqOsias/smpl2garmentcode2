#!/usr/bin/env python3
"""汇总 output/CloSe 下所有 eval_summary.json，计算整体统计。"""
import os, json, csv
import numpy as np

OUTPUT_ROOT = "/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/output/CloSe"
OUT_CSV = os.path.join(OUTPUT_ROOT, "summary_all.csv")
OUT_JSON = os.path.join(OUTPUT_ROOT, "summary_all.json")

rows = []
fs_keys = ["5", "10", "20", "30", "50"]

for sample in sorted(os.listdir(OUTPUT_ROOT)):
    path = os.path.join(OUTPUT_ROOT, sample, "eval_summary.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        m = json.load(f)
    row = {
        "sample": sample,
        "valid_structure": m.get("valid_structure", 0),
        "sim_success": m.get("sim_success", 0),
        "class_acc": m.get("class_acc", 0),
        "upper_correct": m.get("upper_correct", 0),
        "bottom_correct": m.get("bottom_correct", 0),
        "connected_correct": m.get("connected_correct", 0),
        "chamfer_distance": m.get("chamfer_distance"),
    }
    for k in fs_keys:
        fs = m.get("fscores", {}).get(k, {})
        row[f"F@{k}mm"] = fs.get("fscore")
        row[f"P@{k}mm"] = fs.get("precision")
        row[f"R@{k}mm"] = fs.get("recall")
    rows.append(row)

if not rows:
    print("No eval_summary.json found")
    exit(1)

# ---- 汇总统计 ----
def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None

summary = {
    "total_samples": len(rows),
    "val_rate": safe_mean([r["valid_structure"] for r in rows]),
    "sim_success_rate": safe_mean([r["sim_success"] for r in rows]),
    "class_acc": safe_mean([r["class_acc"] for r in rows]),
    "upper_acc": safe_mean([r["upper_correct"] for r in rows]),
    "bottom_acc": safe_mean([r["bottom_correct"] for r in rows]),
    "connected_acc": safe_mean([r["connected_correct"] for r in rows]),
    "mean_cd_cm": safe_mean([r["chamfer_distance"] for r in rows]),
    "median_cd_cm": float(np.median([r["chamfer_distance"] for r in rows if r["chamfer_distance"] is not None])),
}
for k in fs_keys:
    summary[f"mean_F@{k}mm"] = safe_mean([r[f"F@{k}mm"] for r in rows])
    summary[f"median_F@{k}mm"] = float(np.median([r[f"F@{k}mm"] for r in rows if r[f"F@{k}mm"] is not None]))

# ---- 打印 ----
print("=" * 60)
print(f"Summary of {len(rows)} samples")
print("=" * 60)
print(f"  Val.Rate:        {summary['val_rate']*100:.1f}%")
print(f"  SSR:             {summary['sim_success_rate']*100:.1f}%")
print(f"  Meta Acc:        {summary['class_acc']*100:.1f}%")
print(f"    upper:         {summary['upper_acc']*100:.1f}%")
print(f"    bottom:        {summary['bottom_acc']*100:.1f}%")
print(f"    connected:     {summary['connected_acc']*100:.1f}%")
print(f"  CD (mean):       {summary['mean_cd_cm']:.3f} cm")
print(f"  CD (median):     {summary['median_cd_cm']:.3f} cm")
for k in fs_keys:
    print(f"  F@{k}mm (mean):    {summary[f'mean_F@{k}mm']:.4f}")
print()

# ---- 保存 CSV ----
cols = ["sample", "valid_structure", "sim_success", "class_acc",
        "upper_correct", "bottom_correct", "connected_correct", "chamfer_distance"]
for k in fs_keys:
    cols += [f"F@{k}mm", f"P@{k}mm", f"R@{k}mm"]

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"Per-sample CSV: {OUT_CSV}")

# ---- 保存 JSON ----
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON:   {OUT_JSON}")
