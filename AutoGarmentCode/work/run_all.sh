#!/usr/bin/env bash
set -uo pipefail  # 单个样本失败时由批处理调用方决定是否继续

# =============================================================================
# 单入口完整管线:
# HybrIK → 量体 → LLM 设计 → GarmentCode 制版/仿真 → Warp 姿态驱动
#        → SMPL 人体刚性对齐 → CD / F-Score@10mm
#
# 用法:
#   bash run_all.sh [SAMPLE] [GENDER]
# =============================================================================

SAMPLE="${1:-10089_8138}"
GENDER="${2:-male}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMING_CSV="$SCRIPT_DIR/batch_logs/full_pipeline_timing.csv"

mkdir -p "$(dirname "$TIMING_CSV")"
if [ ! -f "$TIMING_CSV" ]; then
  echo "sample,gender,pipeline_sec,status" > "$TIMING_CSV"
fi

START_SEC=$(date +%s)

echo "######################################################################"
echo "# Full Pipeline: $SAMPLE ($GENDER)"
echo "######################################################################"

if bash "$SCRIPT_DIR/run.sh" "$SAMPLE" "sim" "$GENDER"; then
  ELAPSED=$(( $(date +%s) - START_SEC ))
  echo "$SAMPLE,$GENDER,$ELAPSED,success" >> "$TIMING_CSV"
  echo "######################################################################"
  echo "# DONE: $SAMPLE  Total=${ELAPSED}s ($(( ELAPSED / 60 ))m)"
  echo "######################################################################"
else
  ELAPSED=$(( $(date +%s) - START_SEC ))
  echo "$SAMPLE,$GENDER,$ELAPSED,failed" >> "$TIMING_CSV"
  echo "$SAMPLE" >> "$SCRIPT_DIR/failed_samples.txt"
  echo "[FAIL] Full pipeline failed for $SAMPLE" >&2
  exit 1
fi
