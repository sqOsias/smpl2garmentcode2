#!/usr/bin/env bash
set -uo pipefail  # 不用 -e: 单个样本失败不退出，继续下一个

# =============================================================================
# 完整管线: 服装生成 → Warp 驱动 → CD/F-Score 评估
#
# 阶段1 (work/run.sh): HybrIK → 量体 → LLM设计 → GarmentCode制版仿真
# 阶段2 (work/test.sh): Warp 驱动服装到 GT pose → 计算 CD + F-Score
#
# 用法:
#   bash run_all.sh [SAMPLE] [GENDER]
#       SAMPLE  CloSe 样本名（默认 10014_2464）
#       GENDER  性别 male|female（默认 male）
#
# 阶段开关（环境变量）:
#   DO_STAGE1=1  运行 run.sh（服装生成）
#   DO_STAGE2=1  运行 test.sh（驱动+评估）
# =============================================================================

SAMPLE="${1:-10089_8138}"
GENDER="${2:-male}"

DO_STAGE1="${DO_STAGE1:-1}"
DO_STAGE2="${DO_STAGE2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMING_CSV="$SCRIPT_DIR/batch_logs/timing.csv"

# 初始化计时文件（表头）
if [ ! -f "$TIMING_CSV" ]; then
  echo "sample,stage1_sec,stage2_sec,total_sec" > "$TIMING_CSV"
fi

TOTAL_START=$(date +%s)

echo "######################################################################"
echo "#  Full Pipeline: $SAMPLE ($GENDER)"
echo "######################################################################"

STAGE1_SEC=0
STAGE2_SEC=0

# ---- 阶段1: 服装生成 (work/run.sh) ----
if [ "$DO_STAGE1" = "1" ]; then
  echo ""
  echo ">>> Stage 1: Garment Generation (work/run.sh)  [$(date '+%H:%M:%S')]"
  echo ""
  S1_START=$(date +%s)
  if bash "$SCRIPT_DIR/run.sh" "$SAMPLE" "sim" "$GENDER"; then
    STAGE1_SEC=$(( $(date +%s) - S1_START ))
    echo ">>> Stage 1 done: ${STAGE1_SEC}s ($(( STAGE1_SEC / 60 ))m $(( STAGE1_SEC % 60 ))s)"
  else
    STAGE1_SEC=-1
    DO_STAGE2=0  # 跳过阶段2
    echo "[FAIL] Stage 1 failed for $SAMPLE (LLM设计/制版错误), skipping stage 2" >&2
    echo "$SAMPLE" >> "$SCRIPT_DIR/failed_samples.txt"
  fi
else
  echo ""
  echo ">>> Stage 1: SKIPPED"
fi

# ---- 阶段2: Warp 驱动 + 评估 (work/test.sh) ----
if [ "$DO_STAGE2" = "1" ]; then
  echo ""
  echo ">>> Stage 2: Warp Drive + Evaluation (work/test.sh)  [$(date '+%H:%M:%S')]"
  echo ""
  S2_START=$(date +%s)
  if bash "$SCRIPT_DIR/test.sh" "$SAMPLE" "$GENDER"; then
    STAGE2_SEC=$(( $(date +%s) - S2_START ))
    echo ">>> Stage 2 done: ${STAGE2_SEC}s ($(( STAGE2_SEC / 60 ))m $(( STAGE2_SEC % 60 ))s)"
  else
    STAGE2_SEC=-1
    echo "[FAIL] Stage 2 failed for $SAMPLE" >&2
    echo "$SAMPLE" >> "$SCRIPT_DIR/failed_samples.txt"
  fi
else
  echo ""
  echo ">>> Stage 2: SKIPPED"
fi

TOTAL_SEC=$(( $(date +%s) - TOTAL_START ))

# 记录到 CSV
echo "$SAMPLE,$STAGE1_SEC,$STAGE2_SEC,$TOTAL_SEC" >> "$TIMING_CSV"

echo ""
echo "######################################################################"
echo "#  DONE: $SAMPLE"
echo "#  Stage1=${STAGE1_SEC}s  Stage2=${STAGE2_SEC}s  Total=${TOTAL_SEC}s ($(( TOTAL_SEC / 60 ))m)"
echo "######################################################################"
