#!/usr/bin/env bash
set -euo pipefail

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

SAMPLE="${1:-10030_3499}"
GENDER="${2:-female}"

DO_STAGE1="${DO_STAGE1:-1}"
DO_STAGE2="${DO_STAGE2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "######################################################################"
echo "#  Full Pipeline: $SAMPLE ($GENDER)"
echo "######################################################################"

# ---- 阶段1: 服装生成 (work/run.sh) ----
if [ "$DO_STAGE1" = "1" ]; then
  echo ""
  echo ">>> Stage 1: Garment Generation (work/run.sh)"
  echo ""
  bash "$SCRIPT_DIR/run.sh" "$SAMPLE" "$GENDER" || {
    echo "[FATAL] Stage 1 failed" >&2
    exit 1
  }
else
  echo ""
  echo ">>> Stage 1: SKIPPED"
fi

# ---- 阶段2: Warp 驱动 + 评估 (work/test.sh) ----
if [ "$DO_STAGE2" = "1" ]; then
  echo ""
  echo ">>> Stage 2: Warp Drive + Evaluation (work/test.sh)"
  echo ""
  bash "$SCRIPT_DIR/test.sh" "$SAMPLE" "$GENDER" || {
    echo "[FATAL] Stage 2 failed" >&2
    exit 1
  }
else
  echo ""
  echo ">>> Stage 2: SKIPPED"
fi

echo ""
echo "######################################################################"
echo "#  DONE: $SAMPLE"
echo "######################################################################"
