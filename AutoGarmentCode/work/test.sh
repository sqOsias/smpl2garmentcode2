#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Warp 驱动服装 → GT pose → 评估 CD / F-Score
#
# 用法:
#   bash run.sh [SAMPLE] [GENDER]
#       SAMPLE  CloSe 样本名（默认 10014_2464）
#       GENDER  性别 male|female（默认 male）
#
# 阶段开关（环境变量）:
#   DO_DRIVE=1    Warp 驱动服装到 GT pose
#   DO_METRIC=1   计算 CD 和 F-Score
#
# 依赖 conda 环境: nvidiawarp
# =============================================================================

SAMPLE="${1:-10030_3499}"
GENDER="${2:-female}"

DO_DRIVE="${DO_DRIVE:-1}"
DO_METRIC="${DO_METRIC:-1}"

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../AutoGarmentCode/work
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"               # .../AutoGarmentCode

NPZ_PATH="/root/wyc/data/CloSe/data/CloSe-Di/${SAMPLE}.npz"
OUTPUT_SAMPLE_DIR="$PROJECT_ROOT/output/CloSe/${SAMPLE}"
SMPL_JSON="$OUTPUT_SAMPLE_DIR/hybrik/smpl.json"
DRIVEN_OUTPUT_DIR="$OUTPUT_SAMPLE_DIR/driven"

# conda 环境
CONDA_NW="/root/miniconda3/bin/conda run --no-capture-output -n nvidiawarp"

echo "=========================================="
echo " Sample: $SAMPLE    Gender: $GENDER"
echo " npz   : $NPZ_PATH"
echo "=========================================="

if [ ! -f "$NPZ_PATH" ]; then
  echo "[FATAL] cannot find npz: $NPZ_PATH" >&2
  exit 1
fi

if [ ! -f "$SMPL_JSON" ]; then
  echo "[FATAL] cannot find smpl.json: $SMPL_JSON" >&2
  echo "  -> run.sh (HybrIK step) must be executed first" >&2
  exit 1
fi

# ---- 1. Warp 驱动: A-pose garment -> GT pose ----
if [ "$DO_DRIVE" = "1" ]; then
  echo "[1/2] Warp driving garment to GT pose ..."
  $CONDA_NW python "$SCRIPT_DIR/driven_garment_mesh.py" \
      --sample "$SAMPLE" \
      --gender "$GENDER"
  echo "      final_result.obj -> $DRIVEN_OUTPUT_DIR"
else
  echo "[1/2] skipping Warp drive"
fi

# ---- 2. 指标评估: CD + F-Score ----
if [ "$DO_METRIC" = "1" ]; then
  echo "[2/2] computing CD and F-Score ..."
  $CONDA_NW python "$SCRIPT_DIR/metric.py" \
      --sample "$SAMPLE" \
      --data_root "$(dirname "$NPZ_PATH")" \
      --output_root "$PROJECT_ROOT/output/CloSe" \
      --gender "$GENDER"
else
  echo "[2/2] skipping metrics"
fi

echo "=========================================="
echo " finish: $DRIVEN_OUTPUT_DIR"
echo "=========================================="
