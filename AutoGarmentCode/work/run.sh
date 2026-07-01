#!/usr/bin/env bash
set -euo pipefail

SAMPLE="${1:-10014_2464}"
SIM="${2:-sim}"
GENDER="${3:-${GENDER:-female}}"

# ---- 阶段开关 ----
DO_RENDER="${DO_RENDER:-0}"
DO_HYBRIK="${DO_HYBRIK:-1}"
DO_SMPLOBJ="${DO_SMPLOBJ:-1}"
DO_MEASURE="${DO_MEASURE:-1}"
DO_AGENT="${DO_AGENT:-1}"
DO_GARMENT="${DO_GARMENT:-1}"
DO_METRIC="${DO_METRIC:-0}"

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # .../AutoGarmentCode
SMPL2GARMENT="$(cd "$PROJECT_ROOT/.." && pwd)"        # .../smpl2garment

CLOSE_DIR="/root/wyc/data/CloSe"
NPZ_PATH="$CLOSE_DIR/data/CloSe-Di/${SAMPLE}.npz"
# DATA_DIR="$SMPL2GARMENT/data"   
DATA_DIR="/root/wyc/data/CloSe/data/CloSe-Di-render"                      # 渲染图落地目录

OUTPUT_DIR="$PROJECT_ROOT/output/CloSe/${SAMPLE}"
IMG_NAME="${SAMPLE}.png"
IMG_PATH="$DATA_DIR/$IMG_NAME"

MEASURE_BIN="$SMPL2GARMENT/GarmentMeasurements_SMPL/build/measurements"
MEASURE_DATA="$SMPL2GARMENT/GarmentMeasurements_SMPL/data_smpl"
HYBRIK_ROOT="$SMPL2GARMENT/HybrIK"
HYBRIK_OUT_DIR="$OUTPUT_DIR/hybrik"   

# conda 包装：避免污染当前 shell，用 conda run 显式指定环境
CONDA="/root/miniconda3/bin/conda run --no-capture-output -n"

# mkdir -p "$OUTPUT_DIR" "$DATA_DIR"
# echo "=========================================="
# echo " 样本: $SAMPLE   仿真: $SIM"
# echo " npz : $NPZ_PATH"
# echo " 输出: $OUTPUT_DIR"
# echo "=========================================="

if [ ! -f "$NPZ_PATH" ]; then
  echo "[FATAL] cannot find npz: $NPZ_PATH" >&2
  exit 1
fi

# ---- 1. 渲染 npz → 正面图 (close 环境, pytorch3d) ----
if [ "$DO_RENDER" = "1" ]; then
  echo "[1/7] rendering ..."
  ( cd "$CLOSE_DIR" && $CONDA close python render_native.py \
        --npz "$NPZ_PATH" --output "$DATA_DIR/$IMG_NAME" )
  cp "$DATA_DIR/$IMG_NAME" "$IMG_PATH"
  echo "      rendered image -> $DATA_DIR/$IMG_NAME (copy -> $IMG_PATH)"
else
  echo "[1/7] skipping render"
fi



# ---- 2. HybrIK 从图像估计 SMPL → smpl.json (+rendered overlay) ----
if [ "$DO_HYBRIK" = "1" ]; then
  echo "[2/7] estimating SMPL parameters with HybrIK ..."
  mkdir -p "$HYBRIK_OUT_DIR"
  ( cd "$HYBRIK_ROOT" && $CONDA hybrik python "$HYBRIK_ROOT/scripts/demo_image.py" \
        --img-path "$IMG_PATH" --out-dir "$HYBRIK_OUT_DIR" )
  echo "      smpl.json -> $HYBRIK_OUT_DIR/smpl.json"
else
  echo "[2/7] skipping HybrIK"
fi

# ---- 3. 由 betas 重建 A-pose SMPL 网格 smpl.obj (garmentcode 环境, smplx) ----
if [ "$DO_SMPLOBJ" = "1" ]; then
  echo "[3/7] exporting A-pose SMPL mesh smpl.obj ..."
  $CONDA garmentcode python "$PROJECT_ROOT/smpl_estimate/export_smpl_mesh.py" \
        --json "$HYBRIK_OUT_DIR/smpl.json" --output "$OUTPUT_DIR/smpl.obj" --gender "$GENDER"
else
  echo "[3/7] skipping smpl.obj export"
fi

# ---- 4. 量体 smpl.obj → smpl.yaml ----
if [ "$DO_MEASURE" = "1" ]; then
  echo "[4/7] measuring with GarmentMeasurements ..."
  "$MEASURE_BIN" "$OUTPUT_DIR/smpl.obj" "$OUTPUT_DIR/smpl.yaml" --data_dir "$MEASURE_DATA"
  echo "      smpl.yaml -> $OUTPUT_DIR/smpl.yaml"
else
  echo "[4/7] skipping measurement"
fi

# ---- 5. LLM 生成 design.yaml (garmentcode 环境) ----
if [ "$DO_AGENT" = "1" ]; then
  echo "[5/7] calling GPT-4o to generate design.yaml ..."
  ( cd "$PROJECT_ROOT" && $CONDA garmentcode python "$PROJECT_ROOT/work/main.py" \
        --img "$IMG_PATH" --body "$OUTPUT_DIR/smpl.yaml" --output "$OUTPUT_DIR" )
  echo "      design.yaml -> $OUTPUT_DIR/design.yaml"
else
  echo "[5/7] skipping agent"
fi

# ---- 6. GarmentCode 生成样板 + (可选) 仿真 ----
if [ "$DO_GARMENT" = "1" ]; then
  echo "[6/7] generating pattern with GarmentCode (sim=$SIM) ...  [$(date '+%H:%M:%S')]"
  _sim_start=$(date +%s)
  ( cd "$PROJECT_ROOT" && $CONDA garmentcode python "$PROJECT_ROOT/work/garmentcode.py" \
        --design_path "$OUTPUT_DIR/design.yaml" \
        --body_path "$OUTPUT_DIR/smpl.yaml" \
        --sim "$SIM" )
  _sim_elapsed=$(( $(date +%s) - _sim_start ))
  echo "      sim time: ${_sim_elapsed}s ($(( _sim_elapsed / 60 ))m $(( _sim_elapsed % 60 ))s)"
  # 写入单独的仿真耗时日志
  echo "$SAMPLE,$_sim_elapsed" >> "$PROJECT_ROOT/output/CloSe/sim_timing.csv"
else
  echo "[6/7] skipping GarmentCode"
fi

if [ -f "$IMG_PATH" ]; then
  cp "$IMG_PATH" "$OUTPUT_DIR/"
  echo "      rendered image -> $OUTPUT_DIR"
else
  echo "[ERROR] rendered image not found: $IMG_PATH" >&2
  exit 1
fi

# ---- 7. 指标评估 (论文 §4.5.1 五项指标) ----
if [ "$DO_METRIC" = "1" ]; then
  echo "[7/7] evaluating metrics ..."
  ( cd "$PROJECT_ROOT/work" && $CONDA garmentcode python compute_metrics.py \
        --single "$SAMPLE" \
        --data_root "$(dirname "$NPZ_PATH")" \
        --output_root "$PROJECT_ROOT/output/CloSe" )
else
  echo "[7/7] skipping metrics"
fi

echo "=========================================="
echo " finish: $OUTPUT_DIR"
echo "=========================================="
