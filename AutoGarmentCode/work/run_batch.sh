#!/usr/bin/env bash
set -euo pipefai

# ---- 参数解析 ----
IMG_FOLDER="${1:-/root/wyc/code/smpl2garment/data}"
OUT_FOLDER="${2:-/root/wyc/code/smpl2garment/AutoGarmentCode/output/CloSe}"
SIM="${3:-sim}"
FORCE_RERUN="${FORCE_RERUN:-0}" # 0: 跳过已存在，1: 强制重跑
DISABLE_TERMINAL="${DISABLE_TERMINAL:-0}" # 0: 输出到终端和文件，1: 只输出到文件


# 定位单样本脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"

# 前置检查
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "[FATAL] cannot find single-sample script: $RUN_SCRIPT" >&2
  exit 1
fi
if [ ! -d "$IMG_FOLDER" ]; then
  echo "[FATAL] image directory does not exist: $IMG_FOLDER" >&2
  exit 1
fi

# 查找所有图片文件（兼容 png/jpg/jpeg，忽略大小写）
mapfile -t img_paths < <(find "$IMG_FOLDER" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort)
total=${#img_paths[@]}

if [ "$total" -eq 0 ]; then
  echo "[FATAL] cannot find any image files in $IMG_FOLDER" >&2
  exit 1
fi

# 批量渲染图缓存目录（独立隔离，不污染原数据集）
BATCH_DATA_DIR="$OUT_FOLDER/_render_cache"
mkdir -p "$BATCH_DATA_DIR" "$OUT_FOLDER"

LOG_DIR="$OUT_FOLDER/logs"
mkdir -p "$LOG_DIR"
# 日志文件名带时间戳，避免多次运行覆盖
LOG_FILE="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
# 所有输出同时写入终端和日志文件；静默模式则只写文件
if [ "$DISABLE_TERMINAL" = "1" ]; then
  exec > "$LOG_FILE" 2>&1
else
  exec > >(tee "$LOG_FILE") 2>&1
fi

# 统计变量
total_count=0
success_count=0
fail_count=0
skip_count=0
fail_list=()

# ---- 启动信息 ----
echo "=========================================="
echo " batch processing"
echo " total images: $total"
echo " input directory: $IMG_FOLDER"
echo " output directory: $OUT_FOLDER"
echo " simulation mode: $SIM"
echo "=========================================="

# ---- 遍历处理 ----
for idx in "${!img_paths[@]}"; do
    img_path="${img_paths[$idx]}"
    img_basename=$(basename "$img_path" | cut -f 1 -d '.')
    img_name=$(basename "$img_path")
    seq=$((idx + 1))

    echo ""
    echo "=============================="
    echo "[$seq/$total] process image: $img_basename"

    if [ "$FORCE_RERUN" != "1" ] && [ -d "$OUT_FOLDER/$img_basename" ]; then
        echo "skip if exists"
        skip_count=$((skip_count + 1))
        continue
    fi

    echo "image path: $img_path"

    # 1. 复制图片到批量渲染缓存目录，供 run.sh 读取
    cp -f "$img_path" "$BATCH_DATA_DIR/$img_name"

    # 2. 调用单样本 run.sh，透传所有环境变量
    # 强制关闭渲染阶段，传入自定义路径
    DO_RENDER=0 \
    DATA_DIR="$BATCH_DATA_DIR" \
    OUTPUT_ROOT="$OUT_FOLDER" \
    bash "$RUN_SCRIPT" "$img_basename" "$SIM" 2>&1 | tee "$OUT_FOLDER/$img_basename/run.log"
    
    exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        echo "sample $img_basename success"
        success_count=$((success_count + 1))
    else
        echo "sample $img_basename failed"
        fail_count=$((fail_count + 1))
        fail_list+=("$img_basename")
    fi
done

# ---- 结果汇总 ----
echo ""
echo "=========================================="
echo " batch processing complete"
echo " total: $total  success: $success_count  fail: $fail_count  skip: $skip_count"

if [ "$fail_count" -gt 0 ]; then
    echo " fail list:"
    for name in "${fail_list[@]}"; do
        echo "    - $name"
    done
fi

echo " output root: $OUT_FOLDER"
echo "=========================================="