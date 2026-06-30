#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 从 CSV 批量读取样本，调用 run_all.sh
#
# CSV 格式: sample,gender (第一行表头)
#
# 用法:
#   bash run_batch_csv.sh [csv_path]
# =============================================================================

CSV="/root/wyc/code/smpl2garmentcode2/AutoGarmentCode/data/data.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED_FILE="$SCRIPT_DIR/failed_samples.txt"
LOG_DIR="$SCRIPT_DIR/batch_logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"

# 同时输出到终端和日志
exec > >(tee "$LOG_FILE") 2>&1

if [ ! -f "$CSV" ]; then
  echo "[FATAL] CSV not found: $CSV" >&2
  exit 1
fi

# 清空失败记录
> "$FAILED_FILE"

echo "=========================================="
echo " Batch from: $CSV"
echo " Log:        $LOG_FILE"
echo "=========================================="

total=0
success=0
failed=0

# 跳过表头，逐行读取
tail -n +2 "$CSV" | while IFS=, read -r sample gender; do
  sample=$(echo "$sample" | xargs)
  gender=$(echo "$gender" | xargs)
  [ -z "$sample" ] && continue

  total=$((total + 1))
  echo ""
  echo "############################################################"
  echo "#  [$total] $sample ($gender)"
  echo "############################################################"

  if bash "$SCRIPT_DIR/run_all.sh" "$sample" "$gender"; then
    success=$((success + 1))
    echo "#  [$sample] DONE"
  else
    failed=$((failed + 1))
    echo "$sample" >> "$FAILED_FILE"
    echo "#  [$sample] FAILED" >&2
  fi
done

echo ""
echo "=========================================="
echo " Batch finished"
echo " Failed samples: $FAILED_FILE"
echo "=========================================="
