#!/bin/bash

DATASET=${1}

if [[ "$DATASET" != "mvtec" && "$DATASET" != "visa" ]]; then
  echo "Usage: bash calibrate_masks.sh [mvtec|visa]"
  exit 1
fi

uv run src/change_detection/binarize_heatmaps.py \
  --output-dir outputs/$DATASET/gdino_diff__yoloe_gating_sharp_fuse_a0.80_layers_2_3_4_5_6 \
  --dataset-path $DATASET \
  --gen-masks --resume
