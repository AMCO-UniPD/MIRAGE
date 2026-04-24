# MIRAGE: Model-agnostic Industrial Realistic Anomaly Generation and Evaluation for Visual Anomaly Detection

Generate synthetic anomaly images for Visual Anomaly Detection.

## Setup

```bash
uv sync
echo "GOOGLE_API_KEY=your_key" > .env
```

## Repository Structure

```
├ src/
│   ├ change_detection/       # Mask generation pipeline
│   ├ generation_pipeline/    # Image generation using Gemini AI
│   │   ├ main.py             # Main generation script
│   │   └ json_files/         # Config files (synonyms, counts)
│   ├ CLIP-selection/         # CLIP-based generation quality filter
│   └ survey/                 # Survey website and analysis
├ datasets/
└ outputs/
```

## Image Generation Pipeline

Generate synthetic anomaly images using Google's Gemini AI. The pipeline:
1. Loads normal images from MVTec dataset
2. Uses LLM-generated defect descriptions and synonyms
3. Creates synthetic anomalies via image-to-image generation
4. Tracks all generations in a CSV with smart resume capability

**Configuration:**
- Edit object list in `src/generation_pipeline/main.py`
- Defect synonyms: `src/generation_pipeline/json_files/mvtec_final_defect_synonyms.json`
- Target counts: `src/generation_pipeline/json_files/mvtec_num_images_per_category.json`

**Usage:**
```bash
# Dry run (preview what will be generated)
uv run src/generation_pipeline/main.py --generate_images --dry_run

# Production run (actually generate images)
uv run src/generation_pipeline/main.py --generate_images
```

The pipeline automatically resumes if interrupted and fills gaps if images are deleted.


## Heatmap generation pipeline

Use g-dino feature difference and YOLOe gating to generate anomaly heatmaps for the generated images.

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/change_detection/change_detection_pipeline.py \
    --backbone gdino --mode fused --postprocess crf \
    --sharp-fuse --sharp-alpha 0.8 --layers 2 3 4 5 6 \
    --yoloe-gating --resume --no-develop --generate-all-masks \
    --fast-viz --metrics-only-final \
    --dataset-path $DATASET
```


## CLIP-based Generation Quality Filter

Filter out poorly generated anomaly images using CLIP similarity scoring.
Flags images where the anomaly image is more similar to the normal prompt than the anomaly prompt.

**Single image pair:**
```bash
python src/CLIP-selection/CLIP-selection.py pair \
    anomaly.png normal.png \
    "This is a damaged transistor image with cracked epoxy case." \
    "This is an intact transistor image without any damage."
```

**Batch mode from tracking CSV:**
```bash
CUDA_VISIBLE_DEVICES=1 python src/CLIP-selection/CLIP-selection.py batch \
    generation_tracking.csv \
    /path/to/generated/images \
    /path/to/normal/images \
    bad_generations.log
```

CSV format expected: `category, ?, anomaly_prompt, anomaly_image_path, normal_image_path, ...`

Output: appends bad cases to the log file with similarity scores.

**As a module:**
```python
from CLIP_selection import setup_model, clip_score, is_bad_generation

model, preprocess, tokenizer, device = setup_model()
bad = is_bad_generation(a_path, n_path, prompt_anomaly, prompt_normal,
                         model, preprocess, tokenizer, device)
```


## Calibrated Mask generation pipeline

After having created the heatmaps:


**First time (or force recalibrate):**                       
```
  uv run python src/change_detection/binarize_heatmaps.py \                         
      --output-dir outputs/mvtec/gdino_diff__yoloe_gating_sharp_fuse_a0.80_layers_2_3_4_5_6 \
      --dataset-path mvtec --gen-masks
```
→ calibrates, saves thresholds.json, writes our_masks/

**Incremental (as new heatmaps arrive):**
```
  uv run python src/change_detection/binarize_heatmaps.py \
      --output-dir outputs/mvtec/... --dataset-path mvtec \
      --gen-masks --resume
```
→ loads thresholds.json, skips already-written masks

**Force recalibrate (e.g. after more GT data):**
```
  uv run python src/change_detection/binarize_heatmaps.py \
      --output-dir outputs/mvtec/... --dataset-path mvtec \
      --gen-masks --calibrate
```
→ re-runs calibration, overwrites thresholds.json
