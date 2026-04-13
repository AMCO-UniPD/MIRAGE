import os
import csv
import cv2
import math
import time
import torch
import matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DEFAULT_DATASET = "mvtec"

# Grounding DINO model
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# YOLO26 model
MODEL_REPO = "openvision/yolo26-l-seg"
MODEL_FILE = "model.pt"


#  Timing
class TimingStats:
    """Accumulate and report per-operation wall-clock time."""

    def __init__(self):
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._t0: dict[str, float] = {}

    def start(self, op: str) -> float:
        t = time.perf_counter()
        self._t0[op] = t
        return t

    def stop(self, op: str) -> float:
        elapsed = time.perf_counter() - self._t0.pop(op, time.perf_counter())
        self._totals[op] += elapsed
        self._counts[op] += 1
        return elapsed

    def report(self, n_samples: int = 1):
        print(f"\n{'' * 60}")
        print(f"Timing report ({n_samples} samples):")
        total_wall = sum(self._totals.values())
        for op in sorted(self._totals, key=lambda k: -self._totals[k]):
            t = self._totals[op]
            c = self._counts[op]
            pct = 100 * t / (total_wall + 1e-9)
            print(
                f"  {op:<35} {t:7.2f}s  ({pct:5.1f}%)  n={c}  avg={t / max(c, 1) * 1000:.0f}ms"
            )
        print(f"  {'TOTAL':<35} {total_wall:7.2f}s")
        print(f"{'' * 60}\n")


TIMING = TimingStats()


#  Feature Cache
class FeaturesCache:
    """
    Simple LRU-style cache for G-DINO or YOLO backbone features.

    Key: (img_path, text)  : both the image content and text prompt
         affect G-DINO features (cross-attention conditioning).
    Value: whatever extract_* returns (tensor or list of tensors).

    The cache is most useful for base/reference images that are shared
    across many anomaly samples (common in MVTec-style datasets).
    """

    def __init__(self, maxsize: int = 100):
        self._cache: dict = {}
        self._order: list = []
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key):
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
        else:
            self.misses += 1
        return val

    def put(self, key, value):
        if key in self._cache:
            return
        if len(self._cache) >= self._maxsize:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        self._cache[key] = value
        self._order.append(key)

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = 100 * self.hits / max(total, 1)
        return f"cache hits={self.hits}/{total} ({rate:.1f}%)"


BASE_FEATURES_CACHE = FeaturesCache(maxsize=200)  # G-DINO: key=(path, text, layers)
YOLO_BASE_CACHE = FeaturesCache(maxsize=200)  # YOLO: key=path only (vision-only)


def _feats_to_cpu(v):
    """Move tensor(s) to CPU before caching to avoid GPU memory accumulation."""
    if isinstance(v, torch.Tensor):
        return v.cpu()
    return [f.cpu() for f in v]


def _feats_to_device(v, device):
    """Move cached CPU tensor(s) back to the compute device on retrieval."""
    if isinstance(v, torch.Tensor):
        return v.to(device)
    return [f.to(device) for f in v]


#  Shared Utilities
def sharp_fusion(backbone_hm, pixel_hm, sharp_alpha=0.5):
    absdiff_confidence = np.power(pixel_hm, 1.0 - sharp_alpha)
    sharp_fused = backbone_hm * absdiff_confidence
    sharp_fused_nonzero = sharp_fused[sharp_fused > 1e-6]
    if len(sharp_fused_nonzero) > 0:
        p_low = np.percentile(sharp_fused_nonzero, 1)
        p_high = np.percentile(sharp_fused_nonzero, 99)
        if p_high - p_low > 1e-6:
            sharp_fused = (sharp_fused - p_low) / (p_high - p_low)
            sharp_fused = np.clip(sharp_fused, 0, 1)
        else:
            sharp_fused = np.clip(sharp_fused / (sharp_fused.max() + 1e-8), 0, 1)
    return sharp_fused


def postprocessing_pipeline(
    yoloe_hm,
    backbone_name,
    use_fused,
    backbone_hm,
    img_n_np,
    img_a_np,
    postprocess,
    use_bool_fuse=False,
    use_sharp_fuse=False,
    sharp_alpha=0.5,
):
    pp_hm = None
    pp_binary = None
    if postprocess == "crf":
        pp_hm = guided_filter_refine(backbone_hm, img_a_np)

    heatmaps = [(f"{backbone_name}", backbone_hm)]
    if pp_hm is not None:
        heatmaps.append((f"{backbone_name}_{postprocess}", pp_hm))
    if use_bool_fuse:
        assert use_fused, "bool fuse requires using mode fused"
    if use_sharp_fuse:
        assert use_fused, "sharp fuse requires using mode fused"

    sharp_fused = None
    fused_hm = None

    if use_fused:
        pixel_hm = compute_pixel_absdiff(img_n_np, img_a_np)
        fused_hm = np.sqrt(backbone_hm * pixel_hm)

        if use_bool_fuse:
            pixel_mask = compute_otsu_mask(pixel_hm)
            backbone_mask = compute_otsu_mask(backbone_hm)
            and_mask = pixel_mask * backbone_mask
            fused_hm *= and_mask

        if fused_hm.max() > 1e-6:
            fused_hm = fused_hm / fused_hm.max()

        heatmaps += [("absdiff", pixel_hm), ("fused_geom", fused_hm)]

        if use_sharp_fuse:
            sharp_fused = sharp_fusion(backbone_hm, pixel_hm, sharp_alpha)

            if yoloe_hm is not None:
                yoloe_hm = sharp_fusion(yoloe_hm, pixel_hm, sharp_alpha=sharp_alpha)
                sharp_fused = sharp_fused * yoloe_hm

            heatmaps.append(("fused_sharp", sharp_fused))

    return heatmaps, pp_hm, pp_binary, sharp_fused, fused_hm


def compute_otsu_mask(hm):
    hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    local_thresh = otsu_val / 255.0
    return hm > local_thresh


def get_all_gt_samples(csv_path, dataset_root, ref_root):
    return _collect_samples(csv_path, dataset_root, ref_root, require_gt=True)


def get_all_samples(csv_path, dataset_root, ref_root):
    return _collect_samples(csv_path, dataset_root, ref_root, require_gt=False)


def sort_by_base_image(samples: list) -> list:
    """
    Sort samples so all anomalies sharing the same reference image are consecutive.

    Why: G-DINO base features (key=(path, text, layers)) benefit when the same
    base+text pair appears repeatedly. YOLO base features (key=path) benefit even
    more since text doesn't matter : all anomalies with the same base image hit the
    cache regardless of defect type.

    Secondary sort by object name keeps the output tables readable.
    """
    return sorted(
        samples, key=lambda s: (s["base_img_path"], s["object"], s["img_name"])
    )


def _collect_samples(csv_path, dataset_root, ref_root, require_gt=True):
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Collecting"):
            obj = row["object"]
            img_path = row["img_path"]
            base_img_path = row["base_img_path"]
            defect_syn = row["defect_synonym"].strip().strip('"')
            anom_full = os.path.join(dataset_root, img_path)
            base_full = os.path.join(ref_root, base_img_path)
            if not os.path.exists(anom_full) or not os.path.exists(base_full):
                continue
            img_stem = Path(img_path).stem
            mask_path = os.path.join(
                dataset_root,
                os.path.dirname(img_path),
                "masks_modify_total",
                f"{img_stem}_mask.png",
            )
            has_gt = os.path.exists(mask_path)
            if require_gt and not has_gt:
                continue
            clean = defect_syn
            if "based on this image." in clean:
                clean = clean.split("based on this image.")[-1].strip()
            general_defect = clean.split(":")[0].strip()
            samples.append(
                {
                    "object": obj,
                    "base_img": base_full,
                    "base_img_path": base_full,  # used as cache key
                    "anom_img": anom_full,
                    "mask": mask_path if has_gt else None,
                    "img_name": img_stem,
                    "general_defect": general_defect,
                }
            )
    return samples


def compute_pixel_absdiff(img_n_np, img_a_np, blur_ksize=15, blur_sigma=4.0):
    diff = np.abs(img_n_np.astype(np.float32) - img_a_np.astype(np.float32))
    diff_gray = diff.mean(axis=2)
    diff_gray = cv2.GaussianBlur(diff_gray, (blur_ksize, blur_ksize), blur_sigma)
    mn, mx = diff_gray.min(), diff_gray.max()
    if mx - mn > 1e-6:
        diff_gray = (diff_gray - mn) / (mx - mn)
    else:
        diff_gray = np.zeros_like(diff_gray)
    return diff_gray.astype(np.float32)


def best_f1(heatmap, gt_binary, n_thresholds=50):
    """
    Fast best-F1 using sorted cumulative TP/FP sweep.

    O(N log N) sort + O(N) sweep : evaluates ALL pixel values as candidate
    thresholds rather than just 50 fixed ones, and avoids large matrix allocations.
    Much faster than the original Python loop and avoids the memory-bandwidth
    penalty of the T×N numpy broadcast approach on large images.
    """
    gt_flat = gt_binary.flatten().astype(bool)
    hm_flat = heatmap.flatten()

    n_pos = int(gt_flat.sum())
    if n_pos == 0 or n_pos == len(gt_flat):
        return 0.0, 0.5

    # Sort descending by score : sweep from highest to lowest threshold
    order = np.argsort(-hm_flat)
    sorted_gt = gt_flat[order]
    sorted_hm = hm_flat[order]

    # Cumulative TP and FP as we include more pixels (lower threshold)
    tp_cum = np.cumsum(sorted_gt).astype(np.float32)
    fp_cum = np.cumsum(~sorted_gt).astype(np.float32)
    fn_cum = (n_pos - tp_cum).astype(np.float32)

    prec = tp_cum / (tp_cum + fp_cum + 1e-8)
    rec = tp_cum / (tp_cum + fn_cum + 1e-8)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)

    best_idx = int(f1_scores.argmax())
    return float(f1_scores[best_idx]), float(sorted_hm[best_idx])


#  Edge-Aware CRF-like Refinement
def _renormalize(hm):
    mn, mx = hm.min(), hm.max()
    if mx - mn > 1e-6:
        return ((hm - mn) / (mx - mn)).astype(np.float32)
    return np.zeros_like(hm)


def guided_filter_refine(
    heatmap, image_rgb, r_coarse=16, eps_coarse=0.1, r_fine=4, eps_fine=0.001
):
    guide = image_rgb.astype(np.float32)
    step1 = cv2.ximgproc.guidedFilter(
        guide, heatmap.astype(np.float32), r_coarse, eps_coarse
    )
    step1 = _renormalize(np.clip(step1, 0, None))
    step2 = cv2.ximgproc.guidedFilter(guide, step1, r_fine, eps_fine)
    return _renormalize(np.clip(step2, 0, None))


#  Grounding DINO Feature Extraction
def get_spatial_sizes(model, processor, img_pil, text, device):
    inputs = processor(images=img_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_features, _ = model.model.backbone(
            inputs["pixel_values"], inputs.get("pixel_mask", None)
        )
    sizes = [(feat.shape[2], feat.shape[3]) for feat, _ in vision_features]
    last_h, last_w = sizes[-1]
    extra_h = math.ceil(last_h / 2)
    extra_w = math.ceil(last_w / 2)
    sizes.append((extra_h, extra_w))
    return sizes


@torch.no_grad()
def extract_text_conditioned_features(
    model, processor, img_pil, text, device, layers=None
):
    """Single-image feature extraction (kept for compatibility / cache fill)."""
    inputs = processor(images=img_pil, text=text, return_tensors="pt").to(device)

    if layers is None:
        outputs = model.model(**inputs, output_hidden_states=False)
        return outputs.encoder_last_hidden_state_vision
    else:
        outputs = model.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.encoder_vision_hidden_states
        return [hidden_states[li] for li in layers]


@torch.no_grad()
def extract_text_conditioned_features_batched(
    model, processor, img_n_pil, img_a_pil, text, device, layers=None
):
    """
    OPTIMIZED: extract features for BOTH images in a single forward pass.

    Instead of two calls to the transformer (2x encoder cost), we batch
    [img_n, img_a] with [text, text] and split the output.

    Returns: (feats_n, feats_a) : same shapes as single-image extraction.
    """
    TIMING.start("gdino_preprocess")
    inputs = processor(
        images=[img_n_pil, img_a_pil],
        text=[text, text],
        return_tensors="pt",
        padding=True,
    ).to(device)
    TIMING.stop("gdino_preprocess")

    TIMING.start("gdino_forward_batched")
    if layers is None:
        outputs = model.model(**inputs, output_hidden_states=False)
        feats = outputs.encoder_last_hidden_state_vision  # (2, seq_len, D)
        TIMING.stop("gdino_forward_batched")
        return feats[0:1], feats[1:2]
    else:
        outputs = model.model(**inputs, output_hidden_states=True)
        hidden = outputs.encoder_vision_hidden_states  # tuple of (2, seq_len, D)
        TIMING.stop("gdino_forward_batched")
        feats_n = [hidden[li][0:1] for li in layers]
        feats_a = [hidden[li][1:2] for li in layers]
        return feats_n, feats_a


def split_multiscale_features(features, spatial_sizes):
    feat = features.squeeze(0)
    maps = []
    offset = 0
    for h, w in spatial_sizes:
        n = h * w
        level_feat = feat[offset : offset + n]
        level_map = level_feat.reshape(h, w, -1)
        maps.append(level_map)
        offset += n
    return maps


def _maps_from_feats(feats, spatial_sizes, layers):
    """Convert raw feature tensors → list of per-scale maps."""
    if layers is None:
        return split_multiscale_features(feats, spatial_sizes)
    else:
        all_maps = [split_multiscale_features(fn, spatial_sizes) for fn in feats]
        num_scales = len(spatial_sizes)
        maps = []
        for scale_idx in range(num_scales):
            scale_maps = torch.stack(
                [all_maps[li][scale_idx] for li in range(len(layers))]
            )
            maps.append(scale_maps.mean(dim=0))
        return maps


def compute_grounding_dino_heatmap_fast(
    img_n_pil,
    img_a_pil,
    model,
    processor,
    device,
    text,
    spatial_sizes=None,
    layers=None,
    base_img_path: str | None = None,
):
    """
    OPTIMIZED compute_grounding_dino_heatmap with:
      - Batched inference (normal+anomaly in one forward pass)
      - Base-image feature caching (reuse if same reference was seen before)
    """
    if spatial_sizes is None:
        spatial_sizes = get_spatial_sizes(model, processor, img_n_pil, text, device)

    #  Try to load base features from cache
    cache_key = (base_img_path, text, tuple(layers) if layers else None)
    cached_n = BASE_FEATURES_CACHE.get(cache_key) if base_img_path else None

    if cached_n is not None:
        # Cache hit: only run one forward pass for the anomaly image
        print("[cache-hit]", end=" ", flush=True)
        TIMING.start("gdino_preprocess")
        inputs_a = processor(images=img_a_pil, text=text, return_tensors="pt").to(
            device
        )
        TIMING.stop("gdino_preprocess")

        TIMING.start("gdino_forward_anomaly_only")
        with torch.no_grad():  # must be explicit here since we're not in the decorator
            if layers is None:
                outputs = model.model(**inputs_a, output_hidden_states=False)
                feats_a = outputs.encoder_last_hidden_state_vision.detach()
            else:
                outputs = model.model(**inputs_a, output_hidden_states=True)
                hidden = outputs.encoder_vision_hidden_states
                feats_a = [hidden[li].detach() for li in layers]
        TIMING.stop("gdino_forward_anomaly_only")

        feats_n = _feats_to_device(cached_n, device)
    else:
        # Cache miss: batch both images in one forward pass
        feats_n, feats_a = extract_text_conditioned_features_batched(
            model, processor, img_n_pil, img_a_pil, text, device, layers=layers
        )
        # Store on CPU so cached tensors don't accumulate in GPU VRAM
        if base_img_path:
            raw = (
                feats_n.detach()
                if isinstance(feats_n, torch.Tensor)
                else [f.detach() for f in feats_n]
            )
            BASE_FEATURES_CACHE.put(cache_key, _feats_to_cpu(raw))

    #  Build per-scale maps and compute distance
    TIMING.start("gdino_postprocess")
    maps_n = _maps_from_feats(feats_n, spatial_sizes, layers)
    maps_a = _maps_from_feats(feats_a, spatial_sizes, layers)

    target_h, target_w = spatial_sizes[0]
    combined = np.zeros((target_h, target_w), dtype=np.float32)

    for mn, ma in zip(maps_n, maps_a):
        mn_norm = F.normalize(mn.float(), p=2, dim=-1)
        ma_norm = F.normalize(ma.float(), p=2, dim=-1)

        diff = mn_norm - ma_norm
        global_mean = diff.mean(dim=(0, 1), keepdim=True)
        diff = diff - global_mean

        dist = diff.norm(p=2, dim=-1).cpu().numpy()

        mn_val, mx_val = dist.min(), dist.max()
        if mx_val - mn_val > 1e-6:
            dist = (dist - mn_val) / (mx_val - mn_val)
        else:
            dist = np.zeros_like(dist)

        if dist.shape != (target_h, target_w):
            dist = cv2.resize(
                dist, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        combined += dist

    combined /= len(spatial_sizes)
    combined = cv2.GaussianBlur(combined.astype(np.float32), (15, 15), 4.0)
    mn_val, mx_val = combined.min(), combined.max()
    if mx_val - mn_val > 1e-6:
        combined = (combined - mn_val) / (mx_val - mn_val)
    TIMING.stop("gdino_postprocess")

    return combined.astype(np.float32)


#  YOLO26 Feature Extraction
class YOLO26FeatureExtractor:
    def __init__(self, model, device):
        self.model = model.model
        self.device = device
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach()
                elif isinstance(output, (list, tuple)):
                    self.features[name] = [
                        o.detach() if isinstance(o, torch.Tensor) else o for o in output
                    ]

            return hook

        if hasattr(self.model, "model"):
            for i, module in enumerate(self.model.model):
                if i < 12:
                    module.register_forward_hook(get_activation(f"layer_{i}"))

    def _tensor_to_batch(self, image_pil, target_size=(640, 640)):
        img_resized = image_pil.resize(target_size)
        img_np = np.array(img_resized)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_features(self, image_pil, target_size=(640, 640)):
        """Single-image feature extraction."""
        self.features = {}
        img_tensor = self._tensor_to_batch(image_pil, target_size)
        _ = self.model(img_tensor)
        return self._collect_feature_maps()

    @torch.no_grad()
    def extract_features_pair(self, img_n_pil, img_a_pil, target_size=(640, 640)):
        """
        OPTIMIZED: extract features for both images in ONE forward pass.
        Returns: (feats_n, feats_a) each a list of 4D tensors (1, C, H, W).
        """
        self.features = {}
        t_n = self._tensor_to_batch(img_n_pil, target_size)
        t_a = self._tensor_to_batch(img_a_pil, target_size)
        batch = torch.cat([t_n, t_a], dim=0)  # (2, C, H, W)
        _ = self.model(batch)

        # Split batch dimension for each hooked feature map
        feats_n, feats_a = [], []
        for key in sorted(self.features.keys()):
            feat = self.features[key]
            if isinstance(feat, torch.Tensor) and feat.ndim == 4:
                feats_n.append(feat[0:1])
                feats_a.append(feat[1:2])
            elif isinstance(feat, list):
                for f in feat:
                    if isinstance(f, torch.Tensor) and f.ndim == 4:
                        feats_n.append(f[0:1])
                        feats_a.append(f[1:2])
        return feats_n, feats_a

    def _collect_feature_maps(self):
        feature_maps = []
        for key in sorted(self.features.keys()):
            feat = self.features[key]
            if isinstance(feat, torch.Tensor) and len(feat.shape) == 4:
                feature_maps.append(feat)
            elif isinstance(feat, list):
                for f in feat:
                    if isinstance(f, torch.Tensor) and len(f.shape) == 4:
                        feature_maps.append(f)
        return feature_maps


def _compute_dist_maps_from_pairs(feats_n, feats_a):
    """Shared distance computation for both G-DINO and YOLO feature lists."""
    target_h, target_w = feats_n[0].shape[2:]
    combined = np.zeros((target_h, target_w), dtype=np.float32)
    num_levels = len(feats_n)

    for fn, fa in zip(feats_n, feats_a):
        fn = fn.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        fa = fa.squeeze(0).permute(1, 2, 0)

        fn_norm = F.normalize(fn.float(), p=2, dim=-1)
        fa_norm = F.normalize(fa.float(), p=2, dim=-1)

        diff = fn_norm - fa_norm
        diff = diff - diff.mean(dim=(0, 1), keepdim=True)
        dist = diff.norm(p=2, dim=-1).cpu().numpy()

        mn_val, mx_val = dist.min(), dist.max()
        if mx_val - mn_val > 1e-6:
            dist = (dist - mn_val) / (mx_val - mn_val)
        else:
            dist = np.zeros_like(dist)

        if dist.shape != (target_h, target_w):
            dist = cv2.resize(
                dist, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        combined += dist

    combined /= num_levels
    combined = cv2.GaussianBlur(combined.astype(np.float32), (15, 15), 4.0)
    mn_val, mx_val = combined.min(), combined.max()
    if mx_val - mn_val > 1e-6:
        combined = (combined - mn_val) / (mx_val - mn_val)
    return combined.astype(np.float32)


def compute_yolo26_heatmap_fast(
    img_n_pil,
    img_a_pil,
    feature_extractor,
    target_size=(640, 640),
    base_img_path: str | None = None,
):
    """
    OPTIMIZED: batch both images in one YOLO forward pass.
    Also caches base-image features keyed by path (YOLO is vision-only, not
    text-conditioned, so same base → same features regardless of defect text).
    """
    cached_n = YOLO_BASE_CACHE.get(base_img_path) if base_img_path else None

    if cached_n is not None:
        # Cache hit: only run anomaly image through YOLO
        print("[yolo-cache-hit]", end=" ", flush=True)
        TIMING.start("yolo_forward_anomaly_only")
        feats_a = feature_extractor.extract_features(img_a_pil, target_size)
        TIMING.stop("yolo_forward_anomaly_only")
        feats_n = _feats_to_device(cached_n, feature_extractor.device)
    else:
        TIMING.start("yolo_forward_batched")
        feats_n, feats_a = feature_extractor.extract_features_pair(
            img_n_pil, img_a_pil, target_size
        )
        TIMING.stop("yolo_forward_batched")
        if base_img_path:
            YOLO_BASE_CACHE.put(base_img_path, _feats_to_cpu(feats_n))

    TIMING.start("yolo_postprocess")
    result = _compute_dist_maps_from_pairs(feats_n, feats_a)
    TIMING.stop("yolo_postprocess")
    return result


#  Main
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bool-fuse", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--sharp-fuse",
        action=argparse.BooleanOptionalAction,
        help="Absdiff-gated modulation: suppress G-DINO where absdiff is low",
    )
    parser.add_argument(
        "--yoloe-gating",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sharp-alpha",
        type=float,
        default=0.5,
        help="Gating strength (0=harsh gating by absdiff, 1=no gating, keep G-DINO)",
    )
    parser.add_argument(
        "--backbone",
        choices=["gdino", "yoloe"],
        default="gdino",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "fused"],
        default="base",
    )
    parser.add_argument(
        "--postprocess",
        choices=["none", "crf"],
        default="none",
    )
    parser.add_argument("--global-thresh", type=float, default=None)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already processed in the output CSV",
    )
    parser.add_argument(
        "--generate-all-masks",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable base-image feature caching (disabled by default to avoid GPU OOM)",
    )
    parser.add_argument(
        "--timing-every",
        type=int,
        default=10,
        help="Print timing report every N samples (0 = only at end)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization entirely (much faster; still saves NPZ and CSV)",
    )
    parser.add_argument(
        "--fast-viz",
        action="store_true",
        help="Lightweight visualization: save a small 3-panel JPEG "
        "(base | anomaly+heatmap | GT if available) for the final heatmap only. "
        "Much faster than the full matplotlib grid. Implies --no-viz for other heatmaps.",
    )
    parser.add_argument(
        "--fast-viz-size",
        type=int,
        default=256,
        help="Downsample long edge to this size for --fast-viz (default 256px)",
    )
    parser.add_argument(
        "--metrics-only-final",
        action="store_true",
        help="Only compute AUC/AP/F1 for the final heatmap (fused_sharp or backbone), "
        "skip intermediate heatmaps. Saves significant time in metrics loop.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Disable automatic sorting of samples by base image (disables cache ordering)",
    )
    parser.add_argument(
        "--no-develop",
        action="store_true",
        help="Write outputs to outputs/<dataset>/... instead of outputs/develop/<dataset>/... "
        "This matches the original script's paths exactly, so --resume can pick up "
        "where grounding_dino_diff.py left off.",
    )
    args = parser.parse_args()

    if args.backbone == "yoloe" and args.layers is not None:
        print("! warning: YOLO26 ignores --layers option")
        args.layers = None

    use_cache = args.use_cache
    yoloe_gating = args.yoloe_gating
    mode = args.mode
    postprocess = args.postprocess
    use_bool_fuse = args.bool_fuse
    use_sharp_fuse = args.sharp_fuse
    sharp_alpha = args.sharp_alpha
    backbone_name = args.backbone
    layers = args.layers
    resume = args.resume
    generate_all = args.generate_all_masks
    timing_every = args.timing_every
    no_viz = args.no_viz
    fast_viz = args.fast_viz
    fast_viz_size = args.fast_viz_size
    metrics_only_final = args.metrics_only_final
    do_sort = not args.no_sort

    dataset_name = args.dataset_path
    dataset_root = os.path.join(
        PROJECT_ROOT, "datasets", f"all_generated_{dataset_name}"
    )
    ref_root = os.path.join(PROJECT_ROOT, "datasets", dataset_name)
    csv_tracking_path = os.path.join(dataset_root, "generation_tracking.csv")

    #  Output dir: under outputs/develop/ to avoid clobbering live runs
    dir_suffix = (
        f"{mode}_{postprocess}" + "_PPonly" if False else ""
    )  # postprocess-only not used
    dir_suffix += "_yoloe_gating" if yoloe_gating else ""
    if use_bool_fuse:
        dir_suffix += "_bool_fuse"
    if use_sharp_fuse:
        dir_suffix += f"_sharp_fuse_a{sharp_alpha:.2f}"
    if layers is not None:
        dir_suffix += f"_layers_{'_'.join(map(str, layers))}"
    if args.no_develop:
        # Match the original script's path so --resume picks up the same CSV/NPZ files
        output_dir = os.path.join(
            PROJECT_ROOT, "outputs", dataset_name, f"{backbone_name}_diff_{dir_suffix}"
        )
    else:
        output_dir = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "develop",
            dataset_name,
            f"{backbone_name}_diff_{dir_suffix}",
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"change_detection_pipeline.py")
    print(f"  backbone={backbone_name}, mode={mode}, pp={postprocess}")
    print(
        f"  yoloe_gating={yoloe_gating}, sharp_fuse={use_sharp_fuse}, alpha={sharp_alpha}"
    )
    print(
        f"  layers={layers}, cache={'ON' if use_cache else 'OFF'}, sort={'ON' if do_sort else 'OFF'}"
    )
    print(
        f"  viz={'fast-viz' if fast_viz else ('none' if no_viz else 'full')}, metrics={'final-only' if metrics_only_final else 'all'}"
    )
    print(f"  output → {output_dir}")
    print(f"{'=' * 60}\n")

    TIMING.start("model_load")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {backbone_name} on {device}...")
    model = processor = feature_extractor = yolo_feature_extractor = None

    if backbone_name == "gdino":
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
            .to(device)
            .eval()
        )
    if backbone_name == "yoloe" or yoloe_gating:
        print(f"Downloading YOLO26 from {MODEL_REPO}...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        yolo_model = YOLO(model_path)
        yolo_model.to(device)
        if backbone_name == "yoloe":
            feature_extractor = YOLO26FeatureExtractor(yolo_model, device)
        else:
            yolo_feature_extractor = YOLO26FeatureExtractor(yolo_model, device)
    load_time = TIMING.stop("model_load")
    print(f"Model loaded in {load_time:.1f}s")

    TIMING.start("data_load")
    if generate_all:
        print(f"Collecting ALL samples from {dataset_name}...")
        samples = get_all_samples(csv_tracking_path, dataset_root, ref_root)
        n_with_gt = sum(1 for s in samples if s["mask"] is not None)
        print(f"Found {len(samples)} samples ({n_with_gt} with GT).")
    else:
        print(f"Collecting GT samples from {dataset_name}...")
        samples = get_all_gt_samples(csv_tracking_path, dataset_root, ref_root)
        print(f"Found {len(samples)} samples with GT.")
    TIMING.stop("data_load")

    if do_sort:
        samples = sort_by_base_image(samples)
        print(f"Sorted {len(samples)} samples by base image (maximises cache hits).")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        print(f"Limited to {len(samples)} samples.")

    spatial_sizes = None
    last_img_size = None

    use_fused = mode in ("fused", "all")
    pp_label = {"none": "Raw", "crf": "CRF"}[postprocess]

    #  CSV setup
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_fields = [
        "object",
        "img_name",
        "defect",
        f"auc_{backbone_name}",
        f"ap_{backbone_name}",
        f"f1_{backbone_name}",
        f"f1_otsu_{backbone_name}",
    ]
    if postprocess != "none":
        csv_fields += [
            f"auc_{backbone_name}_{postprocess}",
            f"ap_{backbone_name}_{postprocess}",
            f"f1_{backbone_name}_{postprocess}",
            f"f1_otsu_{backbone_name}_{postprocess}",
        ]
    if use_fused:
        csv_fields += [
            "auc_absdiff",
            "ap_absdiff",
            "f1_absdiff",
            "f1_otsu_absdiff",
            "auc_fused_geom",
            "ap_fused_geom",
            "f1_fused_geom",
            "f1_otsu_fused_geom",
        ]
        if use_sharp_fuse:
            csv_fields += [
                "auc_fused_sharp",
                "ap_fused_sharp",
                "f1_fused_sharp",
                "f1_otsu_fused_sharp",
            ]

    npz_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(npz_dir, exist_ok=True)

    already_done = set()
    if resume and os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add((row["object"], row["img_name"]))
        print(f"Resume: {len(already_done)} already processed.")
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    print(f"Running: mode={mode}, postprocess={postprocess}\n")

    # Name of the final heatmap : used for per-sample AUC print and primary metric filter
    final_name = (
        "fused_sharp"
        if use_sharp_fuse
        else ("fused_geom" if use_fused else backbone_name)
    )

    def _thumb(img_np, size=256):
        h, w = img_np.shape[:2]
        scale = size / max(h, w)
        return cv2.resize(
            img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    def _label(img, text):
        out = img.copy()
        font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        (tw, tht), bl = cv2.getTextSize(text, font, sc, th)
        cv2.rectangle(out, (0, 0), (tw + 6, tht + bl + 6), (0, 0, 0), -1)
        cv2.putText(out, text, (3, tht + 3), font, sc, (255, 255, 255), th, cv2.LINE_AA)
        return out

    results = []
    n_processed = 0

    for i, sample in enumerate(samples):
        t_sample_start = time.perf_counter()
        obj = sample["object"]
        img_name = sample["img_name"]
        defect = sample["general_defect"]
        text = f"{defect}. defect. damage."
        has_gt = sample["mask"] is not None
        base_img_path = sample["base_img_path"] if use_cache else None

        print(
            f'[{i + 1}/{len(samples)}] {obj}/{img_name} text="{defect}"...',
            end=" ",
            flush=True,
        )

        if resume and (obj, img_name) in already_done:
            print("skip (resume)")
            continue

        #  Image loading
        TIMING.start("img_load")
        img_n = Image.open(sample["base_img"]).convert("RGB")
        img_a = Image.open(sample["anom_img"]).convert("RGB")
        TIMING.stop("img_load")

        if has_gt:
            TIMING.start("mask_load")
            gt_mask = np.array(Image.open(sample["mask"]).convert("L"))
            gt_h, gt_w = gt_mask.shape
            gt_binary = (gt_mask > 127).astype(np.uint8)
            TIMING.stop("mask_load")
            if gt_binary.sum() == 0 or gt_binary.sum() == gt_binary.size:
                # GT is degenerate (possibly mislabelled) : still run the model,
                # just treat it as no-GT so metrics are skipped.
                bad_gt_mask = gt_mask  # keep for BAD_GT viz before nullifying
                gt_mask = None
                gt_binary = None
                has_gt = False
            else:
                bad_gt_mask = None
        else:
            gt_mask = None
            gt_binary = None
            gt_w, gt_h = img_a.size
            bad_gt_mask = None

        img_n_np = np.array(img_n.resize((gt_w, gt_h)))
        img_a_np = np.array(img_a.resize((gt_w, gt_h)))

        #  BAD_GT: save viz for degenerate GT masks
        if bad_gt_mask is not None:
            bad_gt_obj_dir = os.path.join(output_dir, "BAD_GT", obj)
            os.makedirs(bad_gt_obj_dir, exist_ok=True)
            th_h, tw_w = _thumb(img_a_np).shape[:2]
            gt_small = cv2.resize(
                bad_gt_mask, (tw_w, th_h), interpolation=cv2.INTER_NEAREST
            )
            thumb_gt = np.stack([gt_small] * 3, axis=-1)
            panels = [
                _label(_thumb(img_n_np), "normal"),
                _label(_thumb(img_a_np), "anomaly"),
                _label(thumb_gt, "GT (degenerate)"),
            ]
            max_h = max(p.shape[0] for p in panels)
            strip = np.hstack(
                [
                    np.vstack(
                        [p, np.zeros((max_h - p.shape[0], p.shape[1], 3), np.uint8)]
                    )
                    if p.shape[0] < max_h
                    else p
                    for p in panels
                ]
            )
            Image.fromarray(strip).save(
                os.path.join(bad_gt_obj_dir, f"{img_name}.jpg"), "JPEG", quality=85
            )

        #  Backbone heatmap
        yolo_hm = None
        TIMING.start("backbone_total")

        if backbone_name == "gdino":
            # Recompute spatial sizes when image size changes
            if img_n.size != last_img_size:
                TIMING.start("gdino_spatial_sizes")
                spatial_sizes = get_spatial_sizes(model, processor, img_n, text, device)
                last_img_size = img_n.size
                TIMING.stop("gdino_spatial_sizes")
                print(f"(spatial sizes: {spatial_sizes})", end=" ")

            backbone_hm = compute_grounding_dino_heatmap_fast(
                img_n,
                img_a,
                model,
                processor,
                device,
                text,
                spatial_sizes=spatial_sizes,
                layers=layers,
                base_img_path=base_img_path,
            )
            backbone_hm = cv2.resize(
                backbone_hm, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR
            )

        if backbone_name == "yoloe" or yoloe_gating:
            fe = feature_extractor if not yoloe_gating else yolo_feature_extractor
            yolo_hm = compute_yolo26_heatmap_fast(
                img_n,
                img_a,
                fe,
                base_img_path=base_img_path if use_cache else None,
            )
            yolo_hm = cv2.resize(yolo_hm, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
            if backbone_name == "yoloe":
                backbone_hm = yolo_hm

        TIMING.stop("backbone_total")

        #  Post-processing
        TIMING.start("postprocess")
        heatmaps, pp_hm, pp_binary, sharp_fused, fused_hm = postprocessing_pipeline(
            yolo_hm,
            backbone_name,
            use_fused,
            backbone_hm,
            img_n_np,
            img_a_np,
            postprocess,
            use_bool_fuse,
            use_sharp_fuse,
            sharp_alpha,
        )
        TIMING.stop("postprocess")

        #  Metrics
        TIMING.start("metrics")
        row = {"object": obj, "img_name": img_name, "defect": defect}
        # Cache oracle results so visualization reuses them instead of recomputing
        oracle_cache: dict[
            str, tuple
        ] = {}  # name -> (oracle_f1, oracle_t, oracle_mask)
        # When --metrics-only-final: only compute for the final heatmap
        heatmaps_for_metrics = heatmaps
        if metrics_only_final:
            heatmaps_for_metrics = [
                (n, h) for n, h in heatmaps if n == final_name
            ] or heatmaps[-1:]
        if has_gt:
            for name, hm in heatmaps_for_metrics:
                gt_flat = gt_binary.flatten()
                hm_flat = hm.flatten()
                try:
                    auc = roc_auc_score(gt_flat, hm_flat)
                    ap = average_precision_score(gt_flat, hm_flat)
                except ValueError:
                    auc, ap = float("nan"), float("nan")
                f1, oracle_t = best_f1(hm, gt_binary)
                row[f"auc_{name}"] = auc
                row[f"ap_{name}"] = ap
                row[f"f1_{name}"] = f1

                # Cache oracle mask for viz reuse
                oracle_mask_arr = (hm > oracle_t).astype(np.uint8)
                oracle_cache[name] = (f1, oracle_t, oracle_mask_arr)

                hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
                otsu_val, _ = cv2.threshold(
                    hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                local_thresh = otsu_val / 255.0
                pred_otsu = hm > local_thresh
                gt_b = gt_binary.astype(bool)
                tp_o = int((pred_otsu & gt_b).sum())
                fp_o = int((pred_otsu & ~gt_b).sum())
                fn_o = int((~pred_otsu & gt_b).sum())
                prec_o = tp_o / (tp_o + fp_o + 1e-8)
                rec_o = tp_o / (tp_o + fn_o + 1e-8)
                f1_otsu_val = float(2 * prec_o * rec_o / (prec_o + rec_o + 1e-8))
                row[f"f1_otsu_{name}"] = f1_otsu_val

            results.append(row)
        else:
            for name, hm in heatmaps:
                row[f"auc_{name}"] = float("nan")
                row[f"ap_{name}"] = float("nan")
                row[f"f1_{name}"] = float("nan")
                row[f"f1_otsu_{name}"] = float("nan")
        TIMING.stop("metrics")

        with open(csv_path, "a", newline="") as f:
            for key in row.keys():
                if key not in csv_fields:
                    csv_fields.append(key)
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        #  Final heatmap
        if pp_hm is not None:
            final_hm = pp_hm
        elif use_sharp_fuse and sharp_fused is not None:
            final_hm = sharp_fused
        elif use_fused and fused_hm is not None:
            final_hm = fused_hm
        else:
            final_hm = backbone_hm

        #  Save NPZ
        TIMING.start("save_npz")
        npz_obj_dir = os.path.join(npz_dir, obj)
        os.makedirs(npz_obj_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(npz_obj_dir, f"{img_name}.npz"), heatmap=final_hm
        )
        TIMING.stop("save_npz")

        #  Visualization
        if no_viz and not fast_viz:
            t_sample = time.perf_counter() - t_sample_start
            auc_str = (
                f"AUC={row.get(f'auc_{final_name}', float('nan')):.3f}"
                if has_gt
                else "no GT"
            )
            print(f"{auc_str}  [{t_sample:.1f}s/sample]", flush=True)
            n_processed += 1
            if (
                not args.no_develop
                and timing_every > 0
                and n_processed % timing_every == 0
            ):
                TIMING.report(n_processed)
                print(BASE_FEATURES_CACHE.stats())
                print(
                    YOLO_BASE_CACHE.stats()
                    if yoloe_gating or backbone_name == "yoloe"
                    else ""
                )
            continue

        TIMING.start("visualization")
        obj_dir = os.path.join(output_dir, obj)
        os.makedirs(obj_dir, exist_ok=True)

        if fast_viz:
            # Lightweight JPEG: normal | anomaly | heatmap [| GT | best-F1 mask] with labels
            thumb_n = _thumb(img_n_np, fast_viz_size)
            thumb_a = _thumb(img_a_np, fast_viz_size)
            th, tw = thumb_a.shape[:2]

            # Heatmap as standalone colourmap panel (no overlay)
            hm_small = cv2.resize(final_hm, (tw, th), interpolation=cv2.INTER_LINEAR)
            hm_colored = cv2.applyColorMap(
                (np.clip(hm_small, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            thumb_hm = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

            if has_gt and gt_mask is not None and final_name in oracle_cache:
                oracle_f1, _, oracle_mask_arr = oracle_cache[final_name]
                # GT mask panel
                gt_small = cv2.resize(
                    (gt_mask > 127).astype(np.uint8) * 255,
                    (tw, th),
                    interpolation=cv2.INTER_NEAREST,
                )
                thumb_gt = np.stack([gt_small] * 3, axis=-1)
                # Best-F1 mask panel
                mask_small = cv2.resize(
                    oracle_mask_arr.astype(np.uint8) * 255,
                    (tw, th),
                    interpolation=cv2.INTER_NEAREST,
                )
                thumb_f1 = np.stack([mask_small] * 3, axis=-1)
                panels = [
                    _label(thumb_n, "normal"),
                    _label(thumb_a, "anomaly"),
                    _label(thumb_hm, "heatmap"),
                    _label(thumb_gt, "GT"),
                    _label(thumb_f1, f"best-F1 {oracle_f1:.2f}"),
                ]
            else:
                panels = [
                    _label(thumb_n, "normal"),
                    _label(thumb_a, "anomaly"),
                    _label(thumb_hm, "heatmap"),
                ]

            # Pad all to same height and concatenate horizontally
            max_h = max(p.shape[0] for p in panels)
            padded = []
            for p in panels:
                if p.shape[0] < max_h:
                    pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                    p = np.vstack([p, pad])
                padded.append(p)
            strip = np.hstack(padded)

            out_path = os.path.join(obj_dir, f"{img_name}_{backbone_name}_diff.jpg")
            Image.fromarray(strip).save(out_path, "JPEG", quality=85)
            TIMING.stop("visualization")

            t_sample = time.perf_counter() - t_sample_start
            auc_str = (
                f"AUC={row.get(f'auc_{final_name}', float('nan')):.3f}"
                if has_gt
                else "no GT"
            )
            print(f"{auc_str}  [{t_sample:.1f}s/sample]", flush=True)
            n_processed += 1
            if (
                not args.no_develop
                and timing_every > 0
                and n_processed % timing_every == 0
            ):
                TIMING.report(n_processed)
                print(BASE_FEATURES_CACHE.stats())
                print(
                    YOLO_BASE_CACHE.stats()
                    if yoloe_gating or backbone_name == "yoloe"
                    else ""
                )
            continue

        obj_dir = os.path.join(
            output_dir, obj
        )  # already created above for fast-viz, safe to repeat
        os.makedirs(obj_dir, exist_ok=True)

        hm_titles_map = {
            backbone_name: f'{backbone_name}\n"{defect}"',
            f"{backbone_name}_{postprocess}": f"{backbone_name}+{pp_label}",
            "absdiff": "Pixel AbsDiff",
            "fused_geom": "Fused (Geom)",
            "fused_sharp": f"Gated (α={sharp_alpha:.1f})\n{backbone_name}×AbsDiff^{1 - sharp_alpha:.1f}",
        }

        if has_gt:
            viz = []
            for name, hm in heatmaps:
                hm_title = hm_titles_map.get(name, name)
                if name == f"{backbone_name}_{postprocess}" and pp_binary is not None:
                    oracle_mask = pp_binary.astype(np.uint8)
                    tp = int((pp_binary.astype(bool) & gt_binary.astype(bool)).sum())
                    fp = int((pp_binary.astype(bool) & ~gt_binary.astype(bool)).sum())
                    fn = int((~pp_binary.astype(bool) & gt_binary.astype(bool)).sum())
                    oracle_f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))
                    oracle_title = f"Mask\nF1={oracle_f1:.3f}"
                else:
                    # Reuse cached oracle results from metrics loop : no recomputation!
                    oracle_f1, oracle_t, oracle_mask = oracle_cache[name]
                    oracle_title = f"Oracle\nF1={oracle_f1:.3f} t={oracle_t:.2f}"

                hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
                otsu_val, _ = cv2.threshold(
                    hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                local_t = otsu_val / 255.0
                otsu_mask = (hm > local_t).astype(np.uint8)
                otsu_f1 = row[f"f1_otsu_{name}"]
                otsu_title = f"Otsu\nF1={otsu_f1:.3f} t={local_t:.2f}"
                viz.append(
                    (hm_title, hm, oracle_mask, oracle_title, otsu_mask, otsu_title)
                )

            ncols = max(3, len(viz))
            fig, axes = plt.subplots(4, ncols, figsize=(4 * ncols, 16))
            axes[0, 0].imshow(img_n_np)
            axes[0, 0].set_title("Base", fontsize=9)
            axes[0, 0].axis("off")
            axes[0, 1].imshow(img_a_np)
            axes[0, 1].set_title("Anomaly", fontsize=9)
            axes[0, 1].axis("off")
            axes[0, 2].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
            axes[0, 2].set_title("GT Mask", fontsize=9)
            axes[0, 2].axis("off")
            for j in range(3, ncols):
                axes[0, j].axis("off")
            for j, (hm_title, hm, _, _, _, _) in enumerate(viz):
                axes[1, j].imshow(img_a_np)
                axes[1, j].imshow(hm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                axes[1, j].set_title(hm_title, fontsize=8)
                axes[1, j].axis("off")
            for j in range(len(viz), ncols):
                axes[1, j].axis("off")
            for j, (_, _, oracle_mask, oracle_title, _, _) in enumerate(viz):
                axes[2, j].imshow(oracle_mask, cmap="gray", vmin=0, vmax=1)
                axes[2, j].set_title(oracle_title, fontsize=8)
                axes[2, j].axis("off")
            for j in range(len(viz), ncols):
                axes[2, j].axis("off")
            for j, (_, _, _, _, otsu_mask, otsu_title) in enumerate(viz):
                axes[3, j].imshow(otsu_mask, cmap="gray", vmin=0, vmax=1)
                axes[3, j].set_title(otsu_title, fontsize=8)
                axes[3, j].axis("off")
            for j in range(len(viz), ncols):
                axes[3, j].axis("off")
        else:
            viz = []
            for name, hm in heatmaps:
                hm_title = hm_titles_map.get(name, name)
                hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
                otsu_val, _ = cv2.threshold(
                    hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                local_t = otsu_val / 255.0
                otsu_mask = (hm > local_t).astype(np.uint8)
                otsu_title = f"Otsu t={local_t:.2f}"
                viz.append((hm_title, hm, otsu_mask, otsu_title))

            ncols = max(2, len(viz))
            fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))
            axes[0, 0].imshow(img_n_np)
            axes[0, 0].set_title("Base", fontsize=9)
            axes[0, 0].axis("off")
            axes[0, 1].imshow(img_a_np)
            axes[0, 1].set_title("Anomaly", fontsize=9)
            axes[0, 1].axis("off")
            for j in range(2, ncols):
                axes[0, j].axis("off")
            for j, (hm_title, hm, _, _) in enumerate(viz):
                axes[1, j].imshow(img_a_np)
                axes[1, j].imshow(hm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                axes[1, j].set_title(hm_title, fontsize=8)
                axes[1, j].axis("off")
            for j in range(len(viz), ncols):
                axes[1, j].axis("off")
            for j, (_, _, otsu_mask, otsu_title) in enumerate(viz):
                axes[2, j].imshow(otsu_mask, cmap="gray", vmin=0, vmax=1)
                axes[2, j].set_title(otsu_title, fontsize=8)
                axes[2, j].axis("off")
            for j in range(len(viz), ncols):
                axes[2, j].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(obj_dir, f"{img_name}_{backbone_name}_diff.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()
        TIMING.stop("visualization")

        #  Per-sample timing summary
        t_sample = time.perf_counter() - t_sample_start
        auc_str = (
            f"AUC={row.get(f'auc_{final_name}', float('nan')):.3f}"
            if has_gt
            else "no GT"
        )
        print(f"{auc_str}  [{t_sample:.1f}s/sample]", flush=True)

        n_processed += 1
        if not args.no_develop and timing_every > 0 and n_processed % timing_every == 0:
            TIMING.report(n_processed)
            print(BASE_FEATURES_CACHE.stats())

    #  Final timing report
    if not args.no_develop:
        print(f"\n{'=' * 60}")
        print(f"FINAL TIMING REPORT : {n_processed} samples processed")
        TIMING.report(n_processed)
        print(BASE_FEATURES_CACHE.stats())
        print(f"{'=' * 60}\n")

    #  Summary table
    # Read ALL results from CSV : includes samples processed in previous runs (--resume).
    # This ensures the final report covers the full dataset, not just this session's samples.
    all_csv_results = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for csv_row in csv.DictReader(f):
                parsed: dict = {
                    "object": csv_row.get("object", ""),
                    "img_name": csv_row.get("img_name", ""),
                    "defect": csv_row.get("defect", ""),
                }
                for col in csv_row:
                    if col not in ("object", "img_name", "defect"):
                        try:
                            parsed[col] = float(csv_row[col])
                        except (ValueError, TypeError):
                            parsed[col] = float("nan")
                all_csv_results.append(parsed)

    # Filter to rows that have GT metrics for the final heatmap
    valid = [
        r
        for r in all_csv_results
        if not np.isnan(r.get(f"auc_{final_name}", float("nan")))
    ]

    methods = [(backbone_name, f"{backbone_name} Diff")]
    if postprocess != "none":
        methods.append(
            (f"{backbone_name}_{postprocess}", f"{backbone_name}+{pp_label}")
        )
    if use_fused:
        methods += [("absdiff", "AbsDiff"), ("fused_geom", "Fused(Geom)")]
        if use_sharp_fuse:
            methods.append(("fused_sharp", f"Fused(Sharp α={sharp_alpha})"))
    # When --metrics-only-final, intermediate heatmap columns are absent from the CSV.
    # Prune methods to only those that actually have data in at least one valid row.
    if metrics_only_final:
        methods = [
            (k, l)
            for k, l in methods
            if any(not np.isnan(r.get(f"auc_{k}", float("nan"))) for r in valid)
        ]

    metrics_list = [
        ("auc", "AUC"),
        ("ap", "AP"),
        ("f1", "F1 (Oracle)"),
        ("f1_otsu", "F1 (Local Otsu)"),
    ]
    for metric_key, metric_label in metrics_list:
        print(f"\n{'=' * 80}")
        print(
            f"{metric_label} : {backbone_name} ({len(valid)} images, mode={mode}, pp={postprocess})"
        )
        print(f"{'=' * 80}")
        header = f"{'Category':<15} {'N':>4}"
        for _, label in methods:
            header += f" {label:>14}"
        print(header)
        print("-" * (20 + 15 * len(methods)))

        by_cat = {}
        for r in valid:
            by_cat.setdefault(r["object"], []).append(r)

        all_by_method = {k: [] for k, _ in methods}
        for cat_obj in sorted(by_cat.keys()):
            cr = by_cat[cat_obj]
            row_str = f"{cat_obj:<15} {len(cr):>4}"
            for key, _ in methods:
                vals = [
                    r[f"{metric_key}_{key}"]
                    for r in cr
                    if not np.isnan(r[f"{metric_key}_{key}"])
                ]
                mean_val = np.mean(vals) if vals else float("nan")
                all_by_method[key].extend(vals)
                row_str += f" {mean_val:>14.4f}"
            print(row_str)

        print("-" * (20 + 15 * len(methods)))
        row_str = f"{'OVERALL':<15} {len(valid):>4}"
        for key, _ in methods:
            mean_val = (
                np.mean(all_by_method[key]) if all_by_method[key] else float("nan")
            )
            row_str += f" {mean_val:>14.4f}"
        print(row_str)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as rpt:
        rpt.write("=" * 80 + "\n")
        rpt.write("EXPERIMENT REPORT (change_detection_pipeline.py)\n")
        rpt.write("=" * 80 + "\n\n")
        rpt.write(f"  backbone={backbone_name}, mode={mode}, pp={postprocess}\n")
        rpt.write(f"  layers={layers}, cache={use_cache}\n")
        rpt.write(f"  total samples={len(samples)}, valid={len(valid)}\n")
        rpt.write(f"  output={output_dir}\n\n")
        by_cat = {}
        for r in valid:
            by_cat.setdefault(r["object"], []).append(r)
        for metric_key, metric_label in metrics_list:
            rpt.write(f"{metric_label}\n")
            header = f"{'Category':<15} {'N':>4}"
            for _, label in methods:
                header += f" {label:>14}"
            rpt.write(header + "\n")
            all_by_method_rpt = {k: [] for k, _ in methods}
            for cat_obj in sorted(by_cat.keys()):
                cr = by_cat[cat_obj]
                row_str = f"{cat_obj:<15} {len(cr):>4}"
                for key, _ in methods:
                    vals = [
                        r[f"{metric_key}_{key}"]
                        for r in cr
                        if not np.isnan(r[f"{metric_key}_{key}"])
                    ]
                    mean_val = np.mean(vals) if vals else float("nan")
                    all_by_method_rpt[key].extend(vals)
                    row_str += f" {mean_val:>14.4f}"
                rpt.write(row_str + "\n")
            row_str = f"{'OVERALL':<15} {len(valid):>4}"
            for key, _ in methods:
                mean_val = (
                    np.mean(all_by_method_rpt[key])
                    if all_by_method_rpt[key]
                    else float("nan")
                )
                row_str += f" {mean_val:>14.4f}"
            rpt.write(row_str + "\n\n")

    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Heatmaps: {npz_dir}")
    print(f"Visualizations: {output_dir}")


if __name__ == "__main__":
    main()
