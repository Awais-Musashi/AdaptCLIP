#!/usr/bin/env python3
from pathlib import Path
import contextlib
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import cv2
from scipy.ndimage import gaussian_filter

import adaptcliplib
from adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter, fusion_fun
from tools.utils import get_transform, normalize

QUERY_IMAGE = "/home/awais/Datasets/analyze/query.jpg"
FOLDER = "/home/awais/Datasets/gm_good_images"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

BATCH_SIZE = 256
NUM_WORKERS = 6
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Save cache to: FOLDER/cache/{paths.json, embeds.pt}
CACHE_SUBDIR = "cache"
CACHE_PATHS = "paths.json"
CACHE_EMBEDS = "embeds.pt"

# ---- AdaptCLIP one-shot config ----
CHECKPOINT_PATH = "./adaptclip_checkpoints/epoch_15.pth"
PRETRAINED_MODEL = "ViT-L/14@336px"  # or "VITB16_PLUS_240"
IMAGE_SIZE = 518
FEATURES_LIST = [6, 12, 18, 24]
N_CTX = 12
VL_REDUCTION = 4
PQ_MID_DIM = 128
PQ_CONTEXT = True
SIGMA = 4
FUSION_TYPE = "average_mean"

# ---- Output ----
OUTPUT_DIR = "./outputs"
HEATMAP_ALPHA = 0.6
MIN_REGION_PCT = 0.002
MASK_PERCENTILE = 99.8
MASK_OVERLAY_ALPHA = 0.45
MASK_DISPLAY_BLUR = 2.0

# ---- flip/orientation-sensitive rerank (no timing) ----
TOPK_RERANK = 50     # CLIP shortlist size
PIX_SIZE = 64        # 48/64/96; higher = stricter about layout/orientation
ALPHA = 0.35         # how much pixel-sim matters vs CLIP (0.2–0.6)
FLIP_PENALTY = 0.20  # penalize if candidate matches better when horizontally flipped


def list_images(folder: str):
    p = Path(folder)
    paths = []
    for ext in EXTS:
        paths += list(p.rglob(f"*{ext}"))
        paths += list(p.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


class ImgDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), str(path)


@torch.inference_mode()
def build_cache(img_paths, cache_paths_file: Path, cache_embeds_file: Path, model, preprocess, device: str):
    ds = ImgDS(img_paths, preprocess)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    embeds = []
    out_paths = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    for xb, pb in dl:
        xb = xb.to(device, non_blocking=True)
        with autocast_ctx:
            feat = model.encode_image(xb)
        feat = F.normalize(feat, dim=-1)
        if device == "cuda":
            feat = feat.half()
        embeds.append(feat)
        out_paths += list(pb)

    embeds = torch.cat(embeds, dim=0).contiguous()

    cache_paths_file.write_text(json.dumps(out_paths), encoding="utf-8")
    torch.save(embeds.cpu(), cache_embeds_file)
    return out_paths, embeds  # embeds on GPU


@torch.inference_mode()
def load_cache(cache_paths_file: Path, cache_embeds_file: Path):
    paths = json.loads(cache_paths_file.read_text(encoding="utf-8"))
    embeds = torch.load(cache_embeds_file, map_location="cpu", weights_only=True)  # no warning
    return paths, embeds


@torch.inference_mode()
def pixel_embed(img: Image.Image, device: str) -> torch.Tensor:
    # orientation/layout-sensitive descriptor: low-res raw pixels
    im = img.resize((PIX_SIZE, PIX_SIZE), Image.BILINEAR).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    v = torch.from_numpy(arr).to(device).reshape(-1)  # (H*W*3,)
    v = v / (v.norm() + 1e-12)
    return v


def load_adaptclip(device: str):
    if PRETRAINED_MODEL == "ViT-L/14@336px":
        model, _ = adaptcliplib.load(PRETRAINED_MODEL, device=device)
        model.visual.DAPM_replace(DPAM_layer=20)
        patch_size = 14
        input_dim = 768
        dpam_layer = 20
    elif PRETRAINED_MODEL == "VITB16_PLUS_240":
        model, _ = adaptcliplib.load(PRETRAINED_MODEL, device=device)
        model.visual.DAPM_replace(DPAM_layer=10)
        patch_size = 16
        input_dim = 640
        dpam_layer = 10
    else:
        raise ValueError(f"Unsupported PRETRAINED_MODEL: {PRETRAINED_MODEL}")

    preprocess, _ = get_transform(image_size=IMAGE_SIZE)

    textual_learner = TextualAdapter(model.to("cpu"), IMAGE_SIZE, N_CTX)
    visual_learner = VisualAdapter(IMAGE_SIZE, patch_size, input_dim=input_dim, reduction=VL_REDUCTION)
    pq_learner = PQAdapter(
        IMAGE_SIZE,
        patch_size,
        context=PQ_CONTEXT,
        input_dim=input_dim,
        mid_dim=PQ_MID_DIM,
        layers_num=len(FEATURES_LIST),
    )

    checkpoint_adapter = torch.load(CHECKPOINT_PATH, map_location="cpu")
    textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
    visual_learner.load_state_dict(checkpoint_adapter["visual_learner"])
    pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])

    model.to(device).eval()
    textual_learner.to(device).eval()
    visual_learner.to(device).eval()
    pq_learner.to(device).eval()

    textual_learner.prepare_static_text_feature(model)
    static_text_features = textual_learner.static_text_features

    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text_learn(learned_prompts, tokenized_prompts).float()

    return (
        model,
        preprocess,
        textual_learner,
        visual_learner,
        pq_learner,
        static_text_features,
        learned_text_features,
        dpam_layer,
    )


@torch.inference_mode()
def run_one_shot(query_path: str, prompt_path: str, device: str):
    (
        model,
        preprocess,
        textual_learner,
        visual_learner,
        pq_learner,
        static_text_features,
        learned_text_features,
        dpam_layer,
    ) = load_adaptclip(device)

    query_img = Image.open(query_path).convert("RGB")
    prompt_img = Image.open(prompt_path).convert("RGB")

    query_tensor = preprocess(query_img).unsqueeze(0).to(device)
    prompt_tensor = preprocess(prompt_img).unsqueeze(0).to(device)

    query_feats, query_patch_feats = model.encode_image(
        query_tensor, FEATURES_LIST, DPAM_layer=dpam_layer
    )
    prompt_feats, prompt_patch_feats = model.encode_image(
        prompt_tensor, FEATURES_LIST, DPAM_layer=dpam_layer
    )

    prompt_feats = prompt_feats.unsqueeze(1)
    prompt_patch_feats = [p.unsqueeze(1) for p in prompt_patch_feats]

    global_vl_logit, local_vl_map = visual_learner(query_feats, query_patch_feats, static_text_features)
    local_vl_map = local_vl_map[:, 1].detach()
    global_vl_score = global_vl_logit.softmax(-1)[:, 1].detach()

    global_tl_logit, local_tl_map = textual_learner.compute_global_local_score(
        query_feats, query_patch_feats, learned_text_features
    )
    local_tl_map = local_tl_map[:, 1].detach()
    global_tl_score = global_tl_logit.softmax(-1)[:, 1].detach()

    global_pq_logit, local_pq_map_list, align_score_list = pq_learner(
        query_feats, query_patch_feats, prompt_feats, prompt_patch_feats
    )

    local_pq_map_list = [x[:, 1].unsqueeze(1) for x in local_pq_map_list]
    local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
    align_score = fusion_fun(align_score_list, fusion_type="harmonic_mean")[:, 0]

    if isinstance(global_pq_logit, list):
        global_pq_score = [x.softmax(-1).unsqueeze(-1) for x in global_pq_logit]
        global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1)[:, 1].detach()
    else:
        global_pq_score = global_pq_logit.softmax(-1)[:, 1].detach()

    pixel_anomaly_map = fusion_fun(
        [local_vl_map, local_tl_map, local_pq_map], fusion_type=FUSION_TYPE
    )
    pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type="harmonic_mean")
    pixel_anomaly_map = torch.stack(
        [torch.from_numpy(gaussian_filter(i, sigma=SIGMA)) for i in pixel_anomaly_map.cpu()],
        dim=0,
    ).to(device)

    pixel_anomaly_map = torch.nan_to_num(
        pixel_anomaly_map, nan=0.0, posinf=0.0, neginf=0.0
    )

    anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(1, -1), dim=1)
    image_anomaly_pred = fusion_fun(
        [global_vl_score, global_tl_score, global_pq_score], fusion_type=FUSION_TYPE
    )
    image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max], fusion_type="harmonic_mean")

    return query_img, pixel_anomaly_map[0], float(image_anomaly_pred.item())


def save_outputs(query_img: Image.Image, anomaly_map: torch.Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_map = normalize(anomaly_map).cpu().numpy()
    heat = (anomaly_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    resized_img = query_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    resized_np = np.asarray(resized_img, dtype=np.uint8)
    heatmap = (HEATMAP_ALPHA * resized_np + (1.0 - HEATMAP_ALPHA) * heat).astype(np.uint8)
    Image.fromarray(heatmap).save(output_dir / "anomaly_heatmap.png")

    thresh = float(np.percentile(anomaly_map, MASK_PERCENTILE))
    mask = (anomaly_map >= thresh).astype(np.uint8) * 255
    mask_rgb = np.zeros_like(resized_np)
    mask_rgb[..., 0] = 255
    mask_alpha = (mask[..., None] / 255.0) * MASK_OVERLAY_ALPHA
    mask_overlay = (
        resized_np * (1.0 - mask_alpha) + mask_rgb * mask_alpha
    ).astype(np.uint8)
    Image.fromarray(mask_overlay).save(output_dir / "segmentation_overlay.png")


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(QUERY_IMAGE).exists():
        raise FileNotFoundError(f"QUERY_IMAGE not found: {QUERY_IMAGE}")
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"CHECKPOINT_PATH not found: {CHECKPOINT_PATH}")

    cache_dir = Path(FOLDER) / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths_file = cache_dir / CACHE_PATHS
    cache_embeds_file = cache_dir / CACHE_EMBEDS

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()

    if cache_paths_file.exists() and cache_embeds_file.exists():
        paths, embeds = load_cache(cache_paths_file, cache_embeds_file)
        embeds = embeds.to(device, non_blocking=True)
    else:
        img_paths = list_images(FOLDER)
        if not img_paths:
            raise FileNotFoundError(f"No images found in FOLDER: {FOLDER}")
        paths, embeds = build_cache(img_paths, cache_paths_file, cache_embeds_file, model, preprocess, device)

    # ---- query CLIP embedding ----
    qimg = Image.open(QUERY_IMAGE).convert("RGB")
    q = preprocess(qimg).unsqueeze(0).to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        qfeat = model.encode_image(q)
    qfeat = F.normalize(qfeat, dim=-1).squeeze(0)
    if device == "cuda":
        qfeat = qfeat.half()

    # ---- CLIP shortlist ----
    clip_scores = embeds @ qfeat  # (N,)
    k = min(TOPK_RERANK, clip_scores.numel())
    topk_idx = torch.topk(clip_scores, k=k).indices.tolist()

    # ---- orientation-sensitive rerank with flip penalty ----
    q_pix = pixel_embed(qimg, device)

    best_i = topk_idx[0]
    best_final = -1e9

    for idx in topk_idx:
        cimg = Image.open(paths[idx]).convert("RGB")

        pix_sim = float((pixel_embed(cimg, device) @ q_pix).item())

        cimg_flip = cimg.transpose(Image.FLIP_LEFT_RIGHT)
        flip_sim = float((pixel_embed(cimg_flip, device) @ q_pix).item())

        penalty = FLIP_PENALTY if flip_sim > pix_sim else 0.0
        final = float(clip_scores[idx].item()) + ALPHA * pix_sim - penalty

        if final > best_final:
            best_final = final
            best_i = idx

    print("Most similar:", paths[best_i])
    print("Cosine similarity:", float(clip_scores[best_i].item()))

    query_img, anomaly_map, image_score = run_one_shot(QUERY_IMAGE, paths[best_i], device)
    output_dir = Path(OUTPUT_DIR)
    save_outputs(query_img, anomaly_map, output_dir)
    print("One-shot image anomaly score:", image_score)
    print("Saved segmentation to:", output_dir)


if __name__ == "__main__":
    main()
