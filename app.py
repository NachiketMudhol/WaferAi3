"""
WaferAI v4 — HuggingFace Spaces · Pure API Backend
====================================================
Changes from v3:
  1. HTML frontend REMOVED — served from Render instead
  2. make_wafer_map() REPLACED by make_wafer_map_from_results()
     → Real die-level grid, coloured from actual predictions
  3. detect_and_split_grid() ADDED
     → Canny + HoughLinesP grid detection, splits image into tiles
  4. /predict auto-routes: single image vs grid image
  5. /predict_batch builds real wafer map from all uploaded results
  6. CORS open for Render frontend cross-origin calls

Endpoints:
  GET  /health
  POST /predict        (single or grid image, auto-detected)
  POST /predict_batch  (N images → real wafer map)
"""

import os, io, gc, base64, math
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "best_model.pth"))
NUM_CLASSES = 8
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EXACT class names — must match training order
CLASS_NAMES = ["Bridge", "Clean", "CMP Scratches", "Crack", "LER", "Open", "Other", "Vias"]

# Colour per class for wafer map cells
CLASS_COLORS = {
    "Clean":        "#0d2a3a",   # dark blue  (no defect)
    "Bridge":       "#ff4444",   # red
    "CMP Scratches":"#ffaa00",   # amber
    "Crack":        "#ff6b6b",   # pink-red
    "LER":          "#a78bfa",   # purple
    "Open":         "#00d4ff",   # cyan
    "Other":        "#94a3b8",   # grey
    "Vias":         "#fbbf24",   # yellow
}
CLASS_EDGE_COLORS = {k: ("#ff7777" if v == "#ff4444" else "#0a4060" if v == "#0d2a3a" else v)
                     for k, v in CLASS_COLORS.items()}

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="WaferAI API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # open for Render frontend
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── MODEL (loaded once globally) ──────────────────────────────────────────────
def load_model():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        )
        print(f"✅  Model loaded → {MODEL_PATH}")
    else:
        print(f"⚠️   {MODEL_PATH} not found — demo mode (random weights)")
    m.to(DEVICE).eval()
    return m

model = load_model()

infer_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ── GRADCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, mdl: nn.Module, layer: nn.Module):
        self.mdl   = mdl
        self.grads = None
        self.acts  = None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def run(self, tensor: torch.Tensor):
        self.mdl.zero_grad()
        out = self.mdl(tensor)
        idx = out.argmax(1).item()
        out[0, idx].backward()
        w   = self.grads.mean([0, 2, 3])
        cam = (self.acts[0] * w[:, None, None]).mean(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE)), idx, out

gradcam = GradCAM(model, model.features[-1][0])

# ══════════════════════════════════════════════════════════════════════════════
#  REAL WAFER MAP  — built from actual prediction results
# ══════════════════════════════════════════════════════════════════════════════
def make_wafer_map_from_results(
    results: list[dict],
    grid_cols: int = None,
) -> str:
    """
    Render a real circular wafer map where every cell corresponds
    to one die prediction.

    Parameters
    ----------
    results   : list of dicts, each containing at minimum:
                  {"filename": str, "class": str, "confidence": float}
    grid_cols : force a specific number of columns; if None, auto-computed.

    Returns
    -------
    base64-encoded PNG string.
    """
    n = len(results)
    if n == 0:
        return ""

    # ── Decide grid dimensions ──────────────────────────────────────────────
    if grid_cols is None:
        grid_cols = max(1, math.ceil(math.sqrt(n)))
    grid_rows = math.ceil(n / grid_cols)

    # ── Figure setup ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#050a14")
    ax.set_facecolor("#050a14")
    ax.set_aspect("equal")
    ax.axis("off")

    # Wafer circle has radius = 1.0 in data units.
    # We tile dies across [-1, 1] × [-1, 1] and skip those outside the circle.
    die_w = 2.0 / grid_cols
    die_h = 2.0 / grid_rows
    gap   = 0.008
    wafer_r = 1.0

    # Pre-fill die positions (row-major, left-to-right, top-to-bottom)
    # We map result index → (row, col) in the die grid
    die_positions = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            cx = -1.0 + (col + 0.5) * die_w
            cy =  1.0 - (row + 0.5) * die_h   # y decreases downward
            dist = math.hypot(cx, cy)
            if dist + max(die_w, die_h) * 0.5 * 0.9 > wafer_r:
                die_positions.append(None)          # outside wafer — skip
            else:
                die_positions.append((cx, cy, row, col))

    # Match predictions to *inside-wafer* positions in order
    valid_positions = [p for p in die_positions if p is not None]
    # Pad or truncate so lengths match
    paired = list(zip(valid_positions[:n], results[:n]))

    total   = len(paired)
    defects = 0

    for (cx, cy, row, col), res in paired:
        cls_name = res.get("class", res.get("predicted_class", "Other"))
        is_clean = cls_name == "Clean"
        if not is_clean:
            defects += 1

        x0 = cx - die_w / 2 + gap
        y0 = cy - die_h / 2 + gap
        w  = die_w - 2 * gap
        h  = die_h - 2 * gap

        fc = CLASS_COLORS.get(cls_name, "#ff4444")
        ec = CLASS_EDGE_COLORS.get(cls_name, "#ff7777")

        rect = plt.Rectangle((x0, y0), w, h, lw=0.4,
                              edgecolor=ec, facecolor=fc, alpha=0.93)
        ax.add_patch(rect)

        # Confidence dot in centre of each die (scaled opacity)
        conf = res.get("confidence", 0.0)
        dot_alpha = max(0.3, conf / 100.0)
        dot_r = min(die_w, die_h) * 0.18
        dot_color = "#ffffff" if is_clean else "#ffeeee"
        circ = plt.Circle((cx, cy), dot_r, color=dot_color,
                           alpha=dot_alpha, zorder=3)
        ax.add_patch(circ)

    # ── Wafer boundary + reference rings ────────────────────────────────────
    ax.add_patch(plt.Circle((0, 0), wafer_r,
                             fill=False, ec="#00d4ff", lw=2.5, alpha=0.85))
    ax.plot([-0.12, 0.12], [wafer_r, wafer_r],
            c="#00d4ff", lw=5, solid_capstyle="round", alpha=0.9)   # flat edge

    for ring_r in [0.25, 0.5, 0.75, 0.9]:
        ax.add_patch(plt.Circle((0, 0), ring_r,
                                 fill=False, ec="#00d4ff", lw=0.5, alpha=0.14))

    # Crosshair
    ax.plot([-0.05, 0.05], [0, 0], c="#00d4ff", lw=0.8, alpha=0.5)
    ax.plot([0, 0], [-0.05, 0.05], c="#00d4ff", lw=0.8, alpha=0.5)

    # ── Stats annotation ─────────────────────────────────────────────────────
    yld = (total - defects) / total * 100 if total else 0
    stats_txt = (f"Dies: {total}   Defective: {defects}   "
                 f"Yield: {yld:.1f}%   Grid: {grid_rows}×{grid_cols}")
    ax.text(0, -1.15, stats_txt,
            ha="center", va="center", color="#00d4ff",
            fontsize=7.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=.35", fc="#0a1628",
                      ec="#00d4ff", alpha=0.88))

    # ── Legend ───────────────────────────────────────────────────────────────
    seen_classes = list({res.get("class", res.get("predicted_class", "Other"))
                         for _, res in paired})
    legend_patches = []
    for cls in sorted(seen_classes):
        color = CLASS_COLORS.get(cls, "#aaaaaa")
        legend_patches.append(
            mpatches.Patch(facecolor=color, edgecolor="#444",
                           linewidth=0.5, label=cls)
        )
    if legend_patches:
        ax.legend(handles=legend_patches,
                  loc="lower right", bbox_to_anchor=(1.18, 0.0),
                  fontsize=6.5, framealpha=0.2,
                  labelcolor="white", facecolor="#050a14",
                  edgecolor="#00d4ff")

    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.30, 1.20)
    ax.set_title("Wafer Map — Real Die Predictions",
                 color="#00d4ff", fontsize=11, fontfamily="monospace", pad=8)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor="#050a14")
    plt.close(fig)
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    buf.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GRID DETECTION — splits a contact-sheet image into individual die tiles
# ══════════════════════════════════════════════════════════════════════════════
def detect_and_split_grid(pil_img: Image.Image) -> tuple[list[Image.Image], int, int]:
    """
    Detect whether `pil_img` is a grid of multiple SEM dies.
    If it is, split it into tiles.

    Algorithm
    ---------
    1. Convert to grayscale numpy array.
    2. Apply Canny edge detection.
    3. Use Hough line detection to find strong horizontal and vertical lines.
    4. Cluster line positions to find row / column boundaries.
    5. If ≥2 rows AND ≥2 cols found → split and return tiles.
    6. Otherwise → return [original image] (single die).

    Returns
    -------
    (tiles, n_rows, n_cols)
      tiles   : list[PIL.Image]  — individual die images
      n_rows  : int
      n_cols  : int
    """
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    H, W   = gray.shape

    # ── Edge map ─────────────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # ── Hough line detection ─────────────────────────────────────────────────
    # rho=1 px, theta=1°, threshold = 40% of the shorter dimension
    min_line_len = int(min(H, W) * 0.40)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(min(H, W) * 0.30),
        minLineLength=min_line_len,
        maxLineGap=int(min(H, W) * 0.05),
    )

    h_lines: list[int] = []   # y-coordinates of horizontal lines
    v_lines: list[int] = []   # x-coordinates of vertical lines

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            angle = math.degrees(math.atan2(dy, dx + 1e-6))

            if angle < 10:                # nearly horizontal
                y_mid = (y1 + y2) // 2
                # Ignore lines too close to image border (< 3%)
                if H * 0.03 < y_mid < H * 0.97:
                    h_lines.append(y_mid)
            elif angle > 80:              # nearly vertical
                x_mid = (x1 + x2) // 2
                if W * 0.03 < x_mid < W * 0.97:
                    v_lines.append(x_mid)

    # ── Fallback: gradient-based line detection ───────────────────────────────
    # Use integrated column / row sums of gradient if Hough found nothing
    if len(h_lines) < 1 or len(v_lines) < 1:
        gy = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
        gx = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))

        row_energy = gy.mean(axis=1)
        col_energy = gx.mean(axis=0)

        def peaks_from_energy(energy, length, min_gap_frac=0.06):
            energy = energy / (energy.max() + 1e-9)
            threshold = np.percentile(energy, 85)
            above = energy > threshold
            peaks = []
            in_band, band_start = False, 0
            min_gap = int(length * min_gap_frac)
            for i, v in enumerate(above):
                if v and not in_band:
                    in_band, band_start = True, i
                elif not v and in_band:
                    in_band = False
                    c = (band_start + i) // 2
                    if length * 0.03 < c < length * 0.97:
                        if not peaks or c - peaks[-1] > min_gap:
                            peaks.append(c)
            return peaks

        if len(h_lines) < 1:
            h_lines = peaks_from_energy(row_energy, H)
        if len(v_lines) < 1:
            v_lines = peaks_from_energy(col_energy, W)

    # ── Cluster nearby lines ─────────────────────────────────────────────────
    def cluster_lines(positions: list[int], gap: int = 15) -> list[int]:
        if not positions:
            return []
        positions = sorted(set(positions))
        clusters, group = [], [positions[0]]
        for p in positions[1:]:
            if p - group[-1] <= gap:
                group.append(p)
            else:
                clusters.append(int(np.mean(group)))
                group = [p]
        clusters.append(int(np.mean(group)))
        return clusters

    cluster_gap = max(10, int(min(H, W) * 0.04))
    h_clustered = cluster_lines(h_lines, gap=cluster_gap)
    v_clustered = cluster_lines(v_lines, gap=cluster_gap)

    n_rows_detected = len(h_clustered) + 1
    n_cols_detected = len(v_clustered) + 1

    # ── Single-image threshold ───────────────────────────────────────────────
    # Need at least 2×2 to be treated as a grid
    if n_rows_detected < 2 or n_cols_detected < 2:
        return [pil_img], 1, 1

    # ── Compute tile boundaries ───────────────────────────────────────────────
    row_boundaries = [0] + h_clustered + [H]
    col_boundaries = [0] + v_clustered + [W]

    # Filter out very thin strips (< 5% of dimension) created by edge-case lines
    min_tile_h = int(H * 0.05)
    min_tile_w = int(W * 0.05)

    row_boundaries = [b for i, b in enumerate(row_boundaries)
                      if i == 0 or b - row_boundaries[i-1] >= min_tile_h
                      or i == len(row_boundaries)-1]
    col_boundaries = [b for i, b in enumerate(col_boundaries)
                      if i == 0 or b - col_boundaries[i-1] >= min_tile_w
                      or i == len(col_boundaries)-1]

    n_rows = len(row_boundaries) - 1
    n_cols = len(col_boundaries) - 1

    # ── Extract tiles ─────────────────────────────────────────────────────────
    # Add a small inward margin (2%) to avoid grid-line bleed
    tiles: list[Image.Image] = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = row_boundaries[r]
            y1 = row_boundaries[r + 1]
            x0 = col_boundaries[c]
            x1 = col_boundaries[c + 1]
            tw, th = x1 - x0, y1 - y0
            mx, my = int(tw * 0.02), int(th * 0.02)
            tile_pil = pil_img.crop((x0 + mx, y0 + my, x1 - mx, y1 - my))
            tiles.append(tile_pil)

    return tiles, n_rows, n_cols


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-IMAGE INFERENCE  (with GradCAM)
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(file_bytes: bytes, filename: str) -> dict:
    """
    Run model on one image.
    Returns the full result dict consumed by the frontend.
    """
    pil    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pil    = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    tensor = infer_tf(pil).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    cam, idx, out = gradcam.run(tensor)

    probs  = torch.softmax(out, 1)[0].detach().cpu().numpy()
    scores = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
              for i in range(NUM_CLASSES)}
    pred   = CLASS_NAMES[idx]
    conf   = round(float(probs[idx]) * 100, 2)

    # Original → base64
    orig_np = np.array(pil)
    _, obuf  = cv2.imencode(".png", cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
    orig_b64 = base64.b64encode(obuf).decode()

    # GradCAM overlay → base64
    gray    = cv2.cvtColor(cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY),
                           cv2.COLOR_GRAY2RGB)
    hmap    = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hmap    = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(gray, 0.45, hmap, 0.55, 0)
    _, gbuf  = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    grad_b64 = base64.b64encode(gbuf).decode()

    del tensor, out, probs, cam, gray, hmap, overlay
    gc.collect()

    return dict(
        success         = True,
        mode            = "single",
        predicted_class = pred,
        confidence      = conf,
        scores          = scores,
        original_image  = orig_b64,
        gradcam_image   = grad_b64,
        metadata        = dict(filename=filename, device=str(DEVICE)),
    )


def run_inference_on_tile(tile_pil: Image.Image, tile_label: str) -> dict:
    """
    Lightweight inference on a PIL tile — no GradCAM (saves memory
    when processing many tiles from one grid image).
    Returns a slim result dict suitable for wafer-map construction.
    """
    tile_pil = tile_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    tensor   = infer_tf(tile_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out    = model(tensor)
        probs  = torch.softmax(out, 1)[0].cpu().numpy()

    idx    = int(probs.argmax())
    pred   = CLASS_NAMES[idx]
    conf   = round(float(probs[idx]) * 100, 2)
    scores = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
              for i in range(NUM_CLASSES)}

    # Encode tile thumbnail for the grid card view in the UI
    tile_np  = np.array(tile_pil)
    _, tbuf  = cv2.imencode(".jpg", cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    tile_b64 = base64.b64encode(tbuf).decode()

    del tensor, out, probs
    gc.collect()

    return dict(
        filename   = tile_label,
        cls        = pred,
        confidence = conf,
        scores     = scores,
        thumbnail  = tile_b64,     # small JPEG for UI grid
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        "MobileNetV3-Small",
        "classes":      NUM_CLASSES,
        "class_names":  CLASS_NAMES,
        "device":       str(DEVICE),
        "model_loaded": os.path.exists(MODEL_PATH),
        "version":      "4.0",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Smart single-image endpoint.

    AUTO-DETECTS whether the upload is:
      (a) A single SEM die image  → full GradCAM inference, no wafer map
      (b) A grid / contact sheet  → split tiles, classify each, build wafer map

    Response schema
    ---------------
    mode == "single":
        { success, mode, predicted_class, confidence, scores,
          original_image, gradcam_image, metadata }

    mode == "grid":
        { success, mode, num_tiles, grid_rows, grid_cols,
          tile_results: [{filename, cls, confidence, scores, thumbnail}],
          wafer_map,
          metadata }
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image field")
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        # ── Grid detection ───────────────────────────────────────────────────
        tiles, n_rows, n_cols = detect_and_split_grid(pil)
        is_grid = (n_rows >= 2 and n_cols >= 2)

        if not is_grid:
            # ── Path A: single die ───────────────────────────────────────────
            data = run_inference(raw, image.filename or "upload.png")
            return JSONResponse(content=data)

        # ── Path B: grid image ───────────────────────────────────────────────
        tile_results = []
        for ti, tile in enumerate(tiles):
            label = f"{image.filename or 'grid'}_tile_{ti+1:03d}"
            res   = run_inference_on_tile(tile, label)
            tile_results.append(res)

        # Build wafer-map-friendly list
        wm_input = [{"class": r["cls"], "confidence": r["confidence"],
                     "filename": r["filename"]}
                    for r in tile_results]

        wafer_b64 = make_wafer_map_from_results(
            wm_input, grid_cols=n_cols
        )

        return JSONResponse(content=dict(
            success     = True,
            mode        = "grid",
            num_tiles   = len(tile_results),
            grid_rows   = n_rows,
            grid_cols   = n_cols,
            tile_results= tile_results,
            wafer_map   = wafer_b64,
            metadata    = dict(
                filename = image.filename or "upload.png",
                device   = str(DEVICE),
            ),
        ))

    except Exception as e:
        import traceback; traceback.print_exc()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(images: list[UploadFile] = File(...)):
    """
    Batch endpoint — accepts N individual die images.
    Runs inference on each, then builds a REAL wafer map from all results.

    Response schema
    ---------------
    {
      results: [{ filename, class, confidence, scores,
                  original_image, gradcam_image }],
      wafer_map: base64,
      total: int,
      defective: int,
      yield_pct: float
    }
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")

    results = []
    for f in images:
        try:
            raw = await f.read()
            # Full inference (with GradCAM) per image for batch results panel
            r = run_inference(raw, f.filename or "upload.png")
            results.append({
                "filename":       r["metadata"]["filename"],
                "class":          r["predicted_class"],
                "confidence":     r["confidence"],
                "scores":         r["scores"],
                "original_image": r["original_image"],
                "gradcam_image":  r["gradcam_image"],
                "success":        True,
            })
        except Exception as e:
            results.append({
                "filename":  f.filename or "upload.png",
                "success":   False,
                "error":     str(e),
            })
        finally:
            gc.collect()

    # Build real wafer map from all successfully classified dies
    ok_results = [r for r in results if r.get("success")]
    wm_input   = [{"class": r["class"], "confidence": r["confidence"],
                   "filename": r["filename"]}
                  for r in ok_results]
    wafer_b64  = make_wafer_map_from_results(wm_input) if wm_input else ""

    defective  = sum(1 for r in ok_results if r["class"] != "Clean")
    total_ok   = len(ok_results)
    yield_pct  = round((total_ok - defective) / total_ok * 100, 1) if total_ok else 0.0

    return JSONResponse(content=dict(
        results   = results,
        wafer_map = wafer_b64,
        total     = len(results),
        defective = defective,
        yield_pct = yield_pct,
    ))


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*58}")
    print(f"  WaferAI v4 — HuggingFace Spaces API Backend")
    print(f"{'='*58}")
    print(f"  Listening : http://0.0.0.0:{port}")
    print(f"  Device    : {DEVICE}")
    print(f"  Model     : {'✅ found' if os.path.exists(MODEL_PATH) else '⚠️  NOT FOUND (demo)'}")
    print(f"  Endpoints : /health  /predict  /predict_batch")
    print(f"{'='*58}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
