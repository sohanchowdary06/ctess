"""Stages 2-4 - pretrained model, calibration + uncertainty, explainability.

Two backends share one interface:

* TorchXRayVisionPredictor - real pretrained DenseNet (needs torch, torchvision,
  torchxrayvision). Produces a Pneumonia logit and a Grad-CAM map.
* DemoPredictor - dependency-free stand-in so the UI and API can be exercised
  end to end. It is NOT a medical model; responses are flagged demo_mode=true.
"""
from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Temperature fitted on a held-out set (replace with your own value).
TEMPERATURE = float(os.getenv("CETSS_TEMPERATURE", "1.4"))
N_AUGMENT = int(os.getenv("CETSS_N_AUGMENT", "12"))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RawPrediction:
    logits: list[float]          # one logit per augmented pass
    heatmap: np.ndarray          # HxW in [0, 1]
    finding_region: str


def _augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    dx, dy = int(rng.integers(-w // 40, w // 40 + 1)), int(rng.integers(-h // 40, h // 40 + 1))
    shifted = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
    arr = np.asarray(shifted, dtype=np.float32) * rng.uniform(0.92, 1.08)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _region_name(heat: np.ndarray) -> str:
    h, w = heat.shape
    cy, cx = np.unravel_index(int(np.argmax(heat)), heat.shape)
    vert = "upper" if cy < h / 3 else "mid" if cy < 2 * h / 3 else "lower"
    side = "right" if cx < w / 2 else "left"  # radiograph convention: patient's right is image left
    return f"{side} {vert} zone"


# --------------------------------------------------------------------------- demo
class DemoPredictor:
    name = "demo-intensity-heuristic"
    demo = True

    def _score(self, gray: np.ndarray) -> tuple[float, np.ndarray]:
        h, w = gray.shape
        g = gray / 255.0
        lung = g[int(h * 0.15): int(h * 0.9), int(w * 0.12): int(w * 0.88)]
        baseline = float(np.median(lung))
        opac = np.clip((g - baseline) * 3.0, 0, 1)
        # Emphasise the lower lung fields where consolidation is commonly reported.
        weights = np.linspace(0.6, 1.4, h)[:, None]
        heat = opac * weights
        mask = np.zeros_like(heat)
        mask[int(h * 0.15): int(h * 0.9), int(w * 0.12): int(w * 0.88)] = 1
        heat *= mask
        score = float(heat[mask > 0].mean())
        return (score - 0.18) * 14.0, heat

    def predict(self, img: Image.Image) -> RawPrediction:
        rng = np.random.default_rng(7)
        logits, heat_acc = [], None
        for i in range(N_AUGMENT):
            view = img if i == 0 else _augment(img, rng)
            gray = np.asarray(view.convert("L").resize((224, 224)), dtype=np.float32)
            logit, heat = self._score(gray)
            logits.append(logit)
            heat_acc = heat if heat_acc is None else heat_acc + heat
        heat = heat_acc / max(heat_acc.max(), 1e-6)
        return RawPrediction(logits, heat, _region_name(heat))


# ------------------------------------------------------------------- real model
class TorchXRayVisionPredictor:
    name = "torchxrayvision densenet121-res224-all"
    demo = False

    def __init__(self):
        import torch
        import torchvision
        import torchxrayvision as xrv

        self.torch, self.xrv = torch, xrv
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
        self.idx = self.model.pathologies.index("Pneumonia")
        self.tf = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    def _prep(self, img: Image.Image):
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        arr = self.xrv.datasets.normalize(arr, 255)[None, ...]
        arr = self.tf(arr)
        return self.torch.from_numpy(arr).unsqueeze(0)

    def _gradcam(self, x):
        torch = self.torch
        feats, grads = {}, {}
        layer = self.model.features
        h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        x = x.clone().requires_grad_(True)
        out = self.model(x)[0, self.idx]
        self.model.zero_grad()
        out.backward()
        h1.remove()
        h2.remove()
        a, g = feats["a"][0], grads["g"][0]
        cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0)).detach().numpy()
        cam = np.asarray(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
        return cam / max(cam.max(), 1e-6), float(out.item())

    def predict(self, img: Image.Image) -> RawPrediction:
        rng = np.random.default_rng(7)
        logits, cam0 = [], None
        for i in range(N_AUGMENT):
            view = img if i == 0 else _augment(img, rng)
            x = self._prep(view)
            if i == 0:
                cam0, logit = self._gradcam(x)
            else:
                with self.torch.no_grad():
                    logit = float(self.model(x)[0, self.idx])
            logits.append(logit)
        return RawPrediction(logits, cam0, _region_name(cam0))


def load_predictor():
    if os.getenv("CETSS_FORCE_DEMO") != "1":
        try:
            return TorchXRayVisionPredictor()
        except Exception as exc:  # missing deps / weights unavailable
            print(f"[cetss] real model unavailable ({exc!r}); using DEMO predictor")
    return DemoPredictor()


# --------------------------------------------------------- calibration & output
def calibrate(logits: list[float]) -> tuple[float, tuple[float, float]]:
    """Temperature scaling per pass, then mean + 90% interval across passes
    (test-time-augmentation ensemble as the uncertainty estimate)."""
    probs = np.array([_sigmoid(l / TEMPERATURE) for l in logits])
    lo, hi = np.percentile(probs, [5, 95])
    return float(probs.mean()), (float(lo), float(hi))


def heatmap_overlay(img: Image.Image, heat: np.ndarray, size: int = 384) -> str:
    base = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32)
    h = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
    rgb = np.stack([base, base, base], axis=-1)
    red = np.zeros_like(rgb)
    red[..., 0], red[..., 1] = 255, 90 * (1 - h)
    a = (h ** 1.5 * 0.65)[..., None]
    out = np.clip(rgb * (1 - a) + red * a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
