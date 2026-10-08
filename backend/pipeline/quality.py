"""Stage 1 - quality gate: blur, exposure, contrast, colourfulness and a
simple out-of-distribution (OOD) heuristic, evaluated before any prediction."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def assess(img: Image.Image) -> dict:
    rgb = np.asarray(img.convert("RGB").resize((512, 512)), dtype=np.float32)
    gray = rgb.mean(axis=2)

    # Sharpness (0..1) from Laplacian variance, saturating at ~300.
    sharp = min(_laplacian_variance(gray) / 300.0, 1.0)

    # Exposure: penalise mean far from mid-grey and heavy clipping.
    mean = gray.mean() / 255.0
    clipped = float(((gray < 5) | (gray > 250)).mean())
    exposure = max(0.0, 1.0 - abs(mean - 0.45) * 2.0) * (1.0 - min(clipped * 2, 1.0))

    # Contrast from the spread of intensities.
    contrast = min(float(gray.std()) / 60.0, 1.0)

    # Radiographs are essentially greyscale: colour channels should agree.
    colourfulness = float(np.abs(rgb[..., 0] - rgb[..., 2]).mean()) / 255.0
    greyscale_ok = 1.0 - min(colourfulness * 6.0, 1.0)

    # OOD heuristic: greyscale, usable contrast, not a flat or near-binary image.
    hist, _ = np.histogram(gray, bins=32, range=(0, 255))
    occupancy = float((hist > gray.size * 0.002).mean())
    ood = float(np.clip(0.45 * greyscale_ok + 0.30 * occupancy + 0.25 * contrast, 0, 1))

    score = float(np.clip(0.35 * sharp + 0.30 * exposure + 0.20 * contrast + 0.15 * greyscale_ok, 0, 1))

    issues = []
    if sharp < 0.25:
        issues.append("Image appears blurred")
    if exposure < 0.35:
        issues.append("Poor exposure (too dark, too bright or clipped)")
    if contrast < 0.25:
        issues.append("Low contrast")
    if greyscale_ok < 0.6:
        issues.append("Image is strongly coloured; unlikely to be a radiograph")
    if ood < 0.5:
        issues.append("Image looks out-of-distribution for chest radiographs")

    status = "FAIL" if (score < 0.45 or ood < 0.45) else ("WARN" if score < 0.65 else "PASS")
    return {
        "score": round(score, 3),
        "status": status,
        "ood_score": round(ood, 3),
        "metrics": {
            "sharpness": round(sharp, 3),
            "exposure": round(exposure, 3),
            "contrast": round(contrast, 3),
            "greyscale": round(greyscale_ok, 3),
        },
        "issues": issues,
    }
