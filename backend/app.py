"""CETSS - Clinical Evidence & Triage Support System (educational use only).

Run:  uvicorn app:app --reload --port 8000   (from the backend/ directory)
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from pipeline import quality, triage
from pipeline.model import TEMPERATURE, calibrate, heatmap_overlay, load_predictor

MAX_BYTES = 12 * 1024 * 1024
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

DISCLAIMER = (
    "For education and research only. Not a medical device. Does not diagnose or "
    "recommend treatment. Every output requires review by a qualified professional."
)

app = FastAPI(title="CETSS API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = load_predictor()


@app.get("/api/health")
def health():
    return {"status": "ok", "model": predictor.name, "demo_mode": predictor.demo}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 12 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Unsupported file. Upload a PNG or JPEG image (DICOM is not supported).")

    img = img.convert("RGB")
    q = quality.assess(img)

    # Quality gate: reject before any prediction is made.
    if q["status"] == "FAIL":
        return {
            "finding": None,
            "label": "No prediction",
            "confidence": None,
            "quality": q,
            "evidence": [f"Quality gate: {i}" for i in q["issues"]],
            "triage": triage.decide(0.5, (0.0, 1.0), q),
            "heatmap": None,
            "model": predictor.name,
            "demo_mode": predictor.demo,
            "disclaimer": DISCLAIMER,
        }

    raw = predictor.predict(img)
    p, (lo, hi) = calibrate(raw.logits)
    t = triage.decide(p, (lo, hi), q)

    if p >= triage.POS_THRESHOLD:
        label = "Positive"
        finding = f"Opacity pattern suggestive of pneumonia, maximal in the {raw.finding_region}"
    elif p <= triage.NEG_THRESHOLD:
        label = "Negative"
        finding = "No pneumonia-suggestive opacity detected"
    else:
        label = "Uncertain"
        finding = "Indeterminate - model confidence is inside the abstention band"

    evidence = [
        f"Attention map peaks in the {raw.finding_region}",
        f"{len(raw.logits)}-pass augmentation ensemble: 90% interval [{lo:.2f}, {hi:.2f}] (width {hi - lo:.2f})",
        f"Temperature-scaled probability (T={TEMPERATURE})",
    ]
    if q["status"] == "WARN":
        evidence.append("Quality warning: " + "; ".join(q["issues"]))
    if predictor.demo:
        evidence.append("DEMO MODE: intensity heuristic, not a trained clinical model")

    return {
        "finding": finding,
        "label": label,
        "confidence": {"p": round(p, 3), "interval": [round(lo, 3), round(hi, 3)], "level": 0.90},
        "quality": q,
        "evidence": evidence,
        "triage": t,
        "heatmap": heatmap_overlay(img, raw.heatmap),
        "model": predictor.name,
        "demo_mode": predictor.demo,
        "disclaimer": DISCLAIMER,
    }


# Illustrative content for the walkthrough and validation sections.
EXAMPLES = [
    {
        "id": "positive", "title": "Positive",
        "finding": "Right lower-lobe opacity consistent with consolidation",
        "p": 0.91, "interval": [0.86, 0.95], "quality": 0.94, "quality_status": "PASS",
        "evidence": [
            "Grad-CAM attention concentrated in right lower zone",
            "Similar to 4/5 nearest reference cases labelled pneumonia",
            "Silhouette sign near right hemidiaphragm",
        ],
        "triage": "ESCALATE - priority review by qualified clinician",
    },
    {
        "id": "negative", "title": "Negative",
        "finding": "No focal consolidation, effusion or pneumothorax",
        "p": 0.06, "interval": [0.03, 0.10], "quality": 0.92, "quality_status": "PASS",
        "evidence": [
            "Attention diffuse with no focal peak",
            "Similar to 5/5 nearest reference cases labelled normal",
            "Costophrenic angles sharp bilaterally",
        ],
        "triage": "ROUTINE - standard review queue",
    },
    {
        "id": "uncertain", "title": "Uncertain",
        "finding": "Subtle left perihilar haze; indeterminate",
        "p": 0.52, "interval": [0.31, 0.72], "quality": 0.81, "quality_status": "PASS",
        "evidence": [
            "Wide uncertainty interval (0.41) across ensemble passes",
            "Nearest reference cases split 3 pneumonia / 2 normal",
            "Attention weakly localized",
        ],
        "triage": "DEFER - abstain; refer to senior clinician",
    },
    {
        "id": "poor", "title": "Poor quality / OOD",
        "finding": "No prediction - image rejected by quality gate",
        "p": None, "interval": None, "quality": 0.27, "quality_status": "FAIL",
        "evidence": [
            "Severe motion blur and under-exposure",
            "OOD score 0.18 - view does not match PA/AP chest radiograph",
        ],
        "triage": "REJECT - request repeat acquisition",
    },
]

VALIDATION = {
    "note": "Illustrative held-out test-set results - replace with your own numbers.",
    "metrics": [
        {"key": "AUROC", "value": "0.912"}, {"key": "Sensitivity", "value": "0.89"},
        {"key": "Specificity", "value": "0.84"}, {"key": "ECE (calibrated)", "value": "0.031"},
        {"key": "Brier score", "value": "0.087"}, {"key": "Abstention rate", "value": "11%"},
        {"key": "OOD AUROC", "value": "0.95"}, {"key": "Pointing game", "value": "0.78"},
    ],
}


@app.get("/api/examples")
def examples():
    return EXAMPLES


@app.get("/api/validation")
def validation():
    return VALIDATION


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="static")
