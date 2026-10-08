# CETSS — Clinical Evidence & Triage Support System

Educational medical-image decision support: quality gate → pretrained model → calibration/uncertainty → explainability → triage rules.
**Not a medical device. Not for diagnosis or treatment. Use only public, de-identified images.**

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

The backend serves the frontend, so one process is enough.

## Real model vs demo mode

By default the backend tries to load a TorchXRayVision DenseNet (needs `torch`, `torchvision`, `torchxrayvision` — uncomment them in `requirements.txt`). If unavailable it falls back to a **demo predictor** (an intensity heuristic, not a clinical model); responses then carry `demo_mode: true` and the UI shows a banner. Set `CETSS_FORCE_DEMO=1` to force demo mode.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Model name and demo flag |
| POST | `/api/analyze` | multipart `file` (PNG/JPEG) → finding, calibrated confidence + interval, quality, evidence, heatmap, triage |
| GET | `/api/examples` | Illustrative case walkthrough data |
| GET | `/api/validation` | Illustrative validation metrics (replace with your own) |

## Tuning

- `pipeline/triage.py` — abstention thresholds (`POS_THRESHOLD`, `NEG_THRESHOLD`, `MAX_INTERVAL_WIDTH`).
- `pipeline/model.py` — `CETSS_TEMPERATURE` (fit on your own held-out set), `CETSS_N_AUGMENT` (ensemble passes).
- Validation numbers and example cases in `app.py` are illustrative placeholders.
