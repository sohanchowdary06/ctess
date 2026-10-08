"""Stage 5 - triage rules with abstention thresholds.

Outcomes: ESCALATE, ROUTINE, DEFER, REJECT. These are workflow suggestions for
a qualified reviewer, not diagnoses or treatment recommendations."""
from __future__ import annotations

POS_THRESHOLD = 0.70
NEG_THRESHOLD = 0.30
MAX_INTERVAL_WIDTH = 0.30
MIN_QUALITY = 0.45
MIN_OOD = 0.45


def decide(p: float, interval: tuple[float, float], quality: dict) -> dict:
    width = interval[1] - interval[0]

    if quality["status"] == "FAIL" or quality["ood_score"] < MIN_OOD:
        return {
            "action": "REJECT",
            "label": "REJECT - image failed quality / out-of-distribution check; no prediction made",
            "abstained": True,
            "reason": "; ".join(quality["issues"]) or "Quality gate failed",
        }

    if width > MAX_INTERVAL_WIDTH or NEG_THRESHOLD < p < POS_THRESHOLD:
        return {
            "action": "DEFER",
            "label": "DEFER - model is uncertain; refer to a qualified clinician / repeat imaging",
            "abstained": True,
            "reason": f"Probability {p:.2f} with interval width {width:.2f} is inside the abstention region",
        }

    if p >= POS_THRESHOLD:
        return {
            "action": "ESCALATE",
            "label": "ESCALATE - priority review by qualified clinician",
            "abstained": False,
            "reason": f"Probability {p:.2f} >= {POS_THRESHOLD}",
        }

    return {
        "action": "ROUTINE",
        "label": "ROUTINE - standard review queue (still requires clinician sign-off)",
        "abstained": False,
        "reason": f"Probability {p:.2f} <= {NEG_THRESHOLD}",
    }
