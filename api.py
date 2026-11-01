
from pathlib import Path
import sys

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODEL_PATH = ROOT / "artifacts" / "investor_model.pt"

sys.path.insert(0, str(SRC))

from model import InvestorRankingNN, NeuralRanker

app = FastAPI(
    title="SponsorMatch API",
    description="Rank suitable investors for startups",
    version="1.0.0",
)


class StartupRequest(BaseModel):
    sector_group: str
    stage_bucket: str
    city_group: str
    amount_bin: str
    amount_log10: float | None = None
    company_age: float = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=50)


def load_ranker():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. Run python train_for_api.py first."
        )

    # Only load model files you created or trust.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    feature_maps = checkpoint["feature_maps"]
    numeric_stats = checkpoint["numeric_stats"]
    investors = checkpoint["investors"]

    model = InvestorRankingNN(
        feature_maps,
        len(investors),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return NeuralRanker(
        model=model,
        feature_maps=feature_maps,
        numeric_stats=numeric_stats,
        investors=investors,
        device=torch.device("cpu"),
    )


@app.get("/")
def home():
    return {"message": "SponsorMatch API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/rank")
def rank_investors(startup: StartupRequest):
    try:
        ranker = load_ranker()
        row = startup.model_dump(exclude={"top_k"})

        ranked = ranker.rank(pd.Series(row))

        return {
            "startup": row,
            "ranked_investors": ranked[:startup.top_k],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
