
from pathlib import Path
import sys

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODEL_PATH = ROOT / "artifacts" / "investor_model.pt"

# Allow importing the existing model code
sys.path.insert(0, str(SRC))

from model import InvestorRankingNN, NeuralRanker


# FastAPI application
app = FastAPI(
    title="SponsorMatch API",
    description="AI-powered investor ranking for startups",
    version="1.0.0",
)

# Reuse the loaded model across requests
ranker = None


# Request validation
class StartupRequest(BaseModel):
    sector_group: str = Field(min_length=1)
    stage_bucket: str = Field(min_length=1)
    city_group: str = Field(min_length=1)
    amount_bin: str = Field(min_length=1)
    amount_log10: float | None = None
    company_age: float = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=50)


# Load the saved model and its metadata
def load_ranker():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. "
            "Run python train_for_api.py first."
        )

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


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "SponsorMatch API is running!"
    }


# Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Investor-ranking endpoint
@app.post("/rank")
def rank_investors(startup: StartupRequest):
    global ranker

    try:
        # Load the model only when first needed
        if ranker is None:
            ranker = load_ranker()

        # Convert request into the format expected by the model
        row = startup.model_dump(exclude={"top_k"})
        ranked_investors = ranker.rank(pd.Series(row))

        return {
            "startup": row,
            "ranked_investors": ranked_investors[:startup.top_k],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
