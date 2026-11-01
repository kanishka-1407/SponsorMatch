from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from pitch_requests import router as pitch_router

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "artifacts" / "investor_model.pt"
INVESTOR_PROFILES_PATH = DATA_DIR / "investor_profiles.csv"
STARTUPS_POOL_PATH = DATA_DIR / "startups_pool.csv"
sys.path.insert(0, str(SRC))
from model import InvestorRankingNN, NeuralRanker  # noqa: E402

SECTORS = [
    "Consumer & Retail", "Deep Tech & Hardware", "Edtech",
    "Energy & Climate", "Enterprise & SaaS", "Fintech",
    "Food & Agri", "Healthtech", "Industrial & Manufacturing",
    "Media & Gaming", "Mobility & Logistics", "Other",
    "Real Estate & Construction", "Travel & Hospitality",
]
STAGES = ["Seed", "Series A", "Series B", "Growth", "Debt", "Other", "Undisclosed"]
CITIES = [
    "Ahmedabad", "Bengaluru", "Chennai", "Delhi NCR", "Hyderabad",
    "International", "Kolkata", "Mumbai", "Other India", "Pune", "Unknown",
]
AMOUNT_BINS = [
    "<$100K", "$100K-1M", "$1M-5M", "$5M-20M",
    "$20M-100M", "$100M+", "Undisclosed",
]

Sector = Literal[
    "Consumer & Retail", "Deep Tech & Hardware", "Edtech",
    "Energy & Climate", "Enterprise & SaaS", "Fintech",
    "Food & Agri", "Healthtech", "Industrial & Manufacturing",
    "Media & Gaming", "Mobility & Logistics", "Other",
    "Real Estate & Construction", "Travel & Hospitality",
]
Stage = Literal["Seed", "Series A", "Series B", "Growth", "Debt", "Other", "Undisclosed"]
City = Literal[
    "Ahmedabad", "Bengaluru", "Chennai", "Delhi NCR", "Hyderabad",
    "International", "Kolkata", "Mumbai", "Other India", "Pune", "Unknown",
]
AmountBin = Literal[
    "<$100K", "$100K-1M", "$1M-5M", "$5M-20M",
    "$20M-100M", "$100M+", "Undisclosed",
]


def load_ranker():
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Run python train_for_api.py first.")
    # The checkpoint should be generated locally by this project's trusted trainer.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    feature_maps = checkpoint["feature_maps"]
    numeric_stats = checkpoint["numeric_stats"]
    investors = checkpoint["investors"]
    model = InvestorRankingNN(feature_maps, len(investors))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return NeuralRanker(
        model=model,
        feature_maps=feature_maps,
        numeric_stats=numeric_stats,
        investors=investors,
        device=torch.device("cpu"),
    )


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep the API available for /health and /options even if the model is missing.
    try:
        app.state.ranker = load_ranker()
        app.state.model_load_error = None
    except Exception:
        app.state.ranker = None
        app.state.model_load_error = "model unavailable"
    yield


app = FastAPI(
    title="SponsorMatch API",
    description="AI-powered investor ranking and previously-funded startup matching",
    version="1.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pitch_router)
# Fallback supports test clients that do not run application lifespan.
ranker = None


class StartupRequest(BaseModel):
    sector_group: Sector
    stage_bucket: Stage
    city_group: City
    amount_bin: AmountBin
    amount_log10: float | None = None
    company_age: float = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=50)


class StartupMatchRequest(BaseModel):
    preferred_sectors: list[Sector] = Field(default_factory=list, max_length=14)
    preferred_stages: list[Stage] = Field(default_factory=list, max_length=7)
    min_ticket_usd: float | None = Field(default=None, ge=0)
    max_ticket_usd: float | None = Field(default=None, ge=0)
    preferred_cities: list[City] = Field(default_factory=list, max_length=11)
    top_k: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def check_ticket_range(self):
        if (
            self.min_ticket_usd is not None
            and self.max_ticket_usd is not None
            and self.min_ticket_usd > self.max_ticket_usd
        ):
            raise ValueError("min_ticket_usd must be less than or equal to max_ticket_usd")
        return self


def get_ranker(request: StartupRequest):
    global ranker
    active_ranker = getattr(app.state, "ranker", None) or ranker
    if active_ranker is None:
        try:
            active_ranker = load_ranker()
            ranker = active_ranker
            app.state.ranker = active_ranker
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Ranking model unavailable. Check the trained model checkpoint.",
            )
    return active_ranker


def json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def profile_details(investor_name: str, profiles: pd.DataFrame) -> dict:
    if profiles.empty or "investor" not in profiles.columns:
        return {
            "n_deals": None,
            "ticket_median": None,
            "top_sectors": None,
            "top_stages": None,
            "top_cities": None,
        }
    matched = profiles.loc[profiles["investor"].astype(str) == str(investor_name)]
    if matched.empty:
        return {
            "n_deals": None,
            "ticket_median": None,
            "top_sectors": None,
            "top_stages": None,
            "top_cities": None,
        }
    record = matched.iloc[0]
    return {
        "n_deals": json_value(record.get("n_deals")),
        "ticket_median": json_value(record.get("ticket_median")),
        "top_sectors": json_value(record.get("top_sectors")),
        "top_stages": json_value(record.get("top_stages")),
        "top_cities": json_value(record.get("top_cities")),
    }


def normalize_stage(value) -> str:
    if pd.isna(value):
        return "Undisclosed"
    stage = str(value).strip().lower().replace("-", " ").replace("_", " ")
    if "seed" in stage:
        return "Seed"
    if "series a" in stage or stage == "a":
        return "Series A"
    if "series b" in stage or stage == "b":
        return "Series B"
    if any(word in stage for word in ("growth", "late stage", "late-stage")):
        return "Growth"
    if "debt" in stage:
        return "Debt"
    if stage in ("", "nan", "none", "undisclosed", "unknown"):
        return "Undisclosed"
    return "Other"


def choose_ticket(row: pd.Series):
    latest = pd.to_numeric(row.get("latest_amount_usd"), errors="coerce")
    if pd.notna(latest) and np.isfinite(float(latest)) and float(latest) >= 0:
        return float(latest), "latest_amount_usd"
    total = pd.to_numeric(row.get("total_raised_known_usd"), errors="coerce")
    if pd.notna(total) and np.isfinite(float(total)) and float(total) >= 0:
        return float(total), "total_raised_known_usd"
    return None, None


@app.get("/")
def home():
    return {"message": "SponsorMatch API is running!"}


@app.get("/health")
def health():
    active_ranker = getattr(app.state, "ranker", None) or ranker
    return {"status": "healthy" if active_ranker is not None else "degraded", "model_loaded": active_ranker is not None}


@app.get("/options")
def options():
    return {
        "sector_group": SECTORS,
        "stage_bucket": STAGES,
        "city_group": CITIES,
        "amount_bin": AMOUNT_BINS,
    }


@app.post("/rank")
def rank_investors(startup: StartupRequest):
    active_ranker = get_ranker(startup)
    row = startup.model_dump(exclude={"top_k"})
    try:
        scored = active_ranker.scores(pd.Series(row))
        profiles = read_csv_or_empty(INVESTOR_PROFILES_PATH)
        results = []
        seen = set()
        for name, score in scored:
            if name in seen:
                continue
            seen.add(name)
            results.append({
                "rank": len(results) + 1,
                "name": str(name),
                "score": float(max(0.0, min(1.0, score))),
                **profile_details(str(name), profiles),
            })
            if len(results) >= startup.top_k:
                break
        return {"startup": row, "ranked_investors": results}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to rank investors for this request.",
        )

@app.post("/match/startups")
def match_startups(request: StartupMatchRequest):
    pool = read_csv_or_empty(STARTUPS_POOL_PATH)
    if pool.empty:
        raise HTTPException(status_code=503, detail="Previously funded startup pool is unavailable.")
    required = {"company_name", "city_group", "sector_group", "latest_stage"}
    if not required.issubset(pool.columns):
        raise HTTPException(status_code=503, detail="Startup pool is missing required columns.")

    min_ticket = request.min_ticket_usd
    max_ticket = request.max_ticket_usd
    candidates = []
    for _, row in pool.iterrows():
        ticket, ticket_source = choose_ticket(row)
        # If a ticket range was requested, do not include rows with unknown ticket values.
        if min_ticket is not None or max_ticket is not None:
            if ticket is None:
                continue
            if min_ticket is not None and ticket < min_ticket:
                continue
            if max_ticket is not None and ticket > max_ticket:
                continue

        sector = str(row.get("sector_group", "Unknown")) if pd.notna(row.get("sector_group")) else "Unknown"
        city = str(row.get("city_group", "Unknown")) if pd.notna(row.get("city_group")) else "Unknown"
        stage = normalize_stage(row.get("latest_stage"))
        matched_rules = []
        possible_rules = []

        if request.preferred_sectors:
            possible_rules.append("sector")
            if sector in request.preferred_sectors:
                matched_rules.append("sector")
        if request.preferred_stages:
            possible_rules.append("stage")
            if stage in request.preferred_stages:
                matched_rules.append("stage")
        if min_ticket is not None or max_ticket is not None:
            possible_rules.append("ticket_range")
            # It has already passed the range filter above.
            matched_rules.append("ticket_range")
        if request.preferred_cities:
            possible_rules.append("city")
            if city in request.preferred_cities:
                matched_rules.append("city")

        fit_score = len(matched_rules) / len(possible_rules) if possible_rules else 0.0
        # When preferences are supplied, return startups matching at least one selected rule.
        if possible_rules and not matched_rules:
            continue
        candidates.append({
            "company_name": str(row.get("company_name", "Unknown")),
            "company_domain": json_value(row.get("company_domain")),
            "city": json_value(row.get("city")),
            "city_group": city,
            "sector_group": sector,
            "latest_stage": json_value(row.get("latest_stage")),
            "latest_round_year": json_value(row.get("latest_round_year")),
            "latest_amount_usd": json_value(row.get("latest_amount_usd")),
            "total_raised_known_usd": json_value(row.get("total_raised_known_usd")),
            "ticket_used_usd": ticket,
            "ticket_source": ticket_source,
            "fit_score": float(fit_score),
            "matched_rules": matched_rules,
        })

    candidates.sort(key=lambda item: (item["fit_score"], item["company_name"].lower()), reverse=True)
    return {
        "source": "Previously funded startups in the local dataset; not live startups currently seeking investment.",
        "count": min(len(candidates), request.top_k),
        "matched_startups": candidates[:request.top_k],
    }

