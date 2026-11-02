
from pathlib import Path
import sys
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from model import set_seed, train_model

DATA_PATH = ROOT / "data" / "processed" / "deals_model.csv"
MODEL_DIR = ROOT / "artifacts"
MODEL_PATH = MODEL_DIR / "investor_model.pt"


def main():
    set_seed()

    df = pd.read_csv(DATA_PATH)

    train = df[df["split_random"] == "train"].copy()
    val = df[df["split_random"] == "val"].copy()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, feature_maps, numeric_stats, investors, best_score = (
        train_model(train, val, device)
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_maps": feature_maps,
            "numeric_stats": numeric_stats,
            "investors": investors,
        },
        MODEL_PATH,
    )

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Best validation nDCG@10: {best_score:.4f}")


if __name__ == "__main__":
    main()

