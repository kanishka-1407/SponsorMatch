"""Train the deployable model on train + validation after model selection."""
from pathlib import Path
import sys
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model import set_seed, train_model, train_final_model  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "deals_model.csv"
MODEL_PATH = ROOT / "artifacts" / "investor_model.pt"


def main():
    set_seed(42)
    df = pd.read_csv(DATA_PATH)
    train = df[df["split_random"] == "train"].copy()
    val = df[df["split_random"] == "val"].copy()
    train_val = pd.concat([train, val], ignore_index=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_model, _, _, _, _ = train_model(train, val, device)
    selected_epochs = getattr(selected_model, "best_epoch", 30)
    model, feature_maps, numeric_stats, investors = train_final_model(train_val, device, selected_epochs)
    safe_stats = {key: float(value) for key, value in numeric_stats.items()}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "feature_maps": feature_maps,
                "numeric_stats": safe_stats, "investors": [str(name) for name in investors]}, MODEL_PATH)
    print(f"Selected {selected_epochs} epochs on validation; saved final train+validation model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
