"""
Version 3: Neural network for startup -> investor ranking.

Uses:
Categorical features:
    - sector_group
    - stage_bucket
    - city_group
    - amount_bin

Numeric features:
    - amount_log10
    - company_age

Architecture:
Feature embeddings + normalized numeric features
        ->
Concatenation
        ->
2-layer feed-forward neural network
        ->
Scores for all investors

Model selection uses validation nDCG@10.
"""

from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from baselines import evaluate


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CAT_FEATURES = [
    "sector_group",
    "stage_bucket",
    "city_group",
    "amount_bin",
]

NUM_FEATURES = [
    "amount_log10",
    "company_age",
]

MISSING_VALUES = {
    "sector_group": "Unknown",
    "stage_bucket": "Undisclosed",
    "city_group": "Unknown",
    "amount_bin": "Undisclosed",
}

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 12
DROPOUT = 0.30


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Encoders
# ============================================================

def build_encoders(train):
    feature_maps = {}

    for feature in CAT_FEATURES:
        values = (
            train[feature]
            .fillna(MISSING_VALUES[feature])
            .astype(str)
        )

        categories = sorted(values.unique())

        # 0 is reserved for unseen categories
        feature_maps[feature] = {
            category: i + 1
            for i, category in enumerate(categories)
        }

    investors = sorted(train["investor"].astype(str).unique())

    investor_to_idx = {
        investor: i
        for i, investor in enumerate(investors)
    }

    return feature_maps, investor_to_idx, investors


# ============================================================
# Categorical encoding
# ============================================================

def encode_categorical(df, feature_maps):
    encoded = []

    for feature in CAT_FEATURES:

        values = (
            df[feature]
            .fillna(MISSING_VALUES[feature])
            .astype(str)
        )

        x = (
            values
            .map(feature_maps[feature])
            .fillna(0)
            .astype(int)
            .values
        )

        encoded.append(x)

    return np.column_stack(encoded)


# ============================================================
# Numeric preprocessing
# ============================================================

def fit_numeric_stats(train):
    """
    Learn imputation and normalization statistics ONLY from training data.
    """

    stats = {}

    # amount_log10
    amount = train["amount_log10"]

    median_amount = amount.median()

    amount_filled = amount.fillna(median_amount)

    stats["amount_median"] = median_amount
    stats["amount_mean"] = amount_filled.mean()
    stats["amount_std"] = amount_filled.std()

    # company_age
    age = train["company_age"]

    stats["age_mean"] = age.mean()
    stats["age_std"] = age.std()

    return stats


def encode_numeric(df, stats):
    """
    Creates:
    1. normalized amount_log10
    2. normalized company_age
    3. amount missing indicator
    """

    amount_missing = df["amount_log10"].isna().astype(float).values

    amount = (
        df["amount_log10"]
        .fillna(stats["amount_median"])
        .astype(float)
    )

    amount = (
        (amount - stats["amount_mean"])
        / (stats["amount_std"] + 1e-8)
    )

    age = df["company_age"].astype(float)

    age = (
        (age - stats["age_mean"])
        / (stats["age_std"] + 1e-8)
    )

    return np.column_stack([
        amount.values,
        age.values,
        amount_missing,
    ]).astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class StartupInvestorDataset(Dataset):

    def __init__(self, X_cat, X_num, y=None):

        self.X_cat = torch.tensor(
            X_cat,
            dtype=torch.long,
        )

        self.X_num = torch.tensor(
            X_num,
            dtype=torch.float32,
        )

        self.y = None

        if y is not None:
            self.y = torch.tensor(
                y,
                dtype=torch.long,
            )

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):

        if self.y is None:
            return (
                self.X_cat[idx],
                self.X_num[idx],
            )

        return (
            self.X_cat[idx],
            self.X_num[idx],
            self.y[idx],
        )


# ============================================================
# Neural Network
# ============================================================

class InvestorRankingNN(nn.Module):

    def __init__(
        self,
        feature_maps,
        num_investors,
        embedding_dim=16,
        hidden1=128,
        hidden2=64,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.embeddings = nn.ModuleList()

        for feature in CAT_FEATURES:

            num_categories = (
                len(feature_maps[feature]) + 1
            )

            self.embeddings.append(
                nn.Embedding(
                    num_categories,
                    embedding_dim,
                )
            )

        embedding_size = (
            embedding_dim * len(CAT_FEATURES)
        )

        # +3 numeric inputs:
        # amount_log10, company_age, amount_missing
        input_size = embedding_size + 3

        self.network = nn.Sequential(

            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_investors),
        )

    def forward(self, x_cat, x_num):

        embeddings = []

        for i, embedding in enumerate(self.embeddings):

            embeddings.append(
                embedding(x_cat[:, i])
            )

        x_embed = torch.cat(
            embeddings,
            dim=1,
        )

        x = torch.cat(
            [x_embed, x_num],
            dim=1,
        )

        return self.network(x)


# ============================================================
# Ranking wrapper
# ============================================================

class NeuralRanker:

    name = "Neural Network v3"

    def __init__(
        self,
        model,
        feature_maps,
        numeric_stats,
        investors,
        device,
    ):

        self.model = model
        self.feature_maps = feature_maps
        self.numeric_stats = numeric_stats
        self.investors = list(investors)
        self.device = device

    def rank(self, row):

        # -------- categorical --------

        cat_values = []

        for feature in CAT_FEATURES:

            value = row[feature]

            if pd.isna(value):
                value = MISSING_VALUES[feature]

            index = self.feature_maps[feature].get(
                str(value),
                0,
            )

            cat_values.append(index)

        # -------- numeric --------

        amount_is_missing = pd.isna(
            row["amount_log10"]
        )

        if amount_is_missing:

            amount = (
                self.numeric_stats["amount_median"]
            )

        else:

            amount = float(
                row["amount_log10"]
            )

        amount = (
            (amount - self.numeric_stats["amount_mean"])
            / (
                self.numeric_stats["amount_std"]
                + 1e-8
            )
        )

        age = float(row["company_age"])

        age = (
            (age - self.numeric_stats["age_mean"])
            / (
                self.numeric_stats["age_std"]
                + 1e-8
            )
        )

        num_values = [
            amount,
            age,
            float(amount_is_missing),
        ]

        # -------- tensors --------

        x_cat = torch.tensor(
            [cat_values],
            dtype=torch.long,
            device=self.device,
        )

        x_num = torch.tensor(
            [num_values],
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                x_cat,
                x_num,
            )

            scores = torch.softmax(
                logits,
                dim=1,
            )[0].cpu().numpy()

        order = np.argsort(-scores)

        return [
            self.investors[i]
            for i in order
        ]


# ============================================================
# Validation
# ============================================================

def validation_ndcg(
    model,
    feature_maps,
    numeric_stats,
    investors,
    val,
    device,
):

    ranker = NeuralRanker(
        model,
        feature_maps,
        numeric_stats,
        investors,
        device,
    )

    return evaluate(
        ranker,
        val,
    )["nDCG@10"]


# ============================================================
# Training
# ============================================================

def train_model(train, val, device):

    (
        feature_maps,
        investor_to_idx,
        investors,
    ) = build_encoders(train)

    numeric_stats = fit_numeric_stats(train)

    X_train_cat = encode_categorical(
        train,
        feature_maps,
    )

    X_train_num = encode_numeric(
        train,
        numeric_stats,
    )

    y_train = (
        train["investor"]
        .astype(str)
        .map(investor_to_idx)
        .astype(int)
        .values
    )

    train_dataset = StartupInvestorDataset(
        X_train_cat,
        X_train_num,
        y_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = InvestorRankingNN(
        feature_maps,
        len(investors),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_score = -1
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        total_loss = 0.0

        for (
            X_cat_batch,
            X_num_batch,
            y_batch,
        ) in train_loader:

            X_cat_batch = X_cat_batch.to(device)
            X_num_batch = X_num_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(
                X_cat_batch,
                X_num_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(X_cat_batch)
            )

        train_loss = (
            total_loss
            / len(train_dataset)
        )

        score = validation_ndcg(
            model,
            feature_maps,
            numeric_stats,
            investors,
            val,
            device,
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val nDCG@10: {score:.4f}"
        )

        if score > best_score:

            best_score = score

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"Early stopping at epoch "
                    f"{epoch + 1}"
                )

                break

    model.load_state_dict(best_state)

    print(
        f"Best validation nDCG@10: "
        f"{best_score:.4f}"
    )

    return (
        model,
        feature_maps,
        numeric_stats,
        investors,
        best_score,
    )


# ============================================================
# Run one split
# ============================================================

def run_split(
    df,
    split_col,
    label,
    device,
):

    train = df[
        df[split_col] == "train"
    ].copy()

    val = df[
        df[split_col] == "val"
    ].copy()

    test = df[
        df[split_col] == "test"
    ].copy()

    print("\n" + "=" * 70)
    print(label)

    print(
        f"Train={len(train)}, "
        f"Val={len(val)}, "
        f"Test={len(test)}"
    )

    (
        model,
        feature_maps,
        numeric_stats,
        investors,
        best_val_score,
    ) = train_model(
        train,
        val,
        device,
    )

    ranker = NeuralRanker(
        model,
        feature_maps,
        numeric_stats,
        investors,
        device,
    )

    results = evaluate(
        ranker,
        test,
    )

    results["split"] = label
    results["model"] = ranker.name
    results["best_val_nDCG@10"] = best_val_score

    return results


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    REPORTS.mkdir(
        exist_ok=True,
    )

    df = pd.read_csv(
        PROC / "deals_model.csv"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    results = []

    results.append(
        run_split(
            df,
            "split_random",
            "Random split (by company)",
            device,
        )
    )

    results.append(
        run_split(
            df,
            "split_time",
            "Time split (train<=2023, test 2025-26)",
            device,
        )
    )

    results_df = pd.DataFrame(results)

    columns = [
        "split",
        "model",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@20",
        "MRR",
        "nDCG@10",
        "Coverage@10",
        "n_test",
        "best_val_nDCG@10",
    ]

    results_df = results_df[columns]

    results_df.to_csv(
        REPORTS / "model_results.csv",
        index=False,
    )

    md = [
        "# Neural network model results (Version 3)",
        "",
        "## Model",
        "",
        "A startup-to-investor ranking model using categorical feature "
        "embeddings combined with normalized numeric features, followed "
        "by a two-layer feed-forward neural network.",
        "",
        "Categorical features: "
        + ", ".join(CAT_FEATURES),
        "",
        "Numeric features: "
        + ", ".join(NUM_FEATURES),
        "",
        "Missing amount_log10 values are imputed using the training "
        "median and accompanied by a missing-value indicator.",
        "",
        "Model selection and early stopping use validation nDCG@10.",
        "",
        "## Results",
        "",
        results_df.round(4).to_markdown(index=False),
    ]

    (REPORTS / "model_results.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print("\nFINAL RESULTS")

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()