# SponsorMatch

A local startup and investor matching app. It combines a hybrid investor ranking model, a historical funded-startup explorer, registered member discovery, and editable pitch requests. The web interface uses a warm sand, clay, and olive palette and is served by the API.

## Run locally

Python 3.10 or newer is recommended.

```bash
cd SponsorMatch-pitch-feature
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the API and frontend together:

```bash
uvicorn api:app --reload
```

Open <http://127.0.0.1:8000>. API documentation is at <http://127.0.0.1:8000/docs> and health at <http://127.0.0.1:8000/health>.

The SQLite database is created automatically as `sponsormatch.sqlite3`. Set `SPONSORMATCH_DB` to use another path. Accounts are local to this app. Startup and investor accounts get separate workspaces after sign-in. Startups can rank investors and prepare pitch requests; investors can set their thesis and ticket range, browse matching funded startups, and save companies to a personal watchlist. The investor workflow does not create pitches. Pitch requests change state when the recipient responds, and contact email is disclosed only after acceptance.

## Matching behavior

- `POST /rank` returns the blended neural and rule-based affinity ranking. `score` is a relative percentile within the model's investor pool (0–100); `model_probability` is included separately and is not a deal probability.
- The rank form accepts sector, stage, city, amount bin, log amount, and company age.
- `/pitch/discover` finds actual registered users and ranks them by overlap in their sector, stage, and city profile.
- Pitch generation uses the startup's real registered amount and age. When a registered investor's name is not in the historical dataset, the pitch match falls back to profile overlap instead of dropping the match.
- `POST /match/startups` searches historical funded startups. Results are examples from the dataset, not live companies seeking investment; equal-fit results use latest funding year and then company name.
- Investors use `GET/PUT /pitch/investor/preferences` for saved sectors, stages, cities, ticket bounds, and thesis, and `GET/POST/DELETE /pitch/investor/watchlist` to manage saved startups.

Registration sectors, stages, and cities are restricted to the values returned by `GET /options`. Login is limited to five failed attempts per email before a 15-minute cooldown. Outgoing pitches are limited to five per rolling week, with duplicate pending pairs rejected.

## Model workflow

Install dependencies, then run the five-seed evaluation:

```bash
python src/multi_seed_evaluation.py
```

It evaluates neural and hybrid rankings on the held-out random test split and writes seed-level and aggregate CSVs under `reports/`. Keep the test split out of model selection. Then select the stopping epoch on the validation split and train the API checkpoint on train plus validation rows:

```bash
python train_for_api.py
```

The final trainer selects the epoch count on train/validation, then refits on their combined rows. The saved checkpoint contains tensors, plain Python mappings, floats, and strings and is loaded with PyTorch's restricted `weights_only=True` loader. An older checkpoint that is rejected by the safe loader uses an affinity-only fallback until you run `python train_for_api.py` to replace it.

Preprocess the raw funding data and regenerate derived files with:

```bash
python src/preprocess.py
```

The API expects the processed files under `data/processed/`. Do not run preprocessing if you want to retain the current supplied derived data.

## Tests

Run the test suite with:

```bash
pytest -q
```

It includes API validation, a real-checkpoint `/rank` flow, pitch request behavior, preprocessing checks, and model ranking checks.

## Main files

- `api.py` — FastAPI service, ranking and funded-startup matching.
- `pitch_requests.py` — authentication, registered-user discovery, and pitch request workflow.
- `frontend/index.html` — responsive web app.
- `src/model.py` — neural ranker and train/final-train routines.
- `src/baselines.py` — affinity baseline used by the hybrid.
- `src/multi_seed_evaluation.py` — five-seed evaluation.
- `train_for_api.py` — final train-plus-validation checkpoint builder.
