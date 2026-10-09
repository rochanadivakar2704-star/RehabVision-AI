# RehabVision AI — deadline kit

## Setup (Ubuntu/WSL)
```bash
cd ~/RehabVision_AI
python3 -m venv rehabvision_env
source rehabvision_env/bin/activate
pip install -r requirements.txt
```
Copy the kit contents into `~/RehabVision_AI` or work in the extracted folder.

## Dataset audit
Download REHAB24-6 from https://zenodo.org/records/13305825 and place the files under `data/raw/`. Then run:
```bash
python audit_dataset.py
```
Inspect the dataset documentation and CSV output; do not assume a particular annotation schema.

## ML training
Prepare `data/manifest.csv` with actual annotations and columns `video_path,label,participant_id` (optional `exercise`). See `DATASET_MANIFEST_README.md`.
```bash
python train_model.py
```
The script requires at least 8 usable videos, at least two classes, and at least 3 participant IDs. If these conditions are not met, do not fabricate rows or metrics. Describe the model as not completed and demonstrate feature-analysis mode.

## Run the app
```bash
streamlit run app.py
```
Upload a short video and click Analyse video. It displays pose-derived features, an overlay if detected, a time-series plot, and a history table. The app can run without a trained model, but it will say feature-analysis mode rather than inventing a prediction.

## Report integrity
All metrics must come from `results/model_metrics.csv`. Figures are saved to `results/figures/`. Capture screenshots from your own running app using your OS screenshot tool and save them under `results/screenshots/`. Do not present example/mockup images as your app screenshots.
