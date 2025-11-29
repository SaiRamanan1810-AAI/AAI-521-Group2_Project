# Multi-Crop Disease Classifier (CCMT)

## Architecture

This project implements a Two-Stage classification pipeline for plant disease detection across four species: Cashew, Cassava, Maize, and Tomato.

- Stage 1: Plant Species Classifier (4-way) — EfficientNet-B0 pretrained on ImageNet.
- Stage 2: Per-species Disease Classifiers — Four separate EfficientNet-B0 models, one for each species. Each disease model's output size equals the number of disease classes for that species.
- Inference Router: Predicts species first; if confident, routes image to corresponding disease model. Returns JSON with plant and disease predictions + confidences.

## Project layout

- `data/` (not included) — dataset root; see Data Structure.
- `src/` — Python source modules:
  - `data.py` — data preparation, stratified splits, Dataset classes, transforms.
  - `model.py` — EfficientNet-B0 loaders and helpers.
  - `train.py` — training loop with two-step transfer learning, scheduler, and early stopping.
  - `inference.py` — `InferencePipeline` router.
  - `eda.py` — exploratory data analysis plots and sanity checks.
  - `visualize.py` — training curves, confusion matrices, confidence histograms.
- `models/` — checkpoints (excluded from git).

## Data Structure

Expected directory layout:

```
data/
  plants/
    Cashew/
      img1.jpg
      img2.jpg
    Cassava/
    Maize/
    Tomato/

  diseases/
    Cashew/
      <disease_class_1>/
        img1.jpg
      <disease_class_2>/
    Cassava/
      <disease_class_a>/
      <disease_class_b>/
    Maize/
    Tomato/
```

Notes:
- `data/` is intentionally gitignored; ensure you place your images in this layout.
- Plant classifier expects per-species folders directly under `data/plants/`.
- Disease classifier expects per-species subfolders, each containing one folder per disease class.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate plant-pest
```

Install a specific PyTorch build if needed following https://pytorch.org.

### 2. Extract and organize dataset

If you have the dataset in `data.zip`, run:

```bash
python scripts/setup_data.py
```

This will automatically extract and organize the data into the expected structure under `data/plants/` and `data/diseases/`.

**Options:**
- `--zip-path <path>` — specify a different zip file location (default: `data.zip`)
- `--output-dir <dir>` — specify output directory (default: `data`)
- `--force` — force re-extraction even if data directories exist

The script will verify the structure and report species counts after extraction.

## Usage

Use the CLI scripts below to run common tasks (examples):

Run the pipeline in this order:

1) Extract dataset (if using data.zip)

```bash
python scripts/setup_data.py
```

2) Run EDA to inspect datasets and generate reports

```bash
python scripts/run_eda.py --plants-data data/plants --diseases-data data/diseases --out-dir reports
```

3) Prepare stratified splits (CSV + metadata)

```bash
python scripts/prepare_splits.py --plants-data data/plants --diseases-data data/diseases --out-dir data_splits
```

4) Train the Plant (Stage-1) Classifier using the prepared splits

```bash
python scripts/train_plant.py --splits-dir data_splits --out-dir models --batch-size 32 --epochs-head 5 --epochs-ft 15 --device cpu
```

Training automatically generates:
- Model checkpoint: `models/plant_checkpoint.pth`
- Metadata: `models/plant_checkpoint.pth.meta.json`
- Training history: `models/plant_history.json`
- Training plot: `models/plant_training.png`

5) Train Disease (Stage-2) Classifiers for each species using the prepared splits

```bash
python scripts/train_disease.py --species Cashew --splits-dir data_splits --out-dir models --device cpu
python scripts/train_disease.py --species Cassava --splits-dir data_splits --out-dir models --device cpu
python scripts/train_disease.py --species Maize --splits-dir data_splits --out-dir models --device cpu
python scripts/train_disease.py --species Tomato --splits-dir data_splits --out-dir models --device cpu
```

Each training run automatically generates:
- Model checkpoint: `models/{Species}_checkpoint.pth`
- Metadata: `models/{Species}_checkpoint.pth.meta.json`
- Training history: `models/{Species}_history.json`
- Training plot: `models/{Species}_disease_training.png`

6) Evaluate models (Stage-1 and Stage-2)

```bash
python scripts/evaluate.py --plant-checkpoint models/plant_checkpoint.pth --models-dir models --plants-data data/plants --diseases-data data/diseases --out-dir reports --device cpu
```

This will generate confusion matrices and confidence histograms for both stages in the `reports/` directory.

7) Run Unit Tests

```bash
python -m unittest discover tests
```

**Training Plots**

To generate/regenerate training plots from existing history files:

```bash
python scripts/plot_training.py --models-dir models --out-dir models
```

This will generate plots for all models that have history JSON files.

**Notes**
- Checkpoints are saved to `models/` as `<name>_checkpoint.pth` and metadata is saved alongside as `<checkpoint>.meta.json` (contains class names and other metadata used by the `InferencePipeline`).
- Training scripts automatically generate loss/accuracy plots and save them alongside checkpoints.
- Evaluation outputs (plots) and EDA reports are written to `reports/` by default.

## Notes & Recommendations

- This project uses raw PyTorch training loops (no Lightning) for transparency and easier debugging.
- Adjust learning rates, epochs, batch sizes, and other hyperparameters in `src/train.py` as needed.
- Save model metadata (e.g., class names) as JSON alongside checkpoints for easy inference routing.
