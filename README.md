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

2) Run comprehensive EDA to inspect datasets and generate reports

```bash
python scripts/run_eda.py --data-root data --out-dir reports
```

To automatically remove duplicate images (optional):
```bash
python scripts/run_eda.py --data-root data --out-dir reports --remove-duplicates
```

The comprehensive EDA includes:
- **Class count summary** — Distribution of samples across all classes
- **Class balance analysis** — Visual charts showing imbalance metrics
- **Duplicate detection** — Identify duplicate images using file hashes
- **Duplicate removal** — Automatically remove duplicates, keeping first occurrence (optional with `--remove-duplicates`)
- **Corrupted image detection** — Find and quarantine unreadable images
- **Dimension analysis** — Width, height, and aspect ratio statistics with visualizations
- **Imbalance summary** — Imbalance ratios and coefficient of variation for each class
- **HTML report export** — Interactive HTML report with all findings and embedded visualizations

Generated outputs:
- `reports/eda_report.html` — Comprehensive interactive report with all visualizations embedded
- `reports/plant_class_balance.png` — Plant species distribution
- `reports/{Species}_disease_class_balance.png` — Disease distribution per species
- `reports/data_samples.png` — Sample grid of plant images with labels
- `reports/{Species}_disease_samples.png` — Sample grids of disease images per species
- `reports/dimension_analysis.png` — Image dimension statistics and distributions
- `data/bad_images/` — Directory containing corrupted images (if any found)
- `data/dup_images/` — Directory containing duplicate images (if `--remove-duplicates` used)
- `data/dup_images/duplicates_log.txt` — Log of all moved duplicate files

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

The evaluation generates comprehensive metrics and visualizations for both stages:

**Metrics Calculated:**
- **Accuracy** — Overall correct predictions
- **Precision** — Macro-averaged (treats all classes equally) and weighted (accounts for class imbalance)
- **Recall** — Macro-averaged and weighted
- **F1-Score** — Macro-averaged and weighted
- **Per-Class Metrics** — Precision, recall, and F1 for each individual class

**Outputs Generated:**
- `reports/stage1_metrics.json` — Plant classifier metrics in JSON format
- `reports/stage1_confusion_matrix.png` — Plant species confusion matrix
- `reports/stage1_confidence_hist.png` — Plant prediction confidence distribution
- `reports/stage2_{Species}_metrics.json` — Disease classifier metrics per species
- `reports/stage2_{Species}_confmat.png` — Disease confusion matrices per species
- `reports/stage2_{Species}_conf_hist.png` — Disease confidence distributions per species
- `reports/stage2_all_species_metrics.json` — Combined metrics for all disease classifiers

Metrics are displayed in console with formatted tables and saved to JSON files for further analysis.

7) Run Inference on new images

**Option A: Interactive Web App (Streamlit) - Recommended**

Launch the web application:
```bash
streamlit run app.py
```

Or use the quick start script:
```bash
./run_app.sh
```

The web app provides:
- **Interactive Image Upload** — Drag & drop or browse to upload images
- **Real-Time Classification** — Instant two-stage disease detection
- **Visual Confidence Scores** — Progress bars and color-coded results
- **Adjustable Threshold** — Fine-tune confidence requirements via sidebar
- **Downloadable Results** — Export predictions as JSON
- **Responsive Design** — Works on desktop, tablet, and mobile
- **Clean Interface** — User-friendly with clear visual feedback
- **Detailed Summary** — Complete breakdown of both classification stages

**Features:**
- Configurable model paths and device selection (CPU/CUDA/MPS)
- Color-coded confidence indicators (green > 80%, yellow > 60%, red < 60%)
- Two-stage pipeline visualization with species and disease detection
- Threshold-based filtering for reliable predictions
- Summary table with all classification metrics
- Download button for JSON export of results

**Option B: Command-Line Interface**

**Single image prediction:**
```bash
python scripts/run_inference.py --image path/to/image.jpg --device cpu
```

**Multiple images:**
```bash
python scripts/run_inference.py --image img1.jpg img2.jpg img3.jpg --output predictions.json --device cpu
```

**Directory of images:**
```bash
python scripts/run_inference.py --image-dir path/to/images --output results.json --device cpu
```

**Quiet mode (minimal output):**
```bash
python scripts/run_inference.py --image-dir data/plants/Tomato --output tomato_results.json --device cpu --quiet
```

**Available Options:**
- `--image` — One or more image file paths
- `--image-dir` — Directory containing images to process
- `--plant-checkpoint` — Path to plant model checkpoint (default: `models/plant_checkpoint.pth`)
- `--models-dir` — Directory containing disease model checkpoints (default: `models`)
- `--device` — Device to use: `cpu`, `cuda`, or `mps` (default: `cpu`)
- `--threshold` — Confidence threshold for disease routing (default: `0.5`)
- `--output` / `-o` — Save predictions to JSON file
- `--quiet` / `-q` — Suppress verbose output, only show summary

**The Two-Stage Inference Pipeline:**
1. **Stage 1 (Plant Classification):** Predicts plant species from {Cashew, Cassava, Maize, Tomato}
2. **Stage 2 (Disease Classification):** Routes to species-specific disease classifier
3. **Output Format:** Returns predictions with confidence scores for both stages
4. **Human-Readable Results:** Uses actual species and disease names from metadata

**Output Format (JSON):**
```json
{
  "image.jpg": {
    "plant_species": "Tomato",
    "plant_confidence": 0.98,
    "disease": "leaf blight",
    "disease_confidence": 0.92,
    "above_threshold": true
  }
}
```

**Example Output (Console):**
```
Image: tomato_leaf.jpg
  Plant Species: Tomato (confidence: 0.98)
  Disease: leaf blight (confidence: 0.92)
  Status: ✓ Above threshold
```

8) Run Unit Tests

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
