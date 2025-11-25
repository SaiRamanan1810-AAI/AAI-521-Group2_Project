# Plant Pest Classification (CNN + Transfer Learning)

Project scaffold for an image-based plant pest and disease classification system using PyTorch and Conda. This repository contains preprocessing, baseline CNN, transfer-learning scripts, visualization helpers, and documentation for team working with branch/PR workflow.

Quick start

- Create conda environment:

```bash
conda env create -f environment.yml
conda activate plant-pest
```

- Prepare dataset: place raw images in `data/raw/<class_name>/images...` or follow dataset README instructions.

- Preprocess & split:

```bash
python src/data.py --raw_dir data/raw --out_dir data/processed --img_size 224 --val_split 0.1 --test_split 0.1
```

- Train baseline model:

```bash
python src/train.py --mode baseline --data_dir data/processed --epochs 10 --batch_size 32 --img_size 224
```

- Fine-tune a pre-trained model:

```bash
python src/train.py --mode transfer --model_name resnet18 --data_dir data/processed --epochs 8 --batch_size 32
```

Repository layout

- `data/` - raw and processed dataset files
- `src/` - code for data loading, models, training, and utilities
- `models/` - saved checkpoints
- `notebooks/` - EDA and experiments
- `reports/` - plots, confusion matrices, metric tables
- `docs/` - project documentation
- `.github/` - contributing & workflow guidelines

Contributing

See `.github/CONTRIBUTING.md` for team workflow and PR guidelines.

License

See `LICENSE` in repository root.
