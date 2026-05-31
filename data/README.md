# Data Directory

This directory holds the PlantVillage dataset used for training and evaluation.

## Structure

```
data/
├── Tomato__Tomato_mosaic_virus/
│   ├── image001.jpg
│   └── ...
├── Tomato_healthy/
│   └── ...
└── <ClassName>/
    └── ...
```

Each subdirectory name becomes a class label. Names must be consistent between
training and inference runs (they are sorted alphabetically to assign indices).

## Download

```bash
# Using the Kaggle CLI
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

Dataset source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset  
Original paper: Hughes & Salathé (2015) — https://doi.org/10.1371/journal.pcbi.1004993

## Included sample data

The `tomato_healthy/` and `tomato_leaf_blight/` subdirectories contain a small
number of sample images for smoke-testing the pipeline. They are **not**
sufficient for training a production model.
