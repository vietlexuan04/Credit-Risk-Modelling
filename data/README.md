# Data

Dataset: **Give Me Some Credit** (Kaggle competition, [`GiveMeSomeCredit`](https://www.kaggle.com/c/GiveMeSomeCredit/data)).
Data is not committed to the repository (competition rules; see `.gitignore`). To reproduce:

1. Join the competition on Kaggle and download the files.
2. Place `cs-training.csv` in this folder, so the path is exactly `data/cs-training.csv`.
   `Data Dictionary.xls` can sit here too (reference only; no code reads it).
3. Run the notebooks in order. Notebook 02 creates `data/processed/`
   (`raw_dev.csv`, `raw_holdout.csv`), notebook 03 adds `dev.csv` and `holdout.csv`.

Expected layout after a full run:

```
data/
├── cs-training.csv        # you add this
├── Data Dictionary.xls    # optional
└── processed/             # generated, not committed
    ├── raw_dev.csv
    ├── raw_holdout.csv
    ├── dev.csv
    └── holdout.csv
```

Only `cs-training.csv` is used (150,000 rows, 10 predictors, binary target
`SeriousDlqin2yrs`). `cs-test.csv` has an empty target and is not used.
