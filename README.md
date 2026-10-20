# MovieLens Recommender Systems

Team project for the Data Lab course at the Catholic University
of Eichstätt-Ingolstadt, summer semester 2026.

The project compares low-rank matrix completion methods for movie-rating
prediction on the MovieLens `ml-latest-small` and `ml-latest` datasets:
- Matrix factorization using gradient descent (GD), stochastic gradient
  descent (SGD), and alternating least squares (ALS).
- Hard and soft singular value thresholding.

Experiments examine predictive performance, convergence, and computational
efficiency, using sparse representations, Numba acceleration, and the
Fritz HPC cluster.

## Project report

[Read the project report (PDF)](MovieLens_Recommender_Systems_Report.pdf)

## Authors

- Nikita Mykytenko
- Oleksandr Bobryshev
- Labeat Dobraj
## Repository structure

```text
Data-Lab-Submission/
├── notebooks/
│   ├── 00_download_data.ipynb
│   ├── 01_exploration_ml-latest-small.ipynb
│   ├── 01_exploration_ml-latest.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── final_ml-latest-small_models.ipynb
│   ├── final_ml_latest_models.ipynb
│   ├── model_comparison_graphs.ipynb
│   ├── gd_*_hpc.ipynb
│   └── hard_thresholding_*_hpc.ipynb
├── src/
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── models.py
│   ├── als.py
│   ├── sgd.py
│   ├── gd.py
│   ├── gd_sparce.py
│   ├── gd_sparce_numba.py
│   └── hard_thresholding_hpc.py
├── hpc/
│   ├── gd_*.slurm
│   └── hard_thresholding_*.slurm
├── MovieLens_Recommender_Systems_Report.pdf
└── README.md
```

## Notebook order

1. `00_download_data.ipynb` — download both MovieLens datasets if reproduction is needed.
2. `01_exploration_ml-latest-small.ipynb` and
   `01_exploration_ml-latest.ipynb` — explore the datasets (optional for later steps).
3. `02_preprocessing.ipynb` — create train/test splits and processed data.
4. `final_ml-latest-small_models.ipynb` — hyperparameter selection on a
   train-validation split, followed by retraining on the full training set and
   final evaluation on the test set for `ml-latest-small`.
5. `final_ml_latest_models.ipynb` — the same selection and final-evaluation
   workflow for ALS and SGD on `ml-latest`. These models were run on a personal
   computer.
6. `model_comparison_graphs.ipynb` — generate the final model-comparison plots, including those for models run on the Fritz HPC cluster.

For `ml-latest`, Hard Thresholding and GD were run on the Fritz HPC cluster using the
`hard_thresholding_*_hpc.ipynb` and `gd_*_hpc.ipynb` notebooks together with the
matching SLURM scripts in `hpc/`.

### HPC notebook sequences

Every HPC stage is included in two versions: `*_hpc.ipynb` contains the source
notebook, while `*_hpc_executed.ipynb` contains the saved outputs from its Fritz
run. The executed search notebooks document the hyperparameter-selection
results. The `*_final_training_hpc_executed.ipynb` notebooks contain the final
runs on the full training set and their test-set evaluation. The sequences below
list the source notebooks; the corresponding executed versions follow the same
order.

GD:

```text
gd_coarse_search_hpc.ipynb
→ gd_refined_search_hpc.ipynb
→ gd_fine_search_hpc.ipynb
→ gd_final_training_hpc.ipynb
→ gd_final_training_hpc_executed.ipynb
```

Hard Thresholding:

```text
hard_thresholding_coarse_search_hpc.ipynb
→ hard_thresholding_refined_search_hpc.ipynb
→ hard_thresholding_long_search_hpc.ipynb
→ hard_thresholding_final_fine_search_hpc.ipynb
→ hard_thresholding_final_convergence_hpc.ipynb
→ hard_thresholding_final_training_hpc.ipynb
→ hard_thresholding_final_training_hpc_executed.ipynb
```

## File reference

### `src/`

| File | Purpose |
|---|---|
| `preprocessing.py` | Builds rating matrices from processed ratings |
| `evaluation.py` | RMSE, MAE, and model-evaluation functions |
| `models.py` | Hard and soft thresholding models used for the small dataset |
| `als.py` | ALS implementations |
| `sgd.py` | Numba-accelerated stochastic gradient descent |
| `gd.py` | Dense gradient descent |
| `gd_sparce.py` | Sparse gradient descent |
| `gd_sparce_numba.py` | Numba-accelerated sparse gradient descent |
| `hard_thresholding_hpc.py` | Hard-thresholding functions used by HPC notebooks |
