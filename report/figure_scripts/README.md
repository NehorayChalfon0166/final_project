# Academic figure generation

Regenerates every figure in the project book (`report/`) in a single, consistent
"academic paper" style — serif type matching the LaTeX body, a muted
monochrome-blue palette, hairline grids, no chart-junk — as **vector PDFs** in
`report/figures/`.

## Run

```bash
# from the project root, inside the project venv
python report/figure_scripts/make_figures.py
```

Then recompile the report:

```bash
cd report && latexmk -xelatex main.tex
```

## What it produces

| File | Needs the venv? | Source |
| --- | --- | --- |
| `fig_method_overview.pdf`   | no  | schematic |
| `fig_architecture.pdf`      | no  | schematic |
| `fig_training_curves.pdf`   | no  | `outputs/gnn_training_history.json` |
| `fig_confusion_panel.pdf`   | no  | `outputs/evaluation/three_model_comparison.json` |
| `fig_metric_comparison.pdf` | no  | `outputs/evaluation/three_model_comparison.json` |
| `fig_feature_importance.pdf`| no  | `outputs/evaluation/feature_importance.json` |
| `fig_roc.pdf`               | **yes** | re-scores the cached test ego-graphs |
| `fig_reliability.pdf`       | **yes** | re-scores + applies `outputs/temperature.pt` |

The last two load the trained models (`outputs/gnn_model.pt`,
`outputs/baseline/gcn_model.pt`) and re-score every wallet in the shared test
split, exactly like `scripts/viz/plot_roc_curve.py`. They therefore need
`torch`, `torch_geometric`, `xgboost`, `scikit-learn`, and `pandas` — i.e. run
inside the project venv. Per-sample scores are cached to
`outputs/evaluation/test_scores.npz`, so the second figure is instant and you can
delete that file to force a fresh re-score.

If those packages are missing, the script prints `[skip] roc / reliability …`
and still produces the other six figures. The report references `fig_roc.pdf`
and `fig_reliability.pdf` through `\IfFileExists`, so `main.tex` compiles whether
or not they exist yet — once you generate them and recompile, they appear
automatically.

## Options

```bash
python report/figure_scripts/make_figures.py --no-score      # the six that need no venv
python report/figure_scripts/make_figures.py --only roc reliability
python report/figure_scripts/make_figures.py --only training confusion
```

## Style

`academic_style.py` holds the shared palette and `apply()` (rcParams). Import it
and call `apply()` once to reuse the same look in any other figure script.
