# SF2956 human activity analysis

The numerical results are produced by `run_analysis.py`. The notebook presents
the same analysis and reads the saved arrays from `output/`.

```bash
.venv/bin/python run_analysis.py
./run_jupyter.sh
```

The first command reuses saved topological features and reruns model fitting,
weight selection, evaluation, and `output/results.json`. To calculate every
persistence diagram again from the UCI HAR sensor windows, run:

```bash
.venv/bin/python run_analysis.py --recompute
```

The UCI HAR dataset is downloaded from the UCI Machine Learning Repository if
it is not already in `data/`. A smaller end-to-end check is available with:

```bash
.venv/bin/python run_analysis.py --smoke --recompute
```
