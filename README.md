# Human activity from sensor topology

This branch is a clean mathematical rewrite of the SF2956 project. Read
`ANALYSIS_SPEC.md` before the code. There is deliberately no notebook yet: the
ordinary Python pipeline must work completely before a notebook is generated.

## Files

- `ANALYSIS_SPEC.md` states the research question, equations, controls, and
  acceptance criteria.
- `data.py` downloads and loads the UCI HAR data.
- `tda.py` implements only the topological feature map.
- `evaluate.py` contains one fixed subject-grouped evaluation protocol.
- `run_analysis.py` explicitly runs the representation comparisons and the
  development-only joint-sensor weight analysis.
- `test_math.py` tests the mathematical construction on synthetic and real data.
- `build_notebook.py` generates the explanatory notebook only after the tested
  analysis has completed.

## Environment

Python 3.13 and the pinned packages in `requirements.txt` are used. With `uv`:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/python -m unittest -v test_math.py
.venv/bin/python run_analysis.py --smoke --recompute
.venv/bin/python run_analysis.py
.venv/bin/python build_notebook.py
./run_jupyter.sh
```

The smoke run uses one window per subject and activity. The full run writes
direct numerical caches and `output/results.json`. Delete `output/cache/` or
pass `--recompute` to force feature extraction. No fitted model is serialized.

The generated notebook presents the tested pipeline in the same mathematical
order. It does not contain a second implementation of the analysis.

## Report

After the full analysis has created `output/results.json` and its caches:

```bash
.venv/bin/python report/make_report_figures.py
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=../output/pdf report.tex
```

The final PDF is `output/pdf/report.pdf`.
