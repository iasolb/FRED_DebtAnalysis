# FRED_DebtAnalysis

The data and notebooks behind a Substack piece on the 100 percent
government debt threshold. Analysis is built on FRED data pulled through a
local loader (`loader.py`) and scoring layer (`macro_scores.py`), with
[otter](https://pypi.org/project/otter/) for the modeling.

## Read first

[The 100% threshold: how government debt...](https://open.substack.com/pub/iasolb/p/the-100-threshold-how-government?r=7u0phf&utm_campaign=post-expanded-share&utm_medium=post%20viewer)

## Reproduce

- `DebtRegime_M2Passthrough.ipynb`: main analysis
- `DebtRegime_RobustnessChecks.ipynb`: robustness checks
- `charts/` and `output_graphs/`: figures used in the piece
- `data/fred_master.csv`: the FRED pull; `data/health_service_gap.csv`
- `test_macro_scores.py`: tests for the scoring layer

`pip install -r requirements.txt` pulls everything, including otter. There is
no submodule any more: otter was published to PyPI on 2026-09-11, so the
notebooks import it like any other dependency.

## Start here

Read the Substack piece above, then open `DebtRegime_M2Passthrough.ipynb` to
see how the figures were produced.
