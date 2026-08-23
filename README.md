# FRED_DebtAnalysis

The data and notebooks behind a Substack piece on the 100 percent
government debt threshold. Analysis is built on FRED data pulled through a
local loader (`loader.py`) and scoring layer (`macro_scores.py`), with the
`ResearchFramework` submodule for the modeling.

## Read first

[The 100% threshold: how government debt...](https://open.substack.com/pub/iasolb/p/the-100-threshold-how-government?r=7u0phf&utm_campaign=post-expanded-share&utm_medium=post%20viewer)

## Reproduce

- `DebtRegime_M2Passthrough.ipynb`: main analysis
- `DebtRegime_RobustnessChecks.ipynb`: robustness checks
- `charts/` and `output_graphs/`: figures used in the piece
- `data/fred_master.csv`: the FRED pull; `data/health_service_gap.csv`
- `test_macro_scores.py`: tests for the scoring layer

Clone with `--recursive` to pull the `ResearchFramework` submodule.

## Start here

Read the Substack piece above, then open `DebtRegime_M2Passthrough.ipynb` to
see how the figures were produced.
