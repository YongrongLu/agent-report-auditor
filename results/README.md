# Results

| Folder | Run | Notes |
|---|---|---|
| `v2_1/` | main run | outcome-free pressure wording, length-matched neutral context, LODO layer choice (layer 24) |
| `v2_first_run/` | first run | pressure sentence contained "failed"; validation-based layer choice collapsed to layer 1; includes `human_check.csv` |

Files in each run:
- `samples.csv` — one row per report: scenario factors, log, model completion, parsed claim, category, cross-fitted probe probabilities.
- `results.json` — every statistic reported on the poster (behaviour, probe curves, controls, gap analysis, steering, patching).
- `explorer_data.json` — input for `src/build_explorer.py`.
- `figures/` — notebook figures (`fig1`–`fig4`) and, for v2.1, poster figures (`E1`–`E4`).
