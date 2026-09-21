# Dynamics of Role-Based Passing Networks in Elite Football

Code for my MSc Data Science dissertation (INM363, City St George's, University of London, 2026).
Supervisor: Dr Gennady Andrienko.

The project builds role-based passing networks for Manchester City's 2017/18 Premier League season from Wyscout event data alone, then clusters them into tactical variants (RQ1), studies how those variants change across the season (RQ2), and tests whether they relate to match outcomes (RQ3).

## Files

| File | What it does |
|---|---|
| `Dissertation_python_notebook.ipynb` | Main notebook. Runs the full pipeline and produces every figure and table in the report. |
| `data_loading.py` | Loads the Wyscout JSON files and filters them to Manchester City. |
| `possession_episodes.py` | Splits matches into possession episodes, applies the minimum thresholds and merges episodes across short interruptions. |
| `role_assignment.py` | Assigns each outfield player to one of 10 role slots per time segment using the Hungarian algorithm. |
| `network_construction.py` | Builds the role-based passing network for each episode and computes clustering features. |
| `code_appendix.txt` | All of the code above in one text file (for the originality check). |
| `*.png` | Figures used in the report. |
| `*.npy` | Saved episode feature matrices. |
| `genai_use_log.md` | Log of how generative AI (Claude) was used during the project. |

## How to run

1. Download the Wyscout public dataset (Pappalardo et al., 2019, *Scientific Data* 6, 236, https://doi.org/10.1038/s41597-019-0247-7). You need `events_England.json`, `matches_England.json`, `players.json` and `teams.json`. These are not in the repo because of their size.
2. Put the four JSON files in the same folder as the notebook and the `.py` modules.
3. Install the dependencies (Python 3.13 was used):
   ```
   pip install pandas numpy networkx scikit-learn umap-learn scipy statsmodels matplotlib seaborn jupyter
   ```
4. Open `Dissertation_python_notebook.ipynb` and run all cells in order.

## Data

The Wyscout data is published by Pappalardo et al. (2019) and is not my own work.
