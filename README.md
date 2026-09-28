# Milan Mobile Network Traffic Forecasting

Comparing three forecasting models (**SARIMA**, **gradient-boosted trees / LightGBM**, and **LSTM**) for one-step-ahead Internet traffic prediction on the Telecom Italia Milan dataset.

Everything runs from a single notebook: `notebook.ipynb`. Run it top to bottom and it loads the data, does the analysis, tunes and trains the models, and saves all figures and tables to disk.

---

## 1. What you need

| Requirement | Details |
|---|---|
| **Python** | 3.9 (the notebook was built and tested on **3.9.6**) |
| **RAM** | **8 GB minimum, 16 GB recommended.** Loading is memory-optimised (~1.4 GB), but the wide table built afterwards is larger. |
| **Disk space** | Enough for the raw dataset (several GB) plus a little for outputs |
| **GPU** | Not needed. Everything was run on CPU. |
| **Time** | Plan on **roughly 30-90 minutes** on a laptop CPU. Most of it is LSTM tuning and training. Times vary a lot by machine. |

---

## 2. Get the project set up

### Step 1: Put the files in one folder

```
your-project/
├── notebook.ipynb
├── README.md
├── requirements.txt      <- you'll create this in Step 3
└── data/                 <- you'll fill this in Step 4
```

### Step 2: Create a virtual environment (recommended)

A virtual environment keeps this project's packages separate from everything else on your computer.

**macOS / Linux**
```bash
python3.9 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
py -3.9 -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` at the start of your terminal line.

### Step 3: Install the packages

Create a file called `requirements.txt` with the following contents:

```
numpy
pandas
matplotlib
scikit-learn
statsmodels
tensorflow
lightgbm
psutil
jupyter
```

Then install everything:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes**
- **Apple Silicon Mac (M1/M2/M3)?** If `pip install tensorflow` fails, try `pip install tensorflow-macos` instead.
- **PyTorch is optional.** The notebook only uses it to check whether a GPU exists. If it's not installed, it simply skips that check.
- **LightGBM won't load?** On macOS you may see an error about `libomp` / OpenMP. Fix it with `brew install libomp`. If you can't fix it, **don't worry**: the notebook automatically falls back to scikit-learn's `HistGradientBoostingRegressor`, and prints a message saying so. Just remember to call those results "GBM-fallback" in your report, not "LightGBM".
- `psutil` is only used to record your RAM in the hardware log (needed for the timing section of the report).

### Step 4: Download the dataset

The data comes from the Harvard Dataverse (the two links given in the assignment):

- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

Together they cover **1 November 2013 to 1 January 2014** (62 daily files).

1. Download all the daily `.txt` files from both pages.
2. Put **all of them** directly inside the `data/` folder (no sub-folders).
3. Do **not** rename them. The notebook finds each file's date from its name, which looks like `sms-call-internet-mi-2013-11-01.txt`.

Quick check: `data/` should contain **62** `.txt` files.

### Step 5: Launch the notebook

```bash
jupyter notebook
```

Your browser will open. Click `notebook.ipynb`. Make sure the kernel is **Python 3** (top right). If you used a virtual environment and the packages aren't found, run this once and then choose the "milan-forecast" kernel:

```bash
python -m ipykernel install --user --name milan-forecast
```

---

## 3. Run it

Use **Kernel → Restart & Run All**, or run the cells one by one from top to bottom. **Order matters**, since later cells depend on earlier ones.

Here's what each section of the notebook does:

| Section | What happens | Roughly how long |
|---|---|---|
| **0. Setup** | Imports, folder paths, evaluation dates | Seconds |
| **1. Hardware capture** | Saves your CPU / RAM / GPU info for the timing table | Seconds |
| **2. Data loading & continuity gate** | Loads the files in chunks with small data types, sums Internet traffic across country codes, prints memory used, checks for missing days/time steps | A few minutes |
| **3. Metrics** | Defines MAE, RMSE, MAPE, sMAPE | Instant |
| **4. Exploratory analysis** | Figures 1-5 (distribution, first two weeks, ACF/PACF + ADF test, decomposition, anomalies) | A minute or two |
| **5. Models** | Defines the SARIMA, LightGBM and LSTM classes | Instant |
| **6. Hyperparameter tuning** | Small grid search per model on the **validation week (9-15 Dec)** for the busiest area | **Longest part** (LSTM especially) |
| **7. Final experiments** | Retrains the best configs, then forecasts **16-22 Dec** for the 3 areas | Long |
| **8. Failure analysis** | Highlights the worst-error points of the weakest model | Seconds |

### The built-in safety check

Section 2 has a "continuity gate". If any daily files are missing, or if 16-22 December isn't fully covered, the notebook **stops with a clear message** listing what's missing. If you see that, go back to Step 4 and add the missing files.

---

## 4. Where the results go

The notebook creates an `outputs/` folder automatically:

```
outputs/
├── fig1_traffic_distribution.png
├── fig2_first_two_weeks.png
├── fig3_acf_pacf.png
├── fig4_decomposition.png
├── fig5_anomalies.png
├── forecast_<area>_<model>.png       <- the 9 required forecast plots
├── failure_analysis_<model>.png
└── logs/
    ├── hardware_info.json            <- hardware for the timing section
    ├── continuity_report.json        <- proof the data has no gaps
    ├── adf_test.json                 <- stationarity test result
    ├── tuning_log.csv                <- every hyperparameter trial + its metrics
    ├── final_metrics.csv             <- MAE / MAPE / sMAPE / RMSE per model & area
    └── timing_summary.csv            <- training & prediction times
```

These files are what you pull into the report.

---

## 5. Common problems

| Problem | Fix |
|---|---|
| `No files found in data` | The `data/` folder is missing or in the wrong place. It must sit next to the notebook. |
| The notebook stops at the continuity gate | Some daily files are missing. The message lists which dates. |
| `MemoryError` / the kernel crashes | Close other apps, and make sure you have 8 GB+ of free RAM. Restart the kernel and run again. |
| `ModuleNotFoundError: No module named ...` | Run `pip install <name>` inside your virtual environment, then restart the kernel. |
| LightGBM / OpenMP error | See the note in Step 3. The notebook falls back to scikit-learn automatically. |
| Runs are very slow | That's normal for the LSTM on CPU. To speed up a test run, shrink the grids in Section 6 (fewer configs) or lower `epochs`. |
| Numbers differ slightly from the report | Neural networks have some randomness. Small differences are expected. |

---

## 6. Good to know

- **Time range:** training uses all data before **16 Dec 2013**; testing is **16-22 Dec 2013**.
- **Validation:** tuning uses the week **9-15 Dec**, so the test week is never used to pick settings.
- **Areas evaluated:** the highest-traffic square (found automatically), **4159**, and **4556**.
- **Preprocessing:** every model works on `log1p`-transformed traffic. Predictions are converted back and clipped at zero.
- **Multi-step forecasts** for LightGBM and LSTM are recursive (each prediction feeds the next one).
- **Timings** in `timing_summary.csv` are averaged across the three areas, on your machine's CPU.

---

## 7. Data source and reference

G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Scientific Data*, vol. 2, art. 150055, 2015. https://doi.org/10.1038/sdata.2015.55
