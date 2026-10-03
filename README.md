# Rainfall Shocks and Food Prices in Uganda

How Uganda's rainy seasons, and the Indian Ocean conditions that drive them, show up in the prices of maize, sorghum and beans. The project links WFP market prices (2006–2026) to regional rainfall from the companion [Uganda rainfall analysis](https://github.com/TayeRuta/uganda-rainfall-analysis). It also audits whether FAOSTAT's national crop statistics can support this kind of analysis.

**Read the report: [Rainfall shocks and food prices in Uganda](https://tayeruta.github.io/uganda-food-prices/reports/food_prices_report.html)**

## Key findings

**Prices**

- **The seasonal cycle is the largest and most predictable price risk.** Real maize prices swing about 30% (20–43%) between harvest (October–February) and the lean season (May–July) every year. In Karamoja, sorghum, the staple, swings by up to 44%. Storing grain from harvest to the lean season earns about 25–30% in real terms on average, before storage losses.
- **A dry first season raises maize prices.** A March–May season one standard deviation drier than normal raises real maize prices by about 9–11% in September–December. The effect appears in both retail (2011–2025) and wholesale (2006–2021) data. *Medium confidence* (permutation p ≈ 0.03–0.08).
- **A dry second season raises sorghum prices** by about 8–9% the following February–May. *Suggestive* (one dataset).
- **The July–August Indian Ocean Dipole points to next year's lean-season prices.** Each +1 °C is followed by real maize prices about 24% lower the next February–May (about 10–17% in a strong positive-IOD year). *Suggestive*: permutation p = 0.02, but not replicated in the wholesale data.
- **Beans don't respond** to Ugandan rainfall in either season.
- **Karamoja's maize and sorghum prices move with the national market**, nearly as closely as other markets do. What sets Karamoja apart is the size of its seasonal swings and crisis spikes: real prices nearly doubled in mid-2022.

**Data quality**

- **FAOSTAT's Uganda crop statistics mostly can't detect weather shocks.**
  - A methodological break in 2008 moved crops by −88% to +110% in a single year.
  - Harvested area was effectively frozen in 2013–2015.
  - Implausible jumps occur around 2018–2019.
  - Of the main staples, only sorghum production tracks rainfall. Maize shows no relationship.

## Repository structure

```
.
├── data/
│   ├── raw/                   # downloaded inputs (see Data sources)
│   ├── external/              # rainfall and climate indices from the rainfall project
│   └── processed/             # result tables written by the notebooks
├── reports/
│   └── food_prices_report.html        # the write-up, published on GitHub Pages
├── notebooks/
│   ├── 01_faostat_data_quality.ipynb      # can national crop statistics detect weather shocks?
│   └── 02_rainfall_and_food_prices.ipynb  # seasonal cycles, rainfall shocks, IOD, Karamoja
├── scripts/
│   ├── gee/
│   │   └── uganda_cropland_ndvi_gee.js   # Earth Engine export: monthly crop and pasture greenness by region
│   ├── fetch_data.py          # downloads every input dataset
│   ├── prices.py              # shared loading and modelling code
│   ├── build_notebooks.py     # generates the notebooks from source
│   └── build_report_data.py   # injects the notebook's results into the report
└── requirements.txt
```

## Methods

- **Cleaning:** 10 of 6,783 staple prices removed as recording errors (more than double or under half the local 7-month median); all are listed in notebook 02.
- **Real prices:** WFP retail prices deflated by Uganda's general consumer price index, logged, with each market's linear trend and calendar-month pattern removed.
- **Season design:**
  - First season: March–May rains, with prices measured in September–December.
  - Second season: October–December rains, with prices measured the next February–May.
  - Each market is matched to its region's rainfall.
- **Models:** OLS with market fixed effects, controlling for the world maize price (and the exchange rate as a robustness check), with standard errors clustered by year.
- **Robustness:**
  - permutation tests on yearly averages;
  - leaving out one year at a time;
  - an independent wholesale price sample (2006–2021).
- **Market integration:** correlation of trend- and season-adjusted price anomalies between markets.

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py            # or: --rainfall-repo /path/to/uganda-rainfall-analysis
python scripts/build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
python scripts/build_report_data.py     # refresh the numbers in the report
```

## Data sources

| Dataset | Used for | Provider |
|---|---|---|
| [WFP food prices for Uganda](https://data.humdata.org/dataset/wfp-food-prices-for-uganda) | Market prices, 2006–2026 | World Food Programme, via HDX |
| [FAOSTAT crops and livestock](https://www.fao.org/faostat/en/#data/QCL) | Crop statistics audit | UN FAO |
| [FAOSTAT consumer price indices](https://www.fao.org/faostat/en/#data/CP) | Deflating prices | UN FAO (from national sources) |
| [Global price of maize](https://fred.stlouisfed.org/series/PMAIZMTUSDM) | World price control | IMF, via FRED |
| [CHIRPS rainfall and Dipole Mode Index](https://github.com/TayeRuta/uganda-rainfall-analysis) | Rainfall and IOD | UC Santa Barbara CHC; NOAA PSL |

Each dataset keeps its provider's terms of use; check them before reusing the raw files.

## In progress: crop condition from satellite

`scripts/gee/uganda_cropland_ndvi_gee.js` exports monthly MODIS vegetation indices (NDVI, EVI) over cropland and pasture for each region, 2000 to date. Run it in the [Earth Engine Code Editor](https://code.earthengine.google.com/), start the export from the Tasks tab, and save the CSV as `data/raw/uganda_vegetation_by_region_monthly.csv`. It will be used to test the full chain from rainfall to harvest to prices.

## Limitations

- **Short record:** 15–16 years of prices limit statistical certainty, so the permutation tests are the honest measure.
- **Retail, not farm-gate, prices:** retail prices in a few towns stand in for the prices farmers receive, which are lower and more volatile.
- **National deflator:** the price index is national, and regional inflation can differ.
- **Short Karamoja record:** Karamoja prices start in late 2018, too few seasons to estimate rainfall effects there.

## License

Code and analysis: [MIT](LICENSE).
