"""
Download every input dataset for the project and save Uganda's slice as tidy CSVs.

Usage (from anywhere):
    python scripts/fetch_data.py [--rainfall-repo PATH]

Outputs
  data/raw/wfp_food_prices_uga.csv     WFP market prices (HDX), monthly, 2006–present
  data/raw/wfp_markets_uga.csv         WFP market list with coordinates
  data/raw/faostat_crops_uganda.csv    FAOSTAT crop production, area, yield, with data flags
  data/raw/fao_cpi_uganda.csv          FAOSTAT consumer price indices (general and food), monthly
  data/raw/world_maize_price.csv       IMF global maize price (US$/tonne), monthly, via FRED
  data/external/                       rainfall and Dipole Mode Index from the rainfall project

The rainfall inputs come from github.com/TayeRuta/uganda-rainfall-analysis. If a local
copy is given with --rainfall-repo they are copied from it, otherwise downloaded from GitHub.
"""
import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, EXT = ROOT / 'data' / 'raw', ROOT / 'data' / 'external'

WFP_PRICES = ('https://data.humdata.org/dataset/883929b1-521e-4834-97f5-0ccc2df75b89/resource/'
              'e082d683-cad5-4dcd-bf54-db76ae254d33/download/wfp_food_prices_uga.csv')
WFP_MARKETS = ('https://data.humdata.org/dataset/883929b1-521e-4834-97f5-0ccc2df75b89/resource/'
               '4ef683cd-ccdd-4f96-9dc3-e25e6af88d25/download/wfp_markets_uga.csv')
FAO_CROPS = 'https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_Africa.zip'
FAO_CPI = 'https://bulks-faostat.fao.org/production/ConsumerPriceIndices_E_All_Data.zip'
WORLD_MAIZE = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=PMAIZMTUSDM'
RAIN_REPO = 'https://raw.githubusercontent.com/TayeRuta/uganda-rainfall-analysis/main/'
RAIN_FILES = ['data/raw/uganda_rainfall_by_region_1990_2025.csv', 'data/external/dmi_monthly.csv',
              'data/external/oni_seasonal.csv']


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def read_zip_csv(blob, name):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(name) as f:
            return pd.read_csv(f, encoding='latin-1')


def faostat_long(df, id_cols):
    years = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    vals = df.melt(id_vars=id_cols, value_vars=years, var_name='year', value_name='value')
    if all(f'{y}F' in df.columns for y in years):
        flags = df.melt(id_vars=id_cols, value_vars=[f'{y}F' for y in years], var_name='year', value_name='flag')
        flags['year'] = flags['year'].str[:-1]
        vals = vals.merge(flags, on=id_cols + ['year'])
    vals['year'] = vals['year'].str[1:].astype(int)
    return vals.dropna(subset=['value'])


def main(rainfall_repo):
    RAW.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)

    (RAW / 'wfp_food_prices_uga.csv').write_bytes(get(WFP_PRICES))
    (RAW / 'wfp_markets_uga.csv').write_bytes(get(WFP_MARKETS))
    print('WFP prices and markets saved')

    crops = read_zip_csv(get(FAO_CROPS), 'Production_Crops_Livestock_E_Africa.csv')
    crops = crops[crops['Area'] == 'Uganda']
    out = faostat_long(crops, ['Item', 'Element', 'Unit'])
    out.rename(columns={'Item': 'item', 'Element': 'element', 'Unit': 'unit'}).to_csv(RAW / 'faostat_crops_uganda.csv', index=False)
    print(f'FAOSTAT crops: {len(out):,} rows')

    cpi = read_zip_csv(get(FAO_CPI), 'ConsumerPriceIndices_E_All_Data_NOFLAG.csv')
    cpi = cpi[cpi['Area'] == 'Uganda']
    cpi = faostat_long(cpi, ['Item', 'Months'])
    cpi['month'] = pd.to_datetime(cpi['Months'], format='%B').dt.month
    cpi = cpi.rename(columns={'Item': 'item'})[['item', 'year', 'month', 'value']]
    cpi.to_csv(RAW / 'fao_cpi_uganda.csv', index=False)
    print(f'FAO CPI: {len(cpi):,} rows')

    (RAW / 'world_maize_price.csv').write_bytes(get(WORLD_MAIZE))
    print('World maize price saved')

    for f in RAIN_FILES:
        dest = EXT / Path(f).name
        if rainfall_repo:
            shutil.copy(Path(rainfall_repo) / f, dest)
        else:
            dest.write_bytes(get(RAIN_REPO + f))
    print('Rainfall and climate indices saved')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rainfall-repo', help='local copy of uganda-rainfall-analysis (optional)')
    main(ap.parse_args().rainfall_repo)
