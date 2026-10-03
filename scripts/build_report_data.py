"""
Collect every number shown in reports/food_prices_report.html and inject it into the page.

Model results are read from the tables the notebook exports to data/processed/, so the report
always matches the notebook. Chart series are computed here with the shared code in prices.py.

Usage (from anywhere, after running notebooks/02):
    python scripts/build_report_data.py
"""
import json
import re
import numpy as np
import pandas as pd
from scipy import stats
import prices as P
from prices import ROOT, CORE_MARKETS, KARAMOJA_MARKETS

OUT = ROOT / 'data' / 'processed'
REPORT = ROOT / 'reports' / 'food_prices_report.html'
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def r4(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


def profiles():
    t = pd.read_csv(OUT / 'seasonal_price_profiles.csv', index_col=[0, 1])
    out = {}
    for com in ['Maize (white)', 'Sorghum', 'Beans']:
        sub = t.loc[com]
        out[com] = {m: {'p': [r4(v) for v in sub.loc[m, MON]], 'swing': r4(sub.loc[m, 'swing_%']),
                        'karamoja': m in KARAMOJA_MARKETS} for m in sub.index}
    return out


def effects():
    t = pd.read_csv(OUT / 'rainfall_price_effects.csv')
    t = t[t['controls'] == 'world']
    out = []
    for _, r in t.iterrows():
        # Report the effect of a season one standard deviation DRIER than normal (sign flipped)
        lo, hi = -r['coef'] - 1.96 * r['se'], -r['coef'] + 1.96 * r['se']
        out.append({'commodity': r['commodity'], 'season': r['season'], 'rainfall': r['rainfall'],
                    'pct': r4(100 * (np.exp(-r['coef']) - 1)), 'lo': r4(100 * (np.exp(lo) - 1)),
                    'hi': r4(100 * (np.exp(hi) - 1)), 'p': r4(r['p']), 'years': int(r['years'])})
    return out


def iod_scatter():
    pan = P.season_panel('Maize (white)')
    q = pan[pan['season'] == 'B'].dropna(subset=['ja_iod', 'price_anom'])
    y = q.groupby('year').agg(price=('price_anom', 'mean'), iod=('ja_iod', 'first'))
    return [{'year': int(k), 'iod': r4(v['iod']), 'pct': r4(100 * (np.exp(v['price']) - 1))} for k, v in y.iterrows()]


def karamoja():
    series = {}
    for com in ['Maize (white)', 'Sorghum']:
        lp = P.real_log_price(com, KARAMOJA_MARKETS).loc['2018-10':]
        k = np.exp(lp.mean(axis=1))
        k = 100 * (k / k.mean() - 1)
        series[com] = [{'d': str(p), 'v': r4(v)} for p, v in k.dropna().items()]
    rain = P.seasonal_rain_z([4, 5, 6, 7, 8, 9]).xs('Karamoja').loc[2018:2025]
    integ = pd.read_csv(OUT / 'karamoja_market_integration.csv', index_col=0)
    return {'series': series, 'rain': {int(y): r4(v) for y, v in rain.items()},
            'integration': {c: {k: r4(v) for k, v in row.items()} for c, row in integ.iterrows()}}


def faostat():
    crops = ['Maize (corn)', 'Beans, dry', 'Sorghum', 'Cassava, fresh', 'Plantains and cooking bananas']
    staples = ['Maize (corn)', 'Beans, dry', 'Sorghum', 'Millet', 'Cassava, fresh', 'Sweet potatoes',
               'Plantains and cooking bananas', 'Potatoes', 'Soya beans']
    f = pd.read_csv(P.RAW / 'faostat_crops_uganda.csv')
    f = f[(f['year'] >= 1990) & f['item'].isin(staples)]
    area = f[f['element'] == 'Area harvested'].pivot(index='year', columns='item', values='value')
    prod = f[f['element'] == 'Production'].pivot(index='year', columns='item', values='value')
    change08 = (100 * prod[staples].pct_change()).loc[2008]
    mam = P.seasonal_rain_z([3, 4, 5]).xs('Uganda (national)')
    ond = P.seasonal_rain_z([10, 11, 12]).xs('Uganda (national)')
    rain_year = mam + ond.shift(1)
    yrs = [y for y in range(1991, 2025) if y not in range(2008, 2016)]
    resp = {}
    for c in ['Maize (corn)', 'Sorghum', 'Beans, dry', 'Cassava, fresh', 'Millet', 'Sweet potatoes']:
        x, y = rain_year.reindex(yrs), np.log(prod[c]).diff().reindex(yrs)
        ok = x.notna() & y.notna()
        r, p = stats.pearsonr(x[ok], y[ok])
        resp[c] = {'r': r4(r), 'p': r4(p)}
    return {'area': {c: [{'y': int(y), 'v': r4(v / 1e3)} for y, v in area[c].dropna().items()] for c in crops},
            'change_2008': {c: r4(v) for c, v in change08.items()}, 'rain_response': resp}


def main():
    data = {
        'profiles': profiles(),
        'effects': effects(),
        'robustness': pd.read_csv(OUT / 'rainfall_price_robustness.csv').round(4).to_dict('records'),
        'wholesale': pd.read_csv(OUT / 'wholesale_replication.csv').round(4).to_dict('records'),
        'iod': pd.read_csv(OUT / 'iod_price_effects.csv').round(4).to_dict('records'),
        'iod_scatter': iod_scatter(),
        'karamoja': karamoja(),
        'faostat': faostat(),
    }
    (OUT / 'report_data.json').write_text(json.dumps(data, indent=1))
    s = REPORT.read_text()
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    s, n = re.subn(r'(<script id="report-data" type="application/json">)(.*?)(</script>)',
                   lambda m: m.group(1) + payload + m.group(3), s, flags=re.S)
    if n != 1:
        raise SystemExit(f'expected one report-data block, found {n}')
    REPORT.write_text(s)
    print('updated', REPORT.relative_to(ROOT))


if __name__ == '__main__':
    main()
