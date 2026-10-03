"""
Shared loading and modelling code for the food-price analysis.

Imported by the notebooks and by the report builder, so every number comes from
one implementation. All paths resolve from the repository root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RAW, EXT = ROOT / 'data' / 'raw', ROOT / 'data' / 'external'

# Markets with long WFP records, mapped to the rainfall regions of the rainfall project.
# Karamoja markets are kept separate: their records start in late 2018.
CORE_MARKETS = {
    'Owino': 'Central',       # Kampala
    'Lira': 'Northern',
    'Gulu': 'Northern',
    'Mbarara': 'Western',
    'Iganga': 'Eastern',
    'Kapchorwa': 'Eastern',
    'Jinja': 'Eastern',
}
KARAMOJA_MARKETS = ['Kaabong', 'Kotido', 'Namalu', 'Makaratin', 'Karita', 'Moroto', 'Napak', 'Nabilatuk', 'Karenga']
STAPLES = ['Maize (white)', 'Maize flour', 'Sorghum', 'Beans']

# Season windows. Season A: March–May rains, harvest June–August, outcome = real prices
# September–December of the same year. Season B: October–December rains, harvest
# December–January, outcome = real prices February–May of the next year.
SEASONS = {
    'A': {'rain_months': [3, 4, 5], 'label': 'First season (Mar–May rains)', 'outcome_months': [9, 10, 11, 12], 'year_offset': 0},
    'B': {'rain_months': [10, 11, 12], 'label': 'Second season (Oct–Dec rains)', 'outcome_months': [2, 3, 4, 5], 'year_offset': 1},
}


# ---------------------------------------------------------------- loaders
def load_prices():
    d = pd.read_csv(RAW / 'wfp_food_prices_uga.csv', skiprows=[1])
    d['date'] = pd.to_datetime(d['date']).dt.to_period('M')
    d['price'] = pd.to_numeric(d['price'], errors='coerce')
    d['usdprice'] = pd.to_numeric(d['usdprice'], errors='coerce')
    return d.dropna(subset=['price'])


def load_cpi(kind='General'):
    c = pd.read_csv(RAW / 'fao_cpi_uganda.csv')
    c = c[c['item'].str.contains(kind)]
    idx = pd.PeriodIndex([pd.Period(year=y, month=m, freq='M') for y, m in zip(c['year'], c['month'])])
    return pd.Series(c['value'].values, index=idx).sort_index()


def load_world_maize():
    w = pd.read_csv(RAW / 'world_maize_price.csv')
    w.index = pd.to_datetime(w['observation_date']).dt.to_period('M')
    return w['PMAIZMTUSDM'].rename('world_maize')


def load_exchange_rate():
    # Implied UGX per USD from WFP's own conversions, averaged across all records each month
    d = load_prices()
    d = d[d['usdprice'] > 0]
    return (d['price'] / d['usdprice']).groupby(d['date']).median().rename('ugx_usd')


def load_rain():
    r = pd.read_csv(EXT / 'uganda_rainfall_by_region_1990_2025.csv')
    r['region'] = r['region'].str.replace(' Region region', '', regex=False)
    r[['month', 'year']] = r[['month', 'year']].astype(int)
    return r


def seasonal_rain_z(months):
    """Seasonal rainfall per region and year, standardised over 1990–2025."""
    r = load_rain()
    s = r[r['month'].isin(months)].groupby(['region', 'year'])['rainfall_mm'].sum()
    return s.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())


def load_dmi():
    d = pd.read_csv(EXT / 'dmi_monthly.csv')
    return d


def dmi_mean(months):
    d = load_dmi()
    return d[d['month'].isin(months)].groupby('year')['dmi'].mean()


# ---------------------------------------------------------------- price anomalies
def real_log_price(commodity, markets, pricetype='Retail'):
    """Monthly log real price (deflated by the general CPI), one column per market."""
    d = load_prices()
    k = d[(d['commodity'] == commodity) & (d['pricetype'] == pricetype) & d['market'].isin(markets)]
    wide = k.groupby(['date', 'market'])['price'].mean().unstack()
    cpi = load_cpi().reindex(wide.index)
    return np.log(wide.div(cpi, axis=0))


def price_anomaly(series):
    """Remove a linear trend and the average calendar-month pattern from one market's series."""
    s = series.dropna()
    if len(s) < 24:
        return s * np.nan
    t = np.array([(p - s.index[0]).n for p in s.index], float)
    detr = s - np.polyval(np.polyfit(t, s.values, 1), t)
    return detr - detr.groupby(detr.index.month).transform('mean')


def seasonal_profile(series):
    """Average real price by calendar month, relative to the series mean, in percent."""
    s = series.dropna()
    t = np.array([(p - s.index[0]).n for p in s.index], float)
    detr = s - np.polyval(np.polyfit(t, s.values, 1), t)
    prof = detr.groupby(detr.index.month).mean()
    return 100 * (np.exp(prof - prof.mean()) - 1)


# ---------------------------------------------------------------- season panel
def season_panel(commodity, markets=CORE_MARKETS, min_months=2):
    """One row per market, season and harvest year: mean price anomaly in the outcome window,
    with the season's regional and national rainfall (z-scores), the world maize price and the
    exchange rate over the same window, and the July–August IOD for season B."""
    lp = real_log_price(commodity, list(markets))
    world = np.log(load_world_maize())
    fx = np.log(load_exchange_rate())
    ja_iod = dmi_mean([7, 8])
    rows = []
    for key, cfg in SEASONS.items():
        rz = seasonal_rain_z(cfg['rain_months'])
        for mkt in lp.columns:
            a = price_anomaly(lp[mkt])
            for y in range(2006, 2026):
                oy = y + cfg['year_offset']
                win = [p for p in a.index if p.year == oy and p.month in cfg['outcome_months']]
                if len(win) < min_months:
                    continue
                reg = markets[mkt]
                rows.append({
                    'commodity': commodity, 'market': mkt, 'region': reg, 'season': key, 'year': y,
                    'price_anom': a.loc[win].mean(),
                    'rain_regional': rz.get((reg, y)), 'rain_national': rz.get(('Uganda (national)', y)),
                    'world': world.reindex(win).mean(), 'fx': fx.reindex(win).mean(),
                    'ja_iod': ja_iod.get(y) if key == 'B' else np.nan,
                })
    return pd.DataFrame(rows)


def fit(panel, x, controls=('world',), cluster='year'):
    """OLS of the price anomaly on x, with market fixed effects and controls; standard
    errors clustered by year, because a rainfall shock hits every market in the same year."""
    q = panel.dropna(subset=['price_anom', x] + list(controls))
    rhs = ' + '.join([x] + list(controls) + ['C(market)'])
    m = smf.ols(f'price_anom ~ {rhs}', q).fit(cov_type='cluster', cov_kwds={'groups': q[cluster]})
    return {'coef': m.params[x], 'se': m.bse[x], 'p': m.pvalues[x], 'n': int(m.nobs),
            'years': int(q['year'].nunique()), 'markets': int(q['market'].nunique()),
            'pct': 100 * (np.exp(m.params[x]) - 1)}
