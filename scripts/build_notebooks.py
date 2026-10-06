"""
Generate the analysis notebooks from source, so their structure stays reviewable in git.

Usage (from anywhere):
    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
KERNEL = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python'}}

STYLE = r"""
import sys, warnings
sys.path.insert(0, '../scripts')
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# House style, shared with the rainfall project
WET, DRY, ACCENT, INK, MUTED, GRID = '#2b6c9e', '#c4572e', '#16825a', '#132019', '#7a877f', '#e6e9e4'
plt.rcParams.update({
    'figure.dpi': 110, 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c9cfc8', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': MUTED,
    'legend.frameon': False,
})
pd.set_option('display.width', 200)
"""


def notebook(cells, path):
    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_markdown_cell(c[1].strip()) if c[0] == 'md' else nbf.v4.new_code_cell(c[1].strip())
                   for c in cells]
    nb['metadata'] = KERNEL
    nbf.write(nb, path)
    print('wrote', path.relative_to(ROOT), len(cells), 'cells')


# ======================================================================= 01 FAOSTAT audit
NB1 = [
('md', r"""
# Can FAOSTAT crop statistics detect weather shocks in Uganda?
**Data:** FAOSTAT crop production, area harvested and yield for Uganda, 1990–2024 (bulk release of December 2025), with FAO's data-quality flags
**Purpose:** before linking rainfall to harvests, check whether the national crop statistics actually record year-to-year changes in harvests, or mainly reflect projections and changes in method.

**Why this matters.** FAOSTAT is the default source for country crop data, and is widely used to estimate how weather affects yields. If the series are projected forward between agricultural censuses, a rainfall–yield regression measures the statistical method rather than the harvest.

Each finding is written up in a markdown cell directly after the code and output that produced it.
"""),
('md', '## Setup'),
('code', STYLE + r"""
CROPS = ['Maize (corn)', 'Beans, dry', 'Sorghum', 'Millet', 'Cassava, fresh', 'Sweet potatoes',
         'Plantains and cooking bananas', 'Groundnuts, excluding shelled', 'Rice', 'Potatoes',
         'Soya beans', 'Sesame seed', 'Coffee, green']
fao = pd.read_csv('../data/raw/faostat_crops_uganda.csv')
fao = fao[(fao['year'] >= 1990) & fao['item'].isin(CROPS)]
wide = {el: fao[fao['element'] == el].pivot(index='year', columns='item', values='value')[CROPS]
        for el in ['Production', 'Area harvested', 'Yield']}
print('Years:', fao['year'].min(), '–', fao['year'].max(), '| crops:', len(CROPS))
"""),
('md', '---\n## 1. What the flags say'),
('code', r"""
flags = (fao[fao['element'] == 'Production'].groupby('item')['flag']
         .value_counts(normalize=True).unstack(fill_value=0).reindex(CROPS))
flag_names = {'A': 'Official', 'E': 'Estimated', 'I': 'Imputed', 'X': 'External'}
(100 * flags.rename(columns=flag_names)).round(0)
"""),
('md', r"""
**Finding:** taken at face value, the data look reliable. For the main food crops, over 90% of production values are flagged **official** (reported by the government). The flags record who supplied a number, though, not how it was produced. Sections 2–4 test how much real variation the official numbers contain.
"""),
('md', '---\n## 2. The signature of projected data'),
('code', r"""
growth = 100 * wide['Area harvested'].pct_change()
near_flat = (growth.abs() < 0.5)
same_yield = wide['Yield'].diff().abs() < 1e-9
periods = {'1991–2007': (1991, 2007), '2013–2015': (2013, 2015), '2016–2024': (2016, 2024)}
tab = pd.DataFrame({f'% of years with area change < 0.5%, {k}': 100 * near_flat.loc[a:b].mean() for k, (a, b) in periods.items()})
tab['median area change 2013–2015 (%)'] = growth.loc[2013:2015].abs().median()
tab.round(1)
"""),
('code', r"""
show = ['Maize (corn)', 'Beans, dry', 'Sorghum', 'Cassava, fresh', 'Plantains and cooking bananas']
fig, axes = plt.subplots(1, len(show), figsize=(13, 3.2), sharex=True)
for ax, c in zip(axes, show):
    s = wide['Area harvested'][c] / 1e3
    ax.plot(s.index, s.values, color=WET, lw=2, marker='o', ms=2.5)
    ax.axvspan(2012.5, 2015.5, color=GRID, alpha=.7, lw=0)
    ax.set_title(c.split(',')[0].split(' (')[0], fontsize=10)
axes[0].set_ylabel('Area harvested (000 ha)')
fig.suptitle('Area harvested barely moves in 2013–2015 (shaded), after a large break in 2008', x=0.01, ha='left', fontweight='bold')
plt.tight_layout(); plt.show()
"""),
('md', r"""
**Finding:** in **2013–2015**, the area harvested for beans, sorghum, millet, cassava and sweet potatoes changed by less than half a percent in every one of the three years (median 0.0–0.3%), and maize and bananas by under 1%. Before 2008, a change this small happened in fewer than one year in five for any of these crops. Real harvested area swings with rainfall, prices and insecurity; series this smooth are projections, most likely from the 2008/09 Uganda Census of Agriculture baseline rolled forward. Those years carry essentially no information about weather.
"""),
('md', '---\n## 3. Breaks from changes in method'),
('code', r"""
big = growth.loc[1991:2024].abs()
breaks = (wide['Production'].pct_change() * 100).loc[[2008, 2018, 2019]].T.round(0)
breaks.columns = [f'Production change {y} (%)' for y in breaks.columns]
breaks
"""),
('md', r"""
**Finding:** 2008 is a break, not a harvest. In one year bananas fell 54%, millet 62% and cassava 42%, while maize rose 83% and beans 110%. No weather event moves crops in opposite directions by these amounts: 2008 is when the new census baseline replaced the old series. Smaller jumps around 2018–2019 (soya beans +284%, beans −54% then +80%) coincide with the introduction of UBOS's Annual Agricultural Survey.

Some long-run trends are also implausible. Cassava's reported yield falls from about 13.5 t/ha in the early 2000s to about 2 t/ha in 2024, an 85% collapse that agronomic evidence does not support.
"""),
('md', '---\n## 4. Do the statistics respond to rainfall at all?'),
('code', r"""
from prices import seasonal_rain_z
mam = seasonal_rain_z([3, 4, 5]).xs('Uganda (national)')
ond = seasonal_rain_z([10, 11, 12]).xs('Uganda (national)')
rain_year = (mam + ond.shift(1)).rename('rain')   # first season this year + second season last year

log_change = np.log(wide['Production']).diff()
rows = []
for c in ['Maize (corn)', 'Beans, dry', 'Sorghum', 'Millet', 'Cassava, fresh', 'Sweet potatoes']:
    for label, yrs in [('All years 1991–2024, excl. 2008', [y for y in range(1991, 2025) if y != 2008]),
                       ('Measured-looking years only', [y for y in range(1991, 2025) if y not in range(2008, 2016)])]:
        x, y = rain_year.reindex(yrs), log_change[c].reindex(yrs)
        ok = x.notna() & y.notna()
        r, p = stats.pearsonr(x[ok], y[ok])
        rows.append({'crop': c, 'sample': label, 'r': r, 'p': p, 'n': int(ok.sum())})
resp = pd.DataFrame(rows).pivot(index='crop', columns='sample', values=['r', 'p'])
resp.round(2)
"""),
('md', r"""
**Finding:** the response is patchy. **Sorghum** production does track rainfall (r = 0.37, and 0.54 when the projected years are left out, both significant), which fits a drought-sensitive crop grown mainly in the drier north and northeast. Beans and cassava show weak positive links that are not significant. **Maize, Uganda's main staple and most traded grain, shows no relationship at all** (r = 0.13), and millet and sweet potatoes none either.

**Conclusion for this project:** with a structural break in 2008, projected values in 2013–2015 and implausible jumps around 2018–2019, FAOSTAT's Uganda series can't support a reliable estimate of how rainfall affects harvests. Only sorghum carries a usable weather signal. Published analyses that use these series uncritically should be read with caution. This project instead uses **market prices**, which are collected every month and respond to real supply shocks (notebook 02). A second independent check, satellite vegetation indices over cropland, is a natural next step.
"""),
]

# ======================================================================= 02 prices and rainfall
NB2 = [
('md', r"""
# Rainfall shocks and food prices in Uganda
**Prices:** WFP monthly retail market prices (HDX), 2011–2026 for seven core markets and late 2018–2026 for Karamoja; WFP wholesale maize, 2006–2022
**Deflator:** FAO consumer price index for Uganda (general, 2015 = 100)
**Controls:** world maize price (IMF via FRED) and the implied shilling–dollar exchange rate
**Rainfall and climate:** CHIRPS regional rainfall and the Indian Ocean Dipole index from the rainfall project

**Questions**
1. How large is the seasonal price cycle, and what does it mean for storing grain?
2. Does a poor rainy season raise real food prices in the months that follow?
3. Does the Indian Ocean Dipole (IOD), known before the second season, predict next year's lean-season prices?
4. How does Karamoja's market behave compared with the rest of the country?

**Design.** Prices are converted to real terms with the CPI, logged, and each market's linear trend and average calendar-month pattern are removed, leaving the unexpected part of the price. For each season, the outcome is that anomaly averaged over the months after harvest:

| Season | Rains | Harvest | Price window |
|---|---|---|---|
| First (A) | March–May | June–August | September–December, same year |
| Second (B) | October–December | December–January | February–May, next year |

Each market is matched to its region's rainfall (Kampala → Central, Lira and Gulu → Northern, Mbarara → Western, Iganga, Kapchorwa and Jinja → Eastern). Regressions include market fixed effects and the world maize price, with standard errors clustered by year, because a rainfall shock hits every market in the same year.

Each finding is written up in a markdown cell directly after the code and output that produced it.
"""),
('md', '## Setup'),
('code', STYLE + r"""
import statsmodels.formula.api as smf
import prices as P
from prices import (CORE_MARKETS, KARAMOJA_MARKETS, real_log_price, seasonal_profile, season_panel, fit,
                    seasonal_rain_z, load_cpi, load_world_maize)
rng = np.random.default_rng(42)
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

raw = P.load_prices()
core = raw[raw['market'].isin(CORE_MARKETS) & (raw['pricetype'] == 'Retail')]
cov = core[core['commodity'].isin(P.STAPLES)].groupby(['market', 'commodity'])['date'].agg(['min', 'max', 'count'])
cov.unstack('commodity')['count']
"""),
('md', r"""
**Check:** each core market has 95–200 monthly observations per staple, mostly from April 2011 to late 2025. Jinja has the most gaps (about 45% of months missing), Kapchorwa about 25%; the rest are over 85% complete. The consumer price index runs to March 2026, so later prices are dropped when converting to real terms.
"""),

('code', r"""
# Recording errors: prices more than double or under half the market's centred 7-month median
flagged = P.load_prices(clean=False)
flagged = flagged[flagged['outlier'] & flagged['commodity'].isin(P.STAPLES)
                  & flagged['market'].isin(list(CORE_MARKETS) + KARAMOJA_MARKETS)]
print(f'Removed {len(flagged)} of {len(raw[raw["commodity"].isin(P.STAPLES) & raw["market"].isin(list(CORE_MARKETS) + KARAMOJA_MARKETS)]) + len(flagged):,} staple prices as likely recording errors')
flagged[['date', 'market', 'commodity', 'pricetype', 'price']]
"""),
('md', r"""
**Cleaning:** a handful of prices are clearly mis-recorded; the clearest is Gulu maize in June 2021, entered as 2 shillings a kilo against a normal 1,000–2,000. Removing these matters: that single value alone distorted Gulu's seasonal pattern by 25 points. The rule may occasionally remove a genuine spike (Karita sorghum in April 2022 falls in the food crisis), so all removals are listed above.
"""),
('md', '---\n## 1. The seasonal price cycle'),
('code', r"""
rows = {}
for com in ['Maize (white)', 'Sorghum', 'Beans']:
    lp = real_log_price(com, list(CORE_MARKETS) + KARAMOJA_MARKETS)
    for m in lp.columns:
        if lp[m].notna().sum() >= 48:
            rows[(com, m)] = seasonal_profile(lp[m])
prof = pd.DataFrame(rows).T
prof.columns = MON
prof['cheapest'] = prof[MON].idxmin(axis=1)
prof['dearest'] = prof[MON].idxmax(axis=1)
prof['swing_%'] = prof[MON].max(axis=1) - prof[MON].min(axis=1)
prof.round(0)
"""),
('code', r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, com in zip(axes, ['Maize (white)', 'Sorghum', 'Beans']):
    sub = prof.loc[com, MON]
    kar = sub.loc[[m for m in sub.index if m in KARAMOJA_MARKETS]]
    oth = sub.loc[[m for m in sub.index if m not in KARAMOJA_MARKETS]]
    for _, r in oth.iterrows():
        ax.plot(range(12), r.values, color=WET, alpha=.25, lw=1)
    for _, r in kar.iterrows():
        ax.plot(range(12), r.values, color=DRY, alpha=.25, lw=1)
    ax.plot(range(12), oth.mean().values, color=WET, lw=2.5, label='Other markets (average)')
    ax.plot(range(12), kar.mean().values, color=DRY, lw=2.5, label='Karamoja markets (average)')
    ax.axhline(0, color='#c9cfc8', lw=1)
    ax.set_xticks(range(12)); ax.set_xticklabels([m[0] for m in MON])
    ax.set_title(com.split(' (')[0])
axes[0].set_ylabel('Real price vs annual average (%)')
axes[0].legend(loc='upper left', fontsize=8.5)
plt.tight_layout(); plt.show()
"""),
('md', r"""
**Finding:** real maize prices follow a strong annual cycle: cheapest at harvest (October–February) and dearest in the lean months before the first harvest (May–July). The swing from cheapest to dearest month averages about **30%** across markets (20–43%). Beans follow a similar pattern with a May peak.

The cycle is far larger in **Karamoja for sorghum**, its staple: 28–44% there, against 8–23% elsewhere. Lean-season sorghum in Karamoja (June–August) is the costliest food of the year exactly when household stocks are lowest.

**What it means for farmers.** Selling maize at harvest and buying back in the lean season costs a household about a third of the grain's value. Storing from harvest to the peak earns about 25–30% in real terms on average. That return has to cover storage losses, often 10–20% in traditional storage, and the cost of cash tied up. This is the economic case for hermetic storage bags and warehouse-receipt systems, which cut losses and let farmers borrow against stored grain.
"""),

('md', '---\n## 2. Do poor rainy seasons raise prices?'),
('code', r"""
panels = {com: season_panel(com) for com in P.STAPLES}
rows = []
for com, pan in panels.items():
    for s in ['A', 'B']:
        q = pan[pan['season'] == s]
        for x in ['rain_regional', 'rain_national']:
            for ctrl in [('world',), ('world', 'fx')]:
                r = fit(q, x, ctrl)
                rows.append({'commodity': com, 'season': s, 'rainfall': x, 'controls': ' + '.join(ctrl), **r})
rain_res = pd.DataFrame(rows)
rain_res['dry_pct'] = 100 * (np.exp(-rain_res['coef']) - 1)   # effect of a season one SD DRIER than normal
rain_res.pivot_table(index=['commodity', 'season', 'rainfall'], columns='controls', values=['dry_pct', 'p']).round(3)
"""),
('md', r"""
**Reading the table:** `dry_pct` is the change in real prices over the following months when the season's rainfall is one standard deviation **below** normal (roughly a 1-in-6 dry season). Positive values mean dry seasons raise prices.

**Finding:** after the **first season**, maize prices respond clearly: a season one standard deviation drier than normal raises real maize prices by about **8–9%** in September–December (p = 0.002–0.010), with maize flour moving about 5%. After the **second season**, sorghum responds most (about 8–9%, p ≤ 0.009) and maize somewhat (about 7%, p ≈ 0.09–0.10). Controlling for the exchange rate changes nothing. **Beans do not respond** in either season, plausibly because beans are traded more widely across East Africa and are grown in both seasons in most areas, so one poor season is buffered.

Regional and national rainfall give almost identical answers: Uganda's markets are well connected, so a national shortfall moves prices everywhere.
"""),
('md', '### 2b. How much should we trust this?'),
('code', r"""
def permutation_p(q, x, B=2000):
    q = q.dropna(subset=['price_anom', x, 'world'])
    yearly = q.groupby('year').agg(p=('price_anom', 'mean'), x=(x, 'mean'), w=('world', 'mean'))
    obs = smf.ols('p ~ x + w', yearly).fit()
    null = [smf.ols('p ~ x + w', yearly.assign(x=rng.permutation(yearly['x'].values))).fit().params['x'] for _ in range(B)]
    return obs, (np.sum(np.abs(null) >= abs(obs.params['x'])) + 1) / (B + 1), len(yearly)

checks = [('Maize (white)', 'A', 'rain_national'), ('Maize (white)', 'B', 'rain_national'),
          ('Sorghum', 'B', 'rain_regional'), ('Maize flour', 'A', 'rain_national')]
rows = []
for com, s, x in checks:
    q = panels[com][panels[com]['season'] == s]
    base = fit(q, x)
    loo = [fit(q[q['year'] != y], x)['pct'] for y in sorted(q.dropna(subset=[x])['year'].unique())]
    obs, pp, n = permutation_p(q, x)
    rows.append({'commodity': com, 'season': s, 'rainfall': x, 'panel_pct': base['pct'], 'clustered_p': base['p'],
                 'yearly_pct': 100 * (np.exp(obs.params['x']) - 1), 'permutation_p': pp, 'years': n,
                 'leave_one_year_out_min': min(loo), 'leave_one_year_out_max': max(loo)})
robust = pd.DataFrame(rows)
dry = lambda pct: 100 * (1 / (1 + pct / 100) - 1)   # convert a wetter-season effect to a drier-season one
for c in ['panel_pct', 'yearly_pct', 'leave_one_year_out_min', 'leave_one_year_out_max']:
    robust['dry_' + c] = dry(robust[c])
robust.round(3)
"""),
('code', r"""
# An independent sample: WFP wholesale maize in Kampala, Lira and Busia, 2006–2022
WHOLESALE = {'Owino': 'Central', 'Lira': 'Northern', 'Busia': 'Eastern'}
_retail = P.real_log_price
P.real_log_price = lambda c, m, pricetype='Retail': _retail(c, m, 'Wholesale')
wholesale = P.season_panel('Maize', WHOLESALE)
P.real_log_price = _retail

rows = []
for s, x in [('A', 'rain_national'), ('A', 'rain_regional'), ('B', 'rain_national')]:
    q = wholesale[wholesale['season'] == s]
    base = fit(q, x)
    obs, pp, n = permutation_p(q, x)
    rows.append({'season': s, 'rainfall': x, 'years': f"{q['year'].min()}–{q['year'].max()}", 'panel_pct': base['pct'],
                 'clustered_p': base['p'], 'yearly_pct': 100 * (np.exp(obs.params['x']) - 1), 'permutation_p': pp})
wholesale_res = pd.DataFrame(rows)
wholesale_res['dry_panel_pct'] = 100 * (1 / (1 + wholesale_res['panel_pct'] / 100) - 1)
wholesale_res.round(3)
"""),
('md', r"""
**Finding:** the size and direction of each effect are very stable. No single year drives them (every leave-one-year-out estimate keeps the same sign and a similar size), and collapsing the markets to one national price per year gives the same estimates. The **certainty is lower than the clustered p-values suggest**, though. With only 15–16 years, a permutation test, which compares the result against thousands of reshuffled rainfall histories, gives p ≈ 0.08–0.18 for the retail results. The seven markets move together, so they add little independent information beyond the number of years.

The **first-season maize result replicates** in an independent sample: WFP's wholesale maize prices for 2006–2021, which include the 2008–09 drought, show an 11% rise per standard deviation of drier March–May rain (permutation p = 0.03–0.04). The second-season maize result does not replicate in the wholesale data.

**Confidence:** *medium* for the first-season maize effect (two datasets, consistent size, permutation p ≈ 0.03–0.08); *suggestive* for the second-season sorghum and maize effects (one dataset).
"""),

('md', '---\n## 3. Can the Indian Ocean Dipole warn of next year\'s prices?'),
('code', r"""
rows = []
for com in ['Maize (white)', 'Maize flour', 'Sorghum', 'Beans']:
    q = panels[com][panels[com]['season'] == 'B']
    r = fit(q, 'ja_iod')
    obs, pp, n = permutation_p(q, 'ja_iod')
    loo = [fit(q[q['year'] != y], 'ja_iod')['pct'] for y in sorted(q['year'].unique())]
    rows.append({'commodity': com, 'pct_per_degC': r['pct'], 'clustered_p': r['p'], 'permutation_p': pp,
                 'loo_min': min(loo), 'loo_max': max(loo), 'years': n})
iod_res = pd.DataFrame(rows)
iod_res.round(3)
"""),
('code', r"""
q = panels['Maize (white)'][panels['Maize (white)']['season'] == 'B'].dropna(subset=['ja_iod'])
yearly = q.groupby('year').agg(price=('price_anom', 'mean'), iod=('ja_iod', 'first'))
yearly['price_pct'] = 100 * (np.exp(yearly['price']) - 1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color='#c9cfc8', lw=1); ax.axvline(0, color='#c9cfc8', lw=1)
ax.scatter(yearly['iod'], yearly['price_pct'], s=46, color=WET, edgecolor='white', zorder=3)
for i, (y, r) in enumerate(yearly.sort_values('iod').iterrows()):
    ax.annotate(f'{y}→{(y + 1) % 100:02d}', (r['iod'], r['price_pct']), xytext=(5, 4 if i % 2 else -10), textcoords='offset points', fontsize=8, color=INK)
fit_ = stats.linregress(yearly['iod'], yearly['price_pct'])
xs = np.linspace(yearly['iod'].min(), yearly['iod'].max(), 10)
ax.plot(xs, fit_.intercept + fit_.slope * xs, color=INK, lw=1.5)
ax.set_xlabel('IOD, July–August (°C)'); ax.set_ylabel('Real maize price, Feb–May next year (% vs normal)')
ax.set_title('A positive IOD in July–August is followed by cheaper maize the next year')
plt.tight_layout(); plt.show()
"""),
('md', r"""
**Finding:** the July–August IOD, known by early September, points to next year's lean-season prices. Each +1 °C of the July–August IOD is followed by real maize prices about **24% lower** the following February–May, and sorghum about 19% lower. Strong positive IOD seasons reach about +0.4 to +0.7 °C, so the practical range is about 10–17%. This fits the rainfall analysis: a positive IOD brings a wetter October–December season, hence a bigger harvest.

**Caution:** this rests on 16 years of retail prices. The permutation p-value is 0.02 for maize and 0.08–0.09 for sorghum and maize flour, and the effect does **not** replicate in the 2006–2021 wholesale maize data. Treat it as a promising early-warning indicator to monitor, not an established forecast. Beans show no IOD effect.
"""),

('md', '---\n## 4. Karamoja'),
('code', r"""
from prices import price_anomaly
def mean_pair_corr(a_cols, b_cols, df):
    vals = []
    for a in a_cols:
        for b in b_cols:
            if a == b or (a_cols is b_cols and a > b):
                continue
            x = df[[a, b]].dropna()
            if len(x) > 24:
                vals.append(x.corr().iloc[0, 1])
    return np.mean(vals)

rows = []
for com in ['Maize (white)', 'Sorghum', 'Beans']:
    lp = real_log_price(com, ['Owino', 'Lira'] + list(CORE_MARKETS) + KARAMOJA_MARKETS).loc['2018-10':'2026-01']
    # Price levels: Karamoja average against Kampala and Lira
    kar_cols = [m for m in KARAMOJA_MARKETS if m in lp]
    kar = lp[kar_cols].mean(axis=1)
    gaps = {ref: 100 * (np.exp((kar - lp[ref]).dropna()) - 1).mean() for ref in ['Owino', 'Lira']}
    # Co-movement: correlation of trend- and season-adjusted price anomalies, same period for every pair
    an = pd.DataFrame({m: price_anomaly(lp[m]) for m in lp.columns if lp[m].notna().sum() >= 48})
    k = [m for m in an if m in KARAMOJA_MARKETS]
    c = [m for m in an if m in CORE_MARKETS]
    rows.append({'commodity': com, 'gap_vs_Kampala_%': gaps['Owino'], 'gap_vs_Lira_%': gaps['Lira'],
                 'co-movement core-core': mean_pair_corr(c, c, an), 'co-movement Karamoja-core': mean_pair_corr(k, c, an),
                 'co-movement Karamoja-Karamoja': mean_pair_corr(k, k, an)})
integration = pd.DataFrame(rows).set_index('commodity')
integration.round(2)
"""),
('code', r"""
import matplotlib.dates as mdates
kar_rain = seasonal_rain_z([4, 5, 6, 7, 8, 9]).xs('Karamoja')   # Karamoja's single April–September season
fig, ax1 = plt.subplots(figsize=(11, 4.4))
for com, c in [('Sorghum', DRY), ('Maize (white)', WET)]:
    lp = real_log_price(com, KARAMOJA_MARKETS).loc['2018-10':]
    k = 100 * (np.exp(lp.mean(axis=1)) / np.exp(lp.mean(axis=1)).mean() - 1)
    ax1.plot(k.index.to_timestamp(), k.values, color=c, lw=2, label=f'{com.split(" (")[0]}, Karamoja average')
ax1.axhline(0, color='#c9cfc8', lw=1)
ax1.set_ylabel('Real price vs 2018–26 average (%)')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_title('Karamoja real staple prices, with each year\'s April–September rainfall')
for y in range(2018, 2026):
    z = kar_rain.get(y)
    ax1.annotate(f'{y} rains\n{z:+.1f} SD', (mdates.date2num(pd.Timestamp(f'{y}-07-01')), 0), xycoords=ax1.get_xaxis_transform(),
                 xytext=(0, -30), textcoords='offset points', ha='center', va='top', fontsize=8,
                 color=DRY if z <= -0.5 else WET if z >= 0.5 else MUTED)
plt.tight_layout(); plt.subplots_adjust(bottom=0.24); plt.show()
kar_rain.loc[2018:2025].round(2)
"""),
('md', r"""
**Finding:** Karamoja's **maize and sorghum markets are nearly as connected to the rest of the country as the other markets are to each other.** After removing trends and seasonal patterns, Karamoja's maize prices move with the core markets nearly as closely (correlation 0.67) as the core markets move with one another (0.72); for sorghum the link is, if anything, stronger (0.39 against 0.28). Only **beans** are noticeably less connected (0.36 against 0.66).

The price *levels* show Karamoja's position in the grain trade: maize there costs about 20% less than in Kampala but about 22% **more** than in Lira, the nearby surplus-producing area, consistent with a deficit region buying in grain and paying the transport cost.

What sets Karamoja apart is therefore not isolation but **exposure**: the largest seasonal swing in its staple (sorghum, up to 44%), and the sharpest crisis spikes. Local harvests therefore matter more for local prices, and a shortfall is not quickly offset by grain flowing in.

The **2022 food crisis** is visible: real maize and sorghum prices averaged over 40% above their 2018–26 norm for the year and peaked at nearly double it in June–July 2022. The timing matters. Prices started climbing in early 2022 and peaked in the lean season *before* that year's harvest, so the 2022 rains can't explain the spike. It coincides with the global grain price shock that followed Russia's invasion of Ukraine in February 2022, with insecurity in the region, and with national shortages after poor 2021 seasons elsewhere in Uganda: satellite crop greenness (notebook 03) shows Karamoja's own crops near normal in 2021–22 while the rest of the country had some of its worst seasons on record. The below-average 2022 and 2023 seasons then kept prices high into 2023, and prices fell back only after the better 2024 season. Sorghum in Karamoja is usually much cheaper than in Kampala (about 50% less), but its lean-season spikes are the sharpest anywhere in the data.

With only seven seasons of Karamoja prices, the rainfall–price link there can't be estimated reliably yet; this section is descriptive.
"""),

('md', '---\n## Export'),
('code', r"""
out = Path('../data/processed') if 'Path' in dir() else None
from pathlib import Path
out = Path('../data/processed'); out.mkdir(exist_ok=True)
prof.round(2).to_csv(out / 'seasonal_price_profiles.csv')
rain_res.round(4).to_csv(out / 'rainfall_price_effects.csv', index=False)
robust.round(4).to_csv(out / 'rainfall_price_robustness.csv', index=False)
iod_res.round(4).to_csv(out / 'iod_price_effects.csv', index=False)
integration.round(4).to_csv(out / 'karamoja_market_integration.csv')
wholesale_res.round(4).to_csv(out / 'wholesale_replication.csv', index=False)
print('Saved 6 tables to data/processed/')
"""),
('md', r"""
---
## Summary

**What the prices show**
- **The seasonal cycle is the biggest, most predictable price risk.** Real maize prices swing about 30% between harvest and the lean season every year; sorghum in Karamoja swings up to 44%.
- **A dry first season raises maize prices** by about 9–11% per standard deviation of rainfall shortfall, in both retail (2011–2025) and wholesale (2006–2021) data. *Medium confidence.*
- **A dry second season raises sorghum prices** by about 8–9%. *Suggestive; one dataset.*
- **A positive July–August IOD is followed by cheaper maize and sorghum** the next February–May. *Suggestive; not replicated in wholesale data.*
- **Beans don't respond** to Ugandan rainfall.
- **Karamoja's maize and sorghum prices move with the national market**, but its staple's seasonal swing is the largest in the country, and real prices ran over 40% above normal in the 2022 crisis.

**Caveats**
- 15–16 years of prices limit statistical certainty; the permutation tests are the honest measure.
- Retail prices in seven towns stand in for the prices farmers receive, which are lower and swing more.
- The CPI deflator is national; regional inflation can differ.

**Next steps**
1. Add satellite vegetation (NDVI) over cropland as an independent measure of harvest outcomes, to test the rainfall → harvest → price chain directly.
2. Extend to the regional price data from FEWS NET and the East African Grain Council to lengthen the record.
3. Quantify the storage decision: real returns to storage net of loss rates and interest, by market.
"""),
]

# ======================================================================= 03 crop greenness
NB3 = [
('md', r"""
# From rain to crops to prices: satellite crop greenness in Uganda
**Greenness:** MODIS vegetation indices (NDVI, EVI) at 250 m, monthly, averaged over cropland and over pasture (grassland and shrubland) in each region. Main series from the Terra satellite (February 2000 – August 2026); the Aqua satellite (July 2002 onwards) is used as an independent check. Exported with `scripts/gee/uganda_cropland_ndvi_gee.js`.
**Rainfall, prices:** as in notebook 02.

**Questions**
1. Does crop greenness respond to rainfall, and how quickly?
2. Does it pick up Uganda's known droughts?
3. Does weaker greenness lead to higher food prices, and does it predict prices better than rainfall?
4. What do crops and pasture in Karamoja show about the 2022 food crisis?

**Why greenness.** Rainfall says how much water arrived; greenness says how crops actually grew. If the rainfall-to-price link found in notebook 02 really runs through harvests, greenness should sit in the middle of that chain.

Each finding is written up in a markdown cell directly after the code and output that produced it.
"""),
('md', '## Setup and data checks'),
('code', STYLE + r"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
import prices as P
from prices import load_vegetation, season_greenness, monthly_anomaly, seasonal_rain_z, season_panel, fit
rng = np.random.default_rng(42)
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
REGIONS = ['Uganda (national)', 'Karamoja', 'Northern', 'Eastern', 'Central', 'Western', 'Lake Victoria basin']

terra, aqua = load_vegetation('terra'), load_vegetation('aqua')
print('Terra:', terra['date'].min(), '–', terra['date'].max(), '| rows', len(terra), '| missing NDVI', int(terra['ndvi'].isna().sum()))
print('Aqua: ', aqua['date'].min(), '–', aqua['date'].max(), '| rows', len(aqua), '| missing NDVI', int(aqua['ndvi'].isna().sum()))
area = terra.groupby(['region', 'cover'])['cover_km2'].first().unstack().reindex(REGIONS).round(0)
clear = terra.groupby('month')['good_frac'].mean().round(2)
display(area); print('Average share of area with a clear observation, by calendar month:'); print(clear.to_dict())
"""),
('code', r"""
both = pd.merge(terra, aqua, on=['region', 'cover', 'date'], suffixes=('_terra', '_aqua'))
agree = both.groupby(['region', 'cover']).apply(
    lambda g: pd.Series({'correlation': g['ndvi_terra'].corr(g['ndvi_aqua']),
                         'mean_abs_diff': (g['ndvi_terra'] - g['ndvi_aqua']).abs().mean()})).unstack('cover')
agree.reindex(REGIONS).round(3)
"""),
('md', r"""
**Checks:** both records are complete, with no missing months. Cloud is a modest problem: on average at least 90% of each area has a clear observation in every calendar month, lowest in October. Terra and Aqua, two satellites crossing at different times of day, agree closely (correlations 0.94–0.98), so later results don't hinge on one sensor.

**One limitation:** the cropland map (ESA WorldCover 2021) records about 27,000 km² of cropland nationally, well below the 70,000–90,000 km² Uganda actually cultivates. It captures annual-crop areas in the north and east well, but classes much of the banana–coffee belt of the Central and Western regions as trees or grassland. Cropland greenness in those two regions therefore covers fewer, less typical fields.
"""),

('md', '---\n## 1. The greenness calendar'),
('code', r"""
crop = terra[terra['cover'] == 'cropland'].pivot_table(index=['year', 'month'], columns='region', values='ndvi')
clim = crop.groupby(level=1).mean()[REGIONS]
fig, ax = plt.subplots(figsize=(9, 3.8))
colors = {'Karamoja': DRY, 'Northern': '#8a63d2', 'Central': ACCENT, 'Uganda (national)': INK}
for r in ['Uganda (national)', 'Karamoja', 'Northern', 'Central']:
    ax.plot(range(12), clim[r].values, lw=2.4 if r != 'Uganda (national)' else 1.6, color=colors[r],
            ls='-' if r != 'Uganda (national)' else '--', label=r)
ax.set_xticks(range(12)); ax.set_xticklabels(MON)
ax.set_ylabel('Cropland NDVI'); ax.set_title('Crops green up after the rains: twice a year in the south, once in the north')
ax.legend(ncol=2, fontsize=9); plt.tight_layout(); plt.show()
clim.T.round(2)
"""),
('md', r"""
**Finding:** cropland greenness follows the rain calendar. Central and the Lake Victoria basin green up twice a year, peaking around May and November. Karamoja greens up once, from April to a peak in July–August, and is brown by February. The Northern region stays green from May through October.
"""),

('md', '---\n## 2. How greenness responds to rain'),
('code', r"""
rain = P.load_rain().pivot_table(index=['year', 'month'], columns='region', values='rainfall_mm')
rz, gz = monthly_anomaly(rain), monthly_anomaly(crop)
rows = []
for r in REGIONS:
    row = {'region': r}
    for k in [1, 2, 3, 4]:
        j = pd.concat([gz[r], rz[r].rolling(k).mean()], axis=1).dropna()
        row[f'rain over last {k} mo'] = j.corr().iloc[0, 1]
    rows.append(row)
lag = pd.DataFrame(rows).set_index('region')
lag.round(2)
"""),
('code', r"""
windows = {'Mar–May rain → May–Jul greenness': ((3, 4, 5), (5, 6, 7)),
           'Oct–Dec rain → Nov–Jan greenness': ((10, 11, 12), (11, 12, 1))}
rows = []
for label, (rm, gm) in windows.items():
    rs, gs = seasonal_rain_z(list(rm)), season_greenness(months=gm)
    for r in REGIONS:
        x = rs.xs(r).reindex(gs.index); y = gs[r]
        ok = x.notna() & y.notna()
        c, p = stats.pearsonr(x[ok], y[ok])
        rows.append({'season': label, 'region': r, 'r': c, 'p': p, 'years': int(ok.sum())})
seas = pd.DataFrame(rows).pivot(index='region', columns='season', values=['r', 'p']).reindex(REGIONS)
seas.round(3)
"""),
('md', r"""
**Finding:** greenness tracks the previous **two to three months** of rain, with correlations of about 0.65–0.69 in the north, east and Karamoja, and a weaker 0.45–0.55 in Central and Western, where perennial crops and cloud blur the signal.

The two seasons behave very differently:
- **Second season:** October–December rain explains November–January greenness very closely (r = 0.84–0.88 nationally and in the north and east). When the short rains end early, crops and grass dry out fast.
- **First season:** March–May rain barely moves May–July greenness (r ≈ 0.0–0.35), except in Karamoja (r = 0.58). At the height of the long rains, vegetation in wetter regions is near its greenest whatever the exact rainfall, so the index saturates. Greenness is therefore a much better drought indicator for the second season and for dry Karamoja than for the long rains elsewhere.
"""),

('md', '---\n## 3. Does greenness see the droughts?'),
('code', r"""
gA = season_greenness(months=(5, 6, 7))
gB = season_greenness(months=(11, 12, 1))
worst = pd.DataFrame({
    'first season (May–Jul)': gA['Uganda (national)'].sort_values().head(6).round(2).reset_index().apply(lambda r: f"{int(r['index'])}: {r['Uganda (national)']:+.2f}", axis=1).values,
    'second season (Nov–Jan)': gB['Uganda (national)'].sort_values().head(6).round(2).reset_index().apply(lambda r: f"{int(r['index'])}: {r['Uganda (national)']:+.2f}", axis=1).values,
})
worst.index = range(1, 7)
worst
"""),
('md', r"""
**Finding:** the worst seasons by national cropland greenness line up with Uganda's recorded droughts: the failed **2005** short rains, the **2016–17** drought (second season 2016 and both 2017 seasons), the **2008–09** drought, and **2021–22**. The worst first season in the 26-year record is **2022**, and the second-worst is **2021**.
"""),

('md', '---\n## 4. A puzzle: 2016–2022'),
('code', r"""
def residual_by_year(veg, region, start=2003):
    # Greenness anomaly left unexplained by the previous three months' rain, averaged by year
    w = veg[(veg['cover'] == 'cropland') & (veg['region'] == region) & (veg['year'] >= start)]
    w = w.set_index(['year', 'month'])['ndvi']
    j = pd.concat([monthly_anomaly(w.to_frame())['ndvi'], rz[region].rolling(3).mean()], axis=1, keys=['g', 'r']).dropna()
    return sm.OLS(j['g'], sm.add_constant(j['r'])).fit().resid.groupby(level=0).mean()

res = pd.DataFrame({(r, s): residual_by_year(v, r) for r in ['Uganda (national)', 'Central', 'Karamoja']
                    for s, v in [('Terra', terra), ('Aqua', aqua)]})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, r in zip(axes, ['Uganda (national)', 'Central', 'Karamoja']):
    ax.axhline(0, color='#c9cfc8', lw=1); ax.axvspan(2015.5, 2022.5, color=GRID, alpha=.6, lw=0)
    ax.plot(res.index, res[(r, 'Terra')], color=WET, lw=2, marker='o', ms=3, label='Terra')
    ax.plot(res.index, res[(r, 'Aqua')], color=DRY, lw=2, marker='s', ms=3, ls='--', label='Aqua')
    ax.set_title(r)
axes[0].set_ylabel('Greenness not explained\nby rain (SD)'); axes[0].legend(fontsize=9)
plt.tight_layout(); plt.show()
res.loc[2014:].round(2)
"""),
('md', r"""
**Finding:** from **2016 to 2022**, cropland nationally and in the centre and south was consistently less green than that period's rainfall would predict. The gap is largest in 2021–22 and closes nationally from 2023, though Central dips again in 2025. Karamoja shows no such gap.

**It is not a sensor artefact.** Terra's orbit has drifted since 2020, which can bias its readings, so the same export was run from Aqua, which crosses in the afternoon. Aqua shows the same gap, of almost the same size, including in 2016–2019, before any drift.

**Three explanations remain**, and this data can't fully separate them:
1. **Crops genuinely did worse than monthly rainfall suggests.** Fall armyworm reached Uganda in 2016 and damaged maize widely in 2016–2018, and dry spells *within* months can hurt crops without lowering monthly totals.
2. **The satellite rainfall overstated rain in those years** in the south and centre, for example through changes in the rain gauges that CHIRPS draws on.
3. **Land-use change against a fixed map.** In Central, the gap is not just an episode: it declines steadily from about +0.9 in 2007 to −1.6 in 2022. The cropland map is a 2021 snapshot, so land that was greener bush or forest before being cleared for farming counts as "cropland" throughout and makes earlier years look greener. Central, with fast land conversion around Kampala, is where this would show most.

The 2016–2022 dip appears nationally on top of a much flatter long-run pattern, so explanations 1 and 2 matter there; in Central, explanation 3 likely contributes too. Checking against station rainfall, a second rainfall product, and a year-by-year cropland map would separate them.
"""),

('md', '---\n## 5. Does greenness predict prices?'),
('code', r"""
# Pre-specified windows: the peak-growth months of each season
WINDOWS = {'A': (5, 6, 7), 'B': (11, 12, 1)}
greens = {k: season_greenness(months=m) for k, m in WINDOWS.items()}

def permutation_p(q, x, B=2000):
    q = q.dropna(subset=['price_anom', x, 'world'])
    yearly = q.groupby('year').agg(p=('price_anom', 'mean'), x=(x, 'mean'), w=('world', 'mean'))
    obs = smf.ols('p ~ x + w', yearly).fit().params['x']
    null = [smf.ols('p ~ x + w', yearly.assign(x=rng.permutation(yearly['x'].values))).fit().params['x'] for _ in range(B)]
    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1)

rows = []
for com in ['Maize (white)', 'Maize flour', 'Sorghum', 'Beans']:
    pan = season_panel(com)
    for s in ['A', 'B']:
        q = pan[pan['season'] == s].copy()
        g = greens[s]
        q['green_national'] = q['year'].map(g['Uganda (national)'])
        q['green_regional'] = [g[r].get(y) if r in g else np.nan for r, y in zip(q['region'], q['year'])]
        r_nat, r_reg = fit(q, 'green_national'), fit(q, 'green_regional')
        both = q.dropna(subset=['green_national', 'rain_national', 'price_anom', 'world'])
        race = smf.ols('price_anom ~ green_national + rain_national + world + C(market)', both).fit(
            cov_type='cluster', cov_kwds={'groups': both['year']})
        rows.append({'commodity': com, 'season': s,
                     'less_green_pct': 100 * (np.exp(-r_nat['coef']) - 1), 'p': r_nat['p'],
                     'permutation_p': permutation_p(q, 'green_national'),
                     'regional_less_green_pct': 100 * (np.exp(-r_reg['coef']) - 1), 'regional_p': r_reg['p'],
                     'both_in_model_green_p': race.pvalues['green_national'], 'both_in_model_rain_p': race.pvalues['rain_national'],
                     'years': r_nat['years']})
green_price = pd.DataFrame(rows)
green_price.round(3)
"""),
('md', r"""
**Reading the table:** `less_green_pct` is the change in real prices after a season whose crop greenness was one standard deviation **below** normal (May–July for the first season, November–January for the second). The last two p-value columns put greenness and rainfall in the same model, to see which carries the information.

**Finding:** greenness and prices line up as the chain predicts. Less green seasons are followed by higher prices for maize and maize flour after both seasons, and for sorghum after the second season, by about 6–9% per standard deviation, similar to the rainfall effects in notebook 02. The clearest case is **sorghum after the second season** (about 9%, permutation p ≈ 0.06). With only 15–17 years, none of these passes the permutation test at 5%. Beans again show nothing.

**But greenness doesn't beat rainfall.** With both in the model, neither stands out for the second season: they carry the same information (Section 2: r ≈ 0.85). For the first season, rainfall predicts maize prices better, as expected, since greenness saturates during the long rains.

**A lead for future testing:** among windows tried during exploration, April–June **EVI** (an index that saturates less in lush vegetation) predicted first-season maize prices strongly (about 11% per standard deviation, permutation p ≈ 0.03). Four windows were tried, so after allowing for that search the evidence is about p ≈ 0.13: a hypothesis to test on coming seasons, not a finding.
"""),

('md', '---\n## 6. Karamoja and the 2022 crisis'),
('code', r"""
kar = pd.DataFrame({
    'Karamoja cropland, Jun–Sep': season_greenness(months=(6, 7, 8, 9))['Karamoja'],
    'Karamoja pasture, Jun–Sep': season_greenness(cover='pasture', months=(6, 7, 8, 9))['Karamoja'],
    'National cropland, May–Jul': gA['Uganda (national)'],
    'National cropland, Nov–Jan': gB['Uganda (national)'],
    'National rain, Mar–May': seasonal_rain_z([3, 4, 5]).xs('Uganda (national)'),
    'National rain, Oct–Dec': seasonal_rain_z([10, 11, 12]).xs('Uganda (national)'),
}).loc[2018:2025]
kar.round(2)
"""),
('md', r"""
**Finding:** Karamoja's own crops and pasture were **close to normal or better** in 2021 and 2022. The rest of the country was not: national crop greenness was among the lowest on record in the 2021 first season, the 2021 second season and the 2022 first season. Rainfall points the same way, though less strongly (national March–May rain in 2022 was 0.7 standard deviations below normal, and 1.2 below in the Northern region).

This sharpens the story from notebook 02. Karamoja's staple prices nearly doubled by mid-2022 even though its own harvests were not failing. The spike fits a **national shortage after poor 2021 and 2022 seasons, carried into Karamoja through the market**, compounded by the global grain-price shock of 2022. Karamoja's maize prices move closely with the national market (notebook 02), so a deficit region feels national shortfalls even in a reasonable local year. For early warning, that means national crop conditions matter for Karamoja, not only Karamoja's own rains.
"""),

('md', '---\n## Export'),
('code', r"""
from pathlib import Path
out = Path('../data/processed')
lag.round(4).to_csv(out / 'greenness_rain_lag.csv')
seas.round(4).to_csv(out / 'greenness_rain_seasonal.csv')
res.round(4).to_csv(out / 'greenness_unexplained_by_rain.csv')
green_price.round(4).to_csv(out / 'greenness_price_effects.csv', index=False)
pd.DataFrame({'first_season': gA['Uganda (national)'], 'second_season': gB['Uganda (national)']}).round(4).to_csv(out / 'national_greenness_by_season.csv')
kar.round(4).to_csv(out / 'karamoja_greenness_2018_2025.csv')
print('Saved 6 tables to data/processed/')
"""),
('md', r"""
---
## Summary

- **Greenness follows rain with a lag of two to three months**, very closely for the second season (r ≈ 0.85) and for Karamoja, but weakly for the first season elsewhere, when vegetation is saturated.
- **It detects Uganda's droughts:** 2005, 2008–09, 2016–17 and 2021–22.
- **From 2016 to 2022, crops in the centre and south were less green than rainfall predicts.** Two satellites agree, so it isn't a sensor artefact. Crops may have done worse than rain suggests (fall armyworm, dry spells), the satellite rainfall may have overstated those years, and in Central a fixed 2021 cropland map probably adds a long-run downward drift.
- **Less green seasons are followed by higher maize and sorghum prices**, by about 6–9% per standard deviation (consistent but not individually significant under the permutation test), and greenness adds little beyond rainfall for prediction.
- **Karamoja's 2022 price spike came while its own crops were near normal**, during national shortfalls: a deficit region feels national shortages.

**Next steps**
1. Check the 2016–22 gap against station rainfall or a second rainfall product (ERA5, TAMSAT).
2. Use a cropland map built for smallholder farming to improve the Central and Western series.
3. Test April–June EVI as an early first-season indicator on coming seasons.
"""),
]

if __name__ == '__main__':
    out = ROOT / 'notebooks'
    out.mkdir(exist_ok=True)
    notebook(NB1, out / '01_faostat_data_quality.ipynb')
    notebook(NB2, out / '02_rainfall_and_food_prices.ipynb')
    notebook(NB3, out / '03_crop_greenness.ipynb')
