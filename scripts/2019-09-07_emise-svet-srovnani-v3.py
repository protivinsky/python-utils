import os
import pyreadstat
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from libs.utils import *
from libs.plots import *
from libs.extensions import *
plt.ioff()

# EMISE SVET - SROVNANI (#111) ---
#   1. Load all datasets, possibly with metadata ---
#   2. Unify country codes ---
#   3. Cross-check it against each other and against sums ---
#   4. Create an aggregate dataset ---

# ok, that's still what I want to do, also include some additional time series and transform it into a long form

root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
df = pd.read_parquet(root + '\\wb_with_gca.parquet')
countries = pd.read_csv(root + '\\countries.csv')

data = df[['code', 'year', 'gca_co2', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'EN.ATM.CO2E.KT']] \
    .rename(columns={'year': 'year_data', 'gca_co2': 'gca_co2', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp',
                     'EN.ATM.CO2E.KT': 'co2', 'EN.ATM.GHGT.KT.CE': 'ghg'})

# not bad, but I really want to join edgar
edgar_code = countries[['code', 'edgar_name']].copy()
edgar_code['edgar_name'] = edgar_code['edgar_name'].fillna('')
edgar_code = edgar_code[edgar_code['edgar_name'] != ''].reset_index(drop=True)

path_edgar = root + '\\edgar\\EDGARv5.0_FT2017_fossil_CO2_booklet2018.xls'

edgar = pd.read_excel(path_edgar, sheet_name='fossil_CO2_totals_by_country', skipfooter=5)
edgar = pd.merge(edgar, edgar_code.rename(columns={'edgar_name': 'country_name'}))
edgar = edgar.drop(columns=['country_name']).set_index('code')
edgar.columns = edgar.columns.rename('year')
edgar = edgar.unstack().rename('edgar_co2').reset_index()

edgar_gdp = pd.read_excel(path_edgar, sheet_name='fossil_CO2_per_GDP_by_country')
edgar_gdp = pd.merge(edgar_gdp, edgar_code.rename(columns={'edgar_name': 'country_name'}))
edgar_gdp = edgar_gdp.drop(columns=['country_name']).set_index('code')
edgar_gdp.columns = edgar_gdp.columns.rename('year')
edgar_gdp = edgar_gdp.unstack().rename('edgar_co2_per_gdp').reset_index()

edgar_capita = pd.read_excel(path_edgar, sheet_name='fossil_CO2_per_capita_by_countr')
edgar_capita = pd.merge(edgar_capita, edgar_code.rename(columns={'edgar_name': 'country_name'}))
edgar_capita = edgar_capita.drop(columns=['country_name']).set_index('code')
edgar_capita.columns = edgar_capita.columns.rename('year')
edgar_capita = edgar_capita.unstack().rename('edgar_co2_per_capita').reset_index()

edgar = pd.merge(pd.merge(edgar, edgar_capita, how='outer'), edgar_gdp, how='outer')

# sectors
edgar_sectors_all = pd.read_excel(path_edgar, sheet_name='fossil_CO2_by_sector_and_countr', skipfooter=2)
sectors = edgar_sectors_all['Sector'].unique()

for s in sectors:
    logger(s)
    edgar_sector = edgar_sectors_all[edgar_sectors_all['Sector'] == s].copy().drop(columns=['Sector'])
    edgar_sector = pd.merge(edgar_sector, edgar_code.rename(columns={'edgar_name': 'country_name'}))
    edgar_sector = edgar_sector.drop(columns=['country_name']).set_index('code')
    edgar_sector.columns = edgar_sector.columns.rename('year')
    edgar_sector = edgar_sector.unstack().rename('edgar_co2_{}'.format(slugify(s))).reset_index()
    edgar = pd.merge(edgar, edgar_sector, how='outer')

edgar.show_csv()

df = pd.merge(df, edgar, how='outer')
df.to_csv(root + '\\data.csv', encoding='utf-8-sig', index=False)

# this is the mega dataset

data = df[['code', 'year', 'gca_co2', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'EN.ATM.CO2E.KT', 'EN.ATM.GHGT.KT.CE',
           'edgar_co2']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp',
                     'EN.ATM.CO2E.KT': 'co2', 'EN.ATM.GHGT.KT.CE': 'ghg'})

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2012)
data['year_data'] = np.int_(data['year_data'])

res = pd.merge_asof(codes, data.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
res = pd.merge(res, countries[['code', 'en_short', 'en_region', 'cz_short', 'cz_region']])
res.show_csv()

plots = []
for x, y in [('co2', 'gca_co2'), ('co2', 'edgar_co2'), ('co2', 'ghg')]:
    fig, ax = plt.subplots()
    sns.scatterplot(x, y, data=res)
    ax.set(title=f'{x} - {y}')
    plots.append(ax)

for x, y in [('co2', 'gca_co2'), ('co2', 'edgar_co2'), ('co2', 'ghg')]:
    fig, ax = plt.subplots()
    sns.scatterplot(x, y, data=res)
    ax.set(title=f'{x} - {y} -- log', xscale='log', yscale='log')
    plots.append(ax)

Chart(plots).show()

res['ratio_ghg_co2'] = res['ghg'] / res['co2']
res[res['ratio_ghg_co2'] > 10].show_csv()

# can I just plot it? ---
# and call Ondras ---

import squarify

foo = res[np.isfinite(res['ghg'])]
squarify.plot(sizes=foo['ghg'], label=foo['cz_short'], alpha=0.8).show()


countries['en_category'] = countries['en_category'].fillna('')
cats = pd.DataFrame({'en_category': np.sort(countries['en_category'].unique())[1:]})
cats['cz_category'] = ['Afrika', 'Arabské státy', 'Asie a Oceánie', 'Austrálie a Nový Zéland', 'Čína', 'Evropa',
                       'Evropská unie', 'Indie', 'Severní Amerika', 'Rusko', 'Jižní Amerika', 'Spojené státy americké']
cats['cz_cat_desc'] = ['Afrika', 'Arabské státy', 'Asie a Oceánie (mimo Čínu a Indii)', 'Austrálie a Nový Zéland',
                       'Čína', 'Evropa (mimo EU a Rusko)', 'Evropská unie', 'Indie', 'Severní Amerika (mimo USA)',
                       'Rusko', 'Jižní Amerika', 'Spojené státy americké']

countries[countries['en_category'] == 'European Union']['en_short']

cats

countries = pd.merge(countries, cats, how='left')
countries.to_csv(root + '\\countries.csv', encoding='utf-8-sig', index=False)

countries = pd.read_csv(root + '\\countries.csv')

data = df[['code', 'year', 'gca_co2', 'SP.POP.TOTL', 'NY.GDP.MKTP.KD', 'NY.GDP.MKTP.PP.KD', 'EN.ATM.CO2E.KT',
           'EN.ATM.GHGT.KT.CE', 'edgar_co2']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.KD': 'gdp', 'NY.GDP.MKTP.PP.KD': 'gdp_ppp',
                     'EN.ATM.CO2E.KT': 'co2', 'EN.ATM.GHGT.KT.CE': 'ghg'})

vars = ['pop', 'gdp_ppp', 'gdp', 'ghg', 'co2', 'gca_co2', 'edgar_co2']
data = data.dropna()

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2012)
data['year_data'] = np.int_(data['year_data'])

res = pd.merge_asof(codes, data.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
res = pd.merge(res, countries[['code', 'en_short', 'en_region', 'cz_short', 'cz_region', 'en_category', 'cz_category',
                               'cz_cat_desc']])


agg = res.groupby('en_category')[vars].sum().reset_index()
plt.rcParams['figure.figsize'] = 8, 6

agg['ghg_per_capita'] = 1_000 * agg['ghg'] / agg['pop']
agg['co2_per_capita'] = 1_000 * agg['co2'] / agg['pop']
agg['ghg_per_gdp'] = 1_000_000 * agg['ghg'] / agg['gdp_ppp']
agg['co2_per_gdp'] = 1_000_000 * agg['co2'] / agg['gdp_ppp']

x = 'pop'


colors = sns.color_palette('Paired', n_colors=12)

fig, ax = plt.subplots()
sns.palplot(colors)
ax.show()

plots = []

units = {
    'ghg': 'kt CO2eq',
    'co2': 'kt CO2',
    'gca_co2': 'Mt CO2',
    'edgar_co2': 'Mt CO2',
    'ghg_per_capita': 't CO2eq',
    'co2_per_capita': 't CO2',
    'ghg_per_gdp': 'kg CO2eq / $ PPP',
    'co2_per_gdp': 'kg CO2 / $ PPP',
}


for x in vars:
    logger(x)
    fig, ax = plt.subplots()
    total = agg[x].sum()
    labels = agg.apply(lambda f: '{}\n{:.2g} ({:.1f}%)'.format(f['en_category'], f[x], 100 * f[x] / total), axis=1)
    squarify.plot(sizes=agg[x], label=labels, alpha=0.8, color=colors)
    if x in units:
        ax.set(title='{} ({}) -- {:.3g}'.format(x, units[x], total))
    else:
        ax.set(title='{} -- {:.3g}'.format(x, total))
    plots.append(ax)

for x in ['ghg_per_capita', 'co2_per_capita', 'ghg_per_gdp', 'co2_per_gdp']:
    logger(x)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3)
    sns.barplot(x=x, y='en_category', data=agg)
    ax.set(title='{} ({})'.format(x, units[x]))
    plots.append(ax)

Chart(plots, title='Comparison of regions').show()


agg.dtypes
agg.show_csv()

# include edgar 4.3.2

edgar_files = ['CH4', 'CO2_excl_short-cycle_org_C', 'CO2_org_short-cycle_C', 'N2O']
ef = edgar_files[0]

edgar_df = None

for ef in edgar_files:
    logger(ef)
    frame = pd.read_excel(f'{root}\\edgar_v4.3.2\\v432_{ef}_1970_2012.xls', sheet_name='TOTALS BY COUNTRY',
                          header=7)
    frame = frame[['ISO_A3'] + list(range(1970, 2013))].rename(columns={'ISO_A3': 'code'}).set_index('code')
    frame.columns = frame.columns.rename('year')
    frame = frame.unstack().rename(f'edgar432_{ef}').reset_index()
    frame = frame[~frame['code'].isin(['SEA', 'AIR'])]
    if edgar_df is None:
        edgar_df = frame
    else:
        edgar_df = pd.merge(edgar_df, frame, how='outer')

edgar_df.to_csv(root + '\\edgar_v4.3.2.csv', index=False)

data = edgar_df.copy()

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2012)
data = data.rename(columns={'year': 'year_data'})
data = data.dropna()
data['year_data'] = np.int_(data['year_data'])

res = pd.merge_asof(codes, data.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
res = pd.merge(res, countries[['code', 'en_short', 'en_region', 'cz_short', 'cz_region', 'en_category', 'cz_category',
                               'cz_cat_desc']])

res['co2eq_wo_short'] = res['edgar432_CO2_excl_short-cycle_org_C'] + 25 * res['edgar432_CH4'] + 298 * res['edgar432_N2O']
res['co2eq'] = res['co2eq_wo_short'] + res['edgar432_CO2_org_short-cycle_C']

res.show_csv()

vars = [f'edgar432_{ef}' for ef in edgar_files]
agg = res.groupby('en_category')[vars].sum().reset_index()

agg['co2eq_wo_short'] = agg['edgar432_CO2_excl_short-cycle_org_C'] + 25 * agg['edgar432_CH4'] + 298 * agg['edgar432_N2O']
agg['co2eq'] = agg['co2eq_wo_short'] + agg['edgar432_CO2_org_short-cycle_C']

plots = []
for x in ['co2eq_wo_short', 'co2eq']:
    logger(x)
    fig, ax = plt.subplots()
    total = agg[x].sum()
    labels = agg.apply(lambda f: '{}\n{:.2g} ({:.1f}%)'.format(f['en_category'], f[x], 100 * f[x] / total), axis=1)
    squarify.plot(sizes=agg[x], label=labels, alpha=0.8, color=colors)
    if x in units:
        ax.set(title='{} ({}) -- {:.3g}'.format(x, units[x], total))
    else:
        ax.set(title='{} -- {:.3g}'.format(x, total))
    plots.append(ax)

Chart(plots, title='Comparison of regions').show()

# join it all
df
edgar_df.dtypes

df = pd.merge(df, edgar_df, how='outer')
df.to_csv(root + '\\data_all.csv', encoding='utf-8-sig', index=False)

df['edgar432_co2'] = df['edgar432_CO2_excl_short-cycle_org_C']
df['edgar432_co2_w_short'] = df['edgar432_co2'] + df['edgar432_CO2_org_short-cycle_C']
df['edgar432_co2eq'] = df['edgar432_CO2_excl_short-cycle_org_C'] + 25 * df['edgar432_CH4'] + 298 * df['edgar432_N2O']
df['edgar432_co2eq_w_short'] = df['edgar432_co2eq'] + df['edgar432_CO2_org_short-cycle_C']


data = df[['code', 'year', 'gca_co2', 'SP.POP.TOTL', 'NY.GDP.MKTP.KD', 'NY.GDP.MKTP.PP.KD', 'EN.ATM.CO2E.KT',
           'EN.ATM.GHGT.KT.CE', 'edgar_co2', 'edgar432_co2', 'edgar432_co2_w_short', 'edgar432_co2eq',
           'edgar432_co2eq_w_short', 'edgar432_CH4', 'edgar432_N2O']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.KD': 'gdp', 'NY.GDP.MKTP.PP.KD': 'gdp_ppp',
                     'EN.ATM.CO2E.KT': 'co2', 'EN.ATM.GHGT.KT.CE': 'ghg'})

vars = ['pop', 'gdp_ppp', 'gdp', 'ghg', 'co2', 'gca_co2', 'edgar_co2', 'edgar432_co2', 'edgar432_co2_w_short',
        'edgar432_co2eq', 'edgar432_co2eq_w_short', 'edgar432_CH4', 'edgar432_N2O']
data = data.dropna()

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2012)
data['year_data'] = np.int_(data['year_data'])

res = pd.merge_asof(codes, data.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
res = pd.merge(res, countries[['code', 'en_short', 'en_region', 'cz_short', 'cz_region', 'en_category', 'cz_category',
                               'cz_cat_desc']])


agg = res.groupby('en_category')[vars].sum().reset_index()
plt.rcParams['figure.figsize'] = 8, 6

agg['ghg_per_capita'] = 1_000 * agg['ghg'] / agg['pop']
agg['co2_per_capita'] = 1_000 * agg['co2'] / agg['pop']
agg['ghg_per_gdp'] = 1_000_000 * agg['ghg'] / agg['gdp_ppp']
agg['co2_per_gdp'] = 1_000_000 * agg['co2'] / agg['gdp_ppp']
agg['ghg_per_capita_edgar'] = 1_000 * agg['edgar432_co2eq'] / agg['pop']
agg['ghg_per_gdp_edgar'] = 1_000_000 * agg['edgar432_co2eq'] / agg['gdp_ppp']

colors = sns.color_palette('Paired', n_colors=12)

plots = []

units = {
    'ghg': 'kt CO2eq',
    'co2': 'kt CO2',
    'gca_co2': 'Mt CO2',
    'edgar_co2': 'Mt CO2',
    'ghg_per_capita': 't CO2eq',
    'co2_per_capita': 't CO2',
    'ghg_per_gdp': 'kg CO2eq / $ PPP',
    'co2_per_gdp': 'kg CO2 / $ PPP',
    'ghg_per_capita_edgar': 't CO2eq',
    'ghg_per_gdp_edgar': 'kg CO2eq / $ PPP',
}


for x in vars:
    logger(x)
    fig, ax = plt.subplots()
    total = agg[x].sum()
    labels = agg.apply(lambda f: '{}\n{:.2g} ({:.1f}%)'.format(f['en_category'], f[x], 100 * f[x] / total), axis=1)
    squarify.plot(sizes=agg[x], label=labels, alpha=0.8, color=colors)
    if x in units:
        ax.set(title='{} ({}) -- {:.3g}'.format(x, units[x], total))
    else:
        ax.set(title='{} -- {:.3g}'.format(x, total))
    plots.append(ax)

for x in ['ghg_per_capita', 'co2_per_capita', 'ghg_per_gdp', 'co2_per_gdp', 'ghg_per_capita_edgar', 'ghg_per_gdp_edgar']:
    logger(x)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3)
    sns.barplot(x=x, y='en_category', data=agg)
    ax.set(title='{} ({})'.format(x, units[x]))
    plots.append(ax)

Chart(plots, title='Comparison of regions').show()


# Ok, I am happy with this - wrap it up and upload it ---

df = pd.read_csv(root + '\\data_all.csv')

df['edgar432_co2'] = df['edgar432_CO2_excl_short-cycle_org_C']
df['edgar432_co2_w_short'] = df['edgar432_co2'] + df['edgar432_CO2_org_short-cycle_C']
df['edgar432_co2eq'] = df['edgar432_CO2_excl_short-cycle_org_C'] + 25 * df['edgar432_CH4'] + 298 * df['edgar432_N2O']
df['edgar432_co2eq_w_short'] = df['edgar432_co2eq'] + df['edgar432_CO2_org_short-cycle_C']

data = df[['code', 'year', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'edgar432_co2eq']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp_ppp',
                     'edgar432_co2eq': 'co2eq'})

vars = ['pop', 'gdp_ppp', 'co2eq']
data = data.dropna()

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2012)
data['year_data'] = np.int_(data['year_data'])

res = pd.merge_asof(codes, data.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
res = pd.merge(res, countries[['code', 'en_short', 'en_region', 'cz_short', 'cz_region', 'en_category', 'cz_category',
                               'cz_cat_desc']])

agg = res.groupby(['en_category', 'cz_category', 'cz_cat_desc'])[vars].sum().reset_index()
plt.rcParams['figure.figsize'] = 8, 6

agg['co2eq_per_capita'] = 1_000 * agg['co2eq'] / agg['pop']
agg['co2eq_per_gdp'] = 1_000_000 * agg['co2eq'] / agg['gdp_ppp']

colors = sns.color_palette('Paired', n_colors=12)

plots = []

units = {
    'co2eq': 'kt CO2eq',
    'co2eq_per_capita': 't CO2eq',
    'co2eq_per_gdp': 'kg CO2eq / $ PPP',
}


for x in vars:
    logger(x)
    fig, ax = plt.subplots()
    total = agg[x].sum()
    labels = agg.apply(lambda f: '{}\n{:.2g} ({:.1f}%)'.format(f['en_category'], f[x], 100 * f[x] / total), axis=1)
    squarify.plot(sizes=agg[x], label=labels, alpha=0.8, color=colors)
    if x in units:
        ax.set(title='{} ({}) -- {:.3g}'.format(x, units[x], total))
    else:
        ax.set(title='{} -- {:.3g}'.format(x, total))
    plots.append(ax)

for x in ['co2eq_per_capita', 'co2eq_per_gdp']:
    logger(x)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3)
    sns.barplot(x=x, y='en_category', data=agg)
    ax.set(title='{} ({})'.format(x, units[x]))
    plots.append(ax)

Chart(plots, title='Comparison of regions').show()

res.to_csv(root + '\\output_res.csv', encoding='utf-8-sig', index=False)
agg.to_csv(root + '\\output_agg.csv', encoding='utf-8-sig', index=False)

foo = np.array(["ABW", "AFG", "AGO", "ALB", "AND", "ARB", "ARE", "ARG", "ARM", "ASM", "ATG", "AUS", "AUT", "AZE", "BDI", "BEL", "BEN", "BFA", "BGD", "BGR", "BHR", "BHS", "BIH", "BLR", "BLZ", "BMU", "BOL", "BRA", "BRB", "BRN", "BTN", "BWA", "CAF", "CAN", "CEB", "CHE", "CHI", "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COL", "COM", "CPV", "CRI", "CSS", "CUB", "CUW", "CYM", "CYP", "CZE", "DEU", "DJI", "DMA", "DNK", "DOM", "DZA", "EAP", "EAR", "EAS", "ECA", "ECS", "ECU", "EGY", "EMU", "ERI", "ESP", "EST", "ETH", "EUU", "FCS", "FIN", "FJI", "FRA", "FRO", "FSM", "GAB", "GBR", "GEO", "GHA", "GIB", "GIN", "GMB", "GNB", "GNQ", "GRC", "GRD", "GRL", "GTM", "GUM", "GUY", "HIC", "HKG", "HND", "HPC", "HRV", "HTI", "HUN", "IBD", "IBT", "IDA", "IDB", "IDN", "IDX", "IMN", "IND", "INX", "IRL", "IRN", "IRQ", "ISL", "ISR", "ITA", "JAM", "JOR", "JPN", "KAZ", "KEN", "KGZ", "KHM", "KIR", "KNA", "KOR", "KWT", "LAC", "LAO", "LBN", "LBR", "LBY", "LCA", "LCN", "LDC", "LIC", "LIE", "LKA", "LMC", "LMY", "LSO", "LTE", "LTU", "LUX", "LVA", "MAC", "MAF", "MAR", "MCO", "MDA", "MDG", "MDV", "MEA", "MEX", "MHL", "MIC", "MKD", "MLI", "MLT", "MMR", "MNA", "MNE", "MNG", "MNP", "MOZ", "MRT", "MUS", "MWI", "MYS", "NAC", "NAM", "NCL", "NER", "NGA", "NIC", "NLD", "NOR", "NPL", "NRU", "NZL", "OED", "OMN", "OSS", "PAK", "PAN", "PER", "PHL", "PLW", "PNG", "POL", "PRE", "PRI", "PRK", "PRT", "PRY", "PSE", "PSS", "PST", "PYF", "QAT", "ROU", "RUS", "RWA", "SAS", "SAU", "SDN", "SEN", "SGP", "SLB", "SLE", "SLV", "SMR", "SOM", "SRB", "SSA", "SSD", "SSF", "SST", "STP", "SUR", "SVK", "SVN", "SWE", "SWZ", "SXM", "SYC", "SYR", "TCA", "TCD", "TEA", "TEC", "TGO", "THA", "TJK", "TKM", "TLA", "TLS", "TMN", "TON", "TSA", "TSS", "TTO", "TUN", "TUR", "TUV", "TZA", "UGA", "UKR", "UMC", "URY", "USA", "UZB", "VCT", "VEN", "VGB", "VIR", "VNM", "VUT", "WLD", "WSM", "XKX", "YEM", "ZAF", "ZMB", "ZWE"])

np.all(foo[1:] > foo[:-1])


# I need to fix Arab States
root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
countries = pd.read_csv(root + '\\countries.csv')

cats = countries[['en_category', 'cz_category', 'cz_cat_desc']].drop_duplicates().dropna().sort_values('en_category') \
    .reset_index(drop=True)

arab_africa = ['COM', 'DJI', 'DZA', 'EGY', 'LBY', 'MAR', 'MRT', 'SDN', 'SOM', 'TUN']
arab_asia = ['ARE', 'BHR', 'IRN', 'IRQ', 'JOR', 'KWT', 'LBN', 'OMN', 'PSE', 'QAT', 'SAU', 'YEM']

countries = countries.drop(columns=['cz_category', 'cz_cat_desc']).set_index('code')
countries.loc[arab_africa, 'en_category'] = 'Africa'
countries.loc[arab_asia, 'en_category'] = 'Asia & Pacific'
countries = countries.reset_index()

countries = pd.merge(countries, cats, how='left')
countries = countries.sort_values('code')

col_to_san = ['en_category', 'cz_category', 'cz_cat_desc']
for c in col_to_san:
    countries[c] = countries[c].fillna('')

countries.to_csv(root + '\\countries.csv', encoding='utf-8-sig', index=False)

countries.show_csv()


