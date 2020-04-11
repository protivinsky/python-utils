# imports
#region
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
#endregion


# RESPEKT - datove podklady ---

root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'

df = pd.read_csv(f'{root}\\data_all_w_edgar50.csv')
df.dtypes

df['edgar50_co2'] = df['edgar50_CO2_excl_short-cycle_org_C']
df['edgar50_co2_w_short'] = df['edgar50_co2'] + df['edgar50_CO2_org_short-cycle_C']
# I am using the new sensitivities here, 28 and 265
df['edgar50_co2eq'] = df['edgar50_CO2_excl_short-cycle_org_C'] + 28 * df['edgar50_CH4'] + 265 * df['edgar50_N2O']
# df['edgar50_co2eq'] = df['edgar50_CO2_excl_short-cycle_org_C'] + 25 * df['edgar50_CH4'] + 298 * df['edgar50_N2O']
df['edgar50_co2eq_w_short'] = df['edgar50_co2eq'] + df['edgar50_CO2_org_short-cycle_C']

df.query('year == 2015')['edgar50_co2eq'].sum() / 1e6
df.query('year == 2015')['edgar50_co2'].sum() / 1e6

data = df[['code', 'year', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'edgar50_co2eq', 'edgar50_CO2_excl_short-cycle_org_C',
           'edgar50_CH4', 'edgar50_N2O', 'edgar50_co2eq_w_short', 'edgar50_co2_w_short']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp', 'edgar50_co2eq': 'co2eq',
                     'edgar50_CO2_excl_short-cycle_org_C': 'co2', 'edgar50_CH4': 'ch4', 'edgar50_N2O': 'n2o',
                     'edgar50_co2eq_w_short': 'co2eq_w_short', 'edgar50_co2_w_short': 'co2_w_short'})

data['year_data'] = np.int_(data['year_data'])

vars = ['pop', 'gdp', 'co2']
d18 = data.dropna(subset=vars)
vars15 = ['pop', 'gdp', 'co2', 'ch4', 'n2o', 'co2eq', 'co2eq_w_short']
d15 = data.dropna(subset=vars15)

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2018)

codes15 = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes15['year'] = np.int_(2015)

d18 = pd.merge_asof(codes, d18.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
d15 = pd.merge_asof(codes15, d15.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
countries = pd.read_csv('D:\\projects\\fakta-o-klimatu\\work\\emission-intensity\\countries.csv')
countries = countries.rename(columns={'country_name': 'country', 'final_region': 'cont', 'final_region_en': 'cont_en',
    'second_chart_region': 'region'}).drop(columns=['world_bank_region', 'wiki_region', 'final_region_full'])
countries.show_csv()

d18 = pd.merge(d18, countries, how='inner')
d15 = pd.merge(d15, countries, how='inner')

# full edgar
8710 * 265 + 369341.8 * 28 + 36311982  # 48.96 Gt

# all aviation
0.924726 / 48.96
0.924726 / 36

0.867139 / 49

# SEA + AIR (roughly 60 % sea, 40 % air)
117.4 * 265 + 503 * 28 + 1187011  # 1.23 Gt

0.6 / 50

16.2
11.1 / 16.2

0.6 * 16.2

1.4 / 11.1

194
128
128 / 194
0.6 * 194
(128 - 0.6 * 194) / 128

(194 + 116) / 2

160 / 194


d18.to_parquet('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\d18.parquet')
d15.to_parquet('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\d15.parquet')

d18 = d18.dropna(subset=vars)
d18['year_data'] = np.int_(d18['year_data'])
d15 = d15.dropna(subset=vars15)
d15['year_data'] = np.int_(d15['year_data'])

d15['short_ratio'] = d15['co2eq_w_short'] / d15['co2eq']

d15.show_csv()

d15.co2.sum() / 1e6
d15.co2eq.sum() / 1e6
d15.co2_w_short.sum() / 1e6
d15.co2eq_w_short.sum() / 1e6
d15['pop'].sum() / 1e9

0.6 / 40

d18.show_csv()

d15['pop'].sum()
d18['pop'].sum()

conts = countries[['region', 'cont', 'cont_en']].drop_duplicates().dropna().reset_index(drop=True)
conts.show_csv()

d15.show_csv()

d15_agg = d15.groupby('region')[['pop', 'co2eq', 'gdp']].sum().reset_index()
d15_agg['co2eq_per_pop'] = d15_agg.eval('1000 * co2eq / pop')
d15_agg['co2eq_per_gdp'] = d15_agg.eval('1000000000 * co2eq / gdp')

d18_agg = d18.groupby('region')[['pop', 'co2', 'gdp']].sum().reset_index()
d18_agg['co2_per_pop'] = d18_agg.eval('1000 * co2 / pop')
d18_agg['co2_per_gdp'] = d18_agg.eval('1000000000 * co2 / gdp')

d18_agg.to_csv('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\d18_agg.csv', index=False)
d15_agg.to_csv('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\d15_agg.csv', index=False)


d15['co2eq_per_pop'] = d15.eval('1000 * co2eq / pop')
d15['co2eq_per_gdp'] = d15.eval('1000000000 * co2eq / gdp')

d18['co2_per_pop'] = d18.eval('1000 * co2 / pop')
d18['co2_per_gdp'] = d18.eval('1000000000 * co2 / gdp')

cols = ['code', 'country', 'year', 'year_data', 'pop', 'gdp']
d18[cols + ['co2', 'co2_per_pop', 'co2_per_gdp']] \
    .to_csv('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\d18_countries.csv', index=False)
d15[cols + ['co2eq', 'co2eq_per_pop', 'co2eq_per_gdp']] \
    .to_csv('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\d15_countries.csv', index=False)

d15.query('co2eq_per_pop > 12.2')  # 28th
d18.query('co2_per_pop > 10.5')  # 21st


d15_agg.show_csv()

# PER CAPITA ---
d15_pop = d15_agg.sort_values('co2eq_per_pop', ascending=False).reset_index(drop=True)
d15_pop['start'] = d15_pop['pop'].cumsum().shift(1)
d15_pop.loc[0, 'start'] = 0
d15_pop['end'] = d15_pop['pop'].cumsum()

d18_pop = d18_agg.sort_values('co2_per_pop', ascending=False).reset_index(drop=True)
d18_pop['start'] = d18_pop['pop'].cumsum().shift(1)
d18_pop.loc[0, 'start'] = 0
d18_pop['end'] = d18_pop['pop'].cumsum()

# charts

plt.rcParams['figure.figsize'] = 12, 7

fig15pop, ax15pop = plt.subplots()
patches = []
for i in d15_pop.index:
    plt.text(d15_pop['end'].loc[i] + i * 6e6 + 2e7 - 0.5 * d15_pop['pop'].loc[i], d15_pop['co2eq_per_pop'].loc[i] + 0.4,
             d15_pop['region'].loc[i], rotation=45)
    rec = mpl.patches.Rectangle((d15_pop['start'].loc[i] + i * 6e6, 0), d15_pop['pop'].loc[i],
        d15_pop['co2eq_per_pop'].loc[i])
    patches.append(rec)

ax15pop.add_collection(mpl.collections.PatchCollection(patches))
ax15pop.set(xlim=(0, 8e9), ylim=(0, 36))
ax15pop.set(xlabel='Population', ylabel='t CO2eq per capita', title='CO2eq emissions per capita (2015)')

fig18pop, ax18pop = plt.subplots()
patches = []
for i in d18_pop.index:
    plt.text(d18_pop['end'].loc[i] + i * 6e6 + 2e7 - 0.5 * d18_pop['pop'].loc[i], d18_pop['co2_per_pop'].loc[i] + 0.4,
             d18_pop['region'].loc[i], rotation=45)
    rec = mpl.patches.Rectangle((d18_pop['start'].loc[i] + i * 6e6, 0), d18_pop['pop'].loc[i],
        d18_pop['co2_per_pop'].loc[i])
    patches.append(rec)

ax18pop.add_collection(mpl.collections.PatchCollection(patches))
ax18pop.set(xlim=(0, 8e9), ylim=(0, 36))
ax18pop.set(xlabel='Population', ylabel='t CO2 per capita', title='CO2 emissions per capita (2018)')

# PER GDP ---
d15_gdp = d15_agg.sort_values('co2eq_per_gdp', ascending=False).reset_index(drop=True)
d15_gdp['start'] = d15_gdp['gdp'].cumsum().shift(1)
d15_gdp.loc[0, 'start'] = 0
d15_gdp['end'] = d15_gdp['gdp'].cumsum()
for x in ['gdp', 'start', 'end']:
    d15_gdp[x] = d15_gdp[x] / 1e9

d18_gdp = d18_agg.sort_values('co2_per_gdp', ascending=False).reset_index(drop=True)
d18_gdp['start'] = d18_gdp['gdp'].cumsum().shift(1)
d18_gdp.loc[0, 'start'] = 0
d18_gdp['end'] = d18_gdp['gdp'].cumsum()
for x in ['gdp', 'start', 'end']:
    d18_gdp[x] = d18_gdp[x] / 1e9

d15_gdp.show_csv()

fig15gdp, ax15gdp = plt.subplots()
patches = []
for i in d15_gdp.index:
    plt.text(d15_gdp['end'].loc[i] + i * 2e2 + 5e2 - 0.5 * d15_gdp['gdp'].loc[i], d15_gdp['co2eq_per_gdp'].loc[i] + 0.4,
             d15_gdp['region'].loc[i], rotation=45)
    rec = mpl.patches.Rectangle((d15_gdp['start'].loc[i] + i * 2e2, 0), d15_gdp['gdp'].loc[i],
        d15_gdp['co2eq_per_gdp'].loc[i])
    patches.append(rec)

ax15gdp.add_collection(mpl.collections.PatchCollection(patches))
ax15gdp.set(xlim=(0, 1.2e5), ylim=(0, 1200))
ax15gdp.set(xlabel='GDP PPP (const intl $, billion)', ylabel='g CO2eq per $', title='CO2eq emissions per GDP (2015)')

fig18gdp, ax18gdp = plt.subplots()
patches = []
for i in d18_gdp.index:
    plt.text(d18_gdp['end'].loc[i] + i * 2e2 + 5e2 - 0.5 * d18_gdp['gdp'].loc[i], d18_gdp['co2_per_gdp'].loc[i] + 0.4,
             d18_gdp['region'].loc[i], rotation=45)
    rec = mpl.patches.Rectangle((d18_gdp['start'].loc[i] + i * 2e2, 0), d18_gdp['gdp'].loc[i],
        d18_gdp['co2_per_gdp'].loc[i])
    patches.append(rec)

ax18gdp.add_collection(mpl.collections.PatchCollection(patches))
ax18gdp.set(xlim=(0, 1.3e5), ylim=(0, 1000))
ax18gdp.set(xlabel='GDP PPP (const intl $, billion)', ylabel='g CO2 per ', title='CO2 emissions per GDP (2018)')

# ax18gdp.show()

Chart([ax15pop, ax18pop, ax15gdp, ax18gdp]).show()






fig, ax = plt.subplots()
patches = []

for i in df.index:
    plt.text(df['end'].loc[i] + i * 1e7 + 2e7 - 0.5 * df['pop'].loc[i], df['co2_per_capita'].loc[i] + 0.4,
             df['region'].loc[i], rotation=45)
    rec = mpl.patches.Rectangle((df['start'].loc[i] + i * 1e7, 0), df['pop'].loc[i],
        df['co2_per_capita'].loc[i])
    patches.append(rec)

ax.add_collection(mpl.collections.PatchCollection(patches))
ax.set(xlim=(0, 7.5e9), ylim=(0, 36))
ax.show()





data.show()

no_years = data.groupby('code')['year_data'].count().rename('count').reset_index()
max_pop = data.groupby('code')['pop'].max().reset_index()

pop_years = pd.merge(no_years, max_pop)
pop_years['pop'].sum()  # 7_248_361_589
pop_years[pop_years['count'] < 26]['pop'].sum()  # 139_046_348
pop_years[pop_years['count'] == 26]['pop'].sum()  # 7_109_315_241

pop_years[pop_years['count'] == 23]['pop']
countries.dtypes

countries = pd.merge(countries, pop_years)

countries.final_region.drop_duplicates()

data

regions = pd.merge(data, countries[countries['count'] == 26][['code', 'final_region']])
# regions.final_region.drop_duplicates()
regions.loc[regions.final_region == 'Evropská unie', 'final_region'] = 'Evropa'
regions.loc[regions.final_region == 'Spojené státy americké', 'final_region'] = 'Severní Amerika'
world = regions.drop(columns=['code', 'final_region']).groupby(['year_data']).sum().reset_index()

cze = regions[regions.code == 'CZE'].copy()
cze['final_region'] = 'Česká republika'
regions = pd.concat([regions, cze])
regions = regions.drop(columns=['code']).groupby(['final_region', 'year_data']).sum().reset_index()
# regions.show()

regions['ghg_per_cap'] = 1_000 * regions['co2eq'] / regions['pop']  # t CO2eq / capita
regions['ghg_per_gdp'] = 1_000_000 * regions['co2eq'] / regions['gdp_ppp']  # kg CO2eq / $
regions['gdp_per_cap'] = regions['gdp_ppp'] / regions['pop']
regions['co2eq'] = regions['co2eq'] / 1_000_000  # Gt CO2
regions['gdp_ppp'] = regions['gdp_ppp'] / 1_000_000  # Gt CO2
regions['pop'] = regions['pop'] / 1_000_000_000  # Gt CO2

world['ghg_per_cap'] = 1_000 * world['co2eq'] / world['pop']  # t CO2eq / capita
world['ghg_per_gdp'] = 1_000_000 * world['co2eq'] / world['gdp_ppp']  # kg CO2eq / $
world['gdp_per_cap'] = world['gdp_ppp'] / world['pop']
world['co2eq'] = world['co2eq'] / 1_000_000  # Gt CO2
world['gdp_ppp'] = world['gdp_ppp'] / 1_000_000  # Gt CO2
world['pop'] = world['pop'] / 1_000_000_000  # Gt CO2
world['final_region'] = 'Svět'


titles = {
    'ghg_per_cap': 't CO2eq / person',
    'ghg_per_gdp': 'kg CO2eq / $',
    'gdp_per_cap': '$ / person',
    'pop': 'population (billion)',
    'gdp_ppp': 'GDP (million $)',
    'co2eq': 'Gt CO2eq'
}

plt.rcParams['figure.figsize'] = 12, 7

figs = []
for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2eq']:
    fig, ax = plt.subplots()
    sns.lineplot('year_data', x, data=regions, hue='final_region', marker='o')
    ax.set_title(titles[x] + ' (regions)')
    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    figs.append(fig)

# Chart(figs, cols=2, title='All regions').show()
all_chart = Chart(figs, cols=2, title='All regions')


plt.rcParams['figure.figsize'] = 8, 5
figs = []
for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2eq']:
    fig, ax = plt.subplots()
    sns.lineplot('year_data', x, data=world, marker='o')
    ax.set_title(titles[x] + ' (world)')
    figs.append(fig)

# Chart(figs, cols=3, title='World').show()
world_chart = Chart(figs, cols=3, title='World')


plt.rcParams['figure.figsize'] = 8, 5
charts = []
for r, rdf in regions.groupby('final_region'):
    figs = []
    for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2eq']:
        fig, ax = plt.subplots()
        sns.lineplot('year_data', x, data=rdf, marker='o')
        ax.set_title(titles[x] + f' ({r})')
        figs.append(fig)
    charts.append(Chart(figs, cols=3, title=r))

regions_chart = Selector(charts, title='Per region')
f# regions_chart.show()

rep = Selector([all_chart, world_chart, regions_chart], 'Emissions intensity (2015 update)')
rep.show()


# again, CO2 to 2018 only! ---

data = df[['code', 'year', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'edgar50_co2']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp_ppp',
                     'edgar50_co2': 'co2'})

vars = ['pop', 'gdp_ppp', 'co2']
data = data.dropna(subset=vars)

data['year_data'] = np.int_(data['year_data'])
countries = pd.read_csv('D:\\projects\\fakta-o-klimatu\\work\\emission-intensity\\countries.csv')

no_years = data.groupby('code')['year_data'].count().rename('count').reset_index()
max_pop = data.groupby('code')['pop'].max().reset_index()

pop_years = pd.merge(no_years, max_pop)
pop_years['pop'].sum()  # 7_502_176_200
pop_years[pop_years['count'] < 29]['pop'].sum()  # 225_276_087
pop_years[pop_years['count'] == 29]['pop'].sum()  # 7_276_900_113

countries = pd.merge(countries, pop_years)

regions = pd.merge(data, countries[countries['count'] == 29][['code', 'final_region']])
# regions.final_region.drop_duplicates()
regions.loc[regions.final_region == 'Evropská unie', 'final_region'] = 'Evropa'
regions.loc[regions.final_region == 'Spojené státy americké', 'final_region'] = 'Severní Amerika'
world = regions.drop(columns=['code', 'final_region']).groupby(['year_data']).sum().reset_index()

cze = regions[regions.code == 'CZE'].copy()
cze['final_region'] = 'Česká republika'
regions = pd.concat([regions, cze])
regions = regions.drop(columns=['code']).groupby(['final_region', 'year_data']).sum().reset_index()
# regions.show()

regions['ghg_per_cap'] = 1_000 * regions['co2'] / regions['pop']  # t CO2 / capita
regions['ghg_per_gdp'] = 1_000_000 * regions['co2'] / regions['gdp_ppp']  # kg CO2 / $
regions['gdp_per_cap'] = regions['gdp_ppp'] / regions['pop']
regions['co2'] = regions['co2'] / 1_000_000  # Gt CO2
regions['gdp_ppp'] = regions['gdp_ppp'] / 1_000_000
regions['pop'] = regions['pop'] / 1_000_000_000

world['ghg_per_cap'] = 1_000 * world['co2'] / world['pop']  # t CO2eq / capita
world['ghg_per_gdp'] = 1_000_000 * world['co2'] / world['gdp_ppp']  # kg CO2eq / $
world['gdp_per_cap'] = world['gdp_ppp'] / world['pop']
world['co2'] = world['co2'] / 1_000_000  # Gt CO2
world['gdp_ppp'] = world['gdp_ppp'] / 1_000_000  # Gt CO2
world['pop'] = world['pop'] / 1_000_000_000  # Gt CO2
world['final_region'] = 'Svět'


titles = {
    'ghg_per_cap': 't CO2 / person',
    'ghg_per_gdp': 'kg CO2 / $',
    'gdp_per_cap': '$ / person',
    'pop': 'population (billion)',
    'gdp_ppp': 'GDP (million $)',
    'co2': 'Gt CO2'
}

plt.rcParams['figure.figsize'] = 12, 7

figs = []
for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2']:
    fig, ax = plt.subplots()
    sns.lineplot('year_data', x, data=regions, hue='final_region', marker='o')
    ax.set_title(titles[x] + ' (regions)')
    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    figs.append(fig)

# Chart(figs, cols=2, title='All regions').show()
all_chart = Chart(figs, cols=2, title='All regions')


plt.rcParams['figure.figsize'] = 8, 5
figs = []
for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2']:
    fig, ax = plt.subplots()
    sns.lineplot('year_data', x, data=world, marker='o')
    ax.set_title(titles[x] + ' (world)')
    figs.append(fig)

# Chart(figs, cols=3, title='World').show()
world_chart = Chart(figs, cols=3, title='World')


plt.rcParams['figure.figsize'] = 8, 5
charts = []
for r, rdf in regions.groupby('final_region'):
    figs = []
    for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2']:
        fig, ax = plt.subplots()
        sns.lineplot('year_data', x, data=rdf, marker='o')
        ax.set_title(titles[x] + f' ({r})')
        figs.append(fig)
    charts.append(Chart(figs, cols=3, title=r))

regions_chart = Selector(charts, title='Per region')
# regions_chart.show()

rep = Selector([all_chart, world_chart, regions_chart], 'Emissions intensity (CO2 only, 2018 update)')
rep.show()




