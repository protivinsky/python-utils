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


# load new EDGAR v5.0 data ---

root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
edgar_files = ['CH4', 'CO2_excl_short-cycle_org_C', 'CO2_org_short-cycle_C', 'N2O']
ef = edgar_files[0]

edgar_df = None

for ef in edgar_files:
    logger(ef)
    ey = 2018 if ef == 'CO2_excl_short-cycle_org_C' else 2015
    frame = pd.read_excel(f'{root}\\edgar_v5.0\\v50_{ef}_1970_{ey}.xls', sheet_name='TOTALS BY COUNTRY',
                          header=9)
    frame = frame[['ISO_A3'] + list(range(1970, ey + 1))].rename(columns={'ISO_A3': 'code'}).set_index('code')
    frame.columns = frame.columns.rename('year')
    frame = frame.unstack().rename(f'edgar50_{ef}').reset_index()
    frame = frame[~frame['code'].isin(['SEA', 'AIR'])]
    if edgar_df is None:
        edgar_df = frame
    else:
        edgar_df = pd.merge(edgar_df, frame, how='outer')

edgar_df.to_csv(root + '\\edgar_v5.0.csv', index=False)
edgar_df.show()

data = edgar_df.copy()

# find sensible GDP vs population vs CO2eq (or CO2) data vs time ?

root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
df = pd.read_csv(root + '\\data_all.csv')
df.show_csv()
df.query('code == "CZE"').show_csv()

df = pd.merge(df, edgar_df, how='left', on=['code', 'year'])
df.to_csv(f'{root}\\data_all_w_edgar50.csv', index=False)
df = pd.read_csv(f'{root}\\data_all_w_edgar50.csv')


df['edgar432_co2'] = df['edgar432_CO2_excl_short-cycle_org_C']
df['edgar432_co2_w_short'] = df['edgar432_co2'] + df['edgar432_CO2_org_short-cycle_C']
# actually, these are old sensitivities!
df['edgar432_co2eq'] = df['edgar432_CO2_excl_short-cycle_org_C'] + 25 * df['edgar432_CH4'] + 298 * df['edgar432_N2O']
df['edgar432_co2eq_w_short'] = df['edgar432_co2eq'] + df['edgar432_CO2_org_short-cycle_C']

df['edgar50_co2'] = df['edgar50_CO2_excl_short-cycle_org_C']
df['edgar50_co2_w_short'] = df['edgar50_co2'] + df['edgar50_CO2_org_short-cycle_C']
# I am using the new sensitivities here, 28 and 265
df['edgar50_co2eq'] = df['edgar50_CO2_excl_short-cycle_org_C'] + 28 * df['edgar50_CH4'] + 265 * df['edgar50_N2O']
df['edgar50_co2eq_w_short'] = df['edgar50_co2eq'] + df['edgar50_CO2_org_short-cycle_C']


data = df[['code', 'year', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'edgar50_co2eq']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp_ppp',
                     'edgar50_co2eq': 'co2eq'})

data

sns.lineplot(x='year_data', y='co2eq', data=data, units='code', estimator=None).show()
sns.lineplot(x='year_data', y='pop', data=data, units='code', estimator=None).show()
sns.lineplot(x='year_data', y='gdp_ppp', data=data, units='code', estimator=None).show()

vars = ['pop', 'gdp_ppp', 'co2eq']
data = data.dropna(subset=vars)

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2012)
data['year_data'] = np.int_(data['year_data'])

res = pd.merge_asof(codes, data.sort_values('year_data'), by='code', left_on='year', right_on='year_data')
res = pd.merge(res, countries[['code', 'en_short', 'en_region', 'cz_short', 'cz_region', 'en_category', 'cz_category',
                               'cz_cat_desc']])

df.dtypes
data

countries = pd.read_csv('D:\\projects\\fakta-o-klimatu\\work\\emission-intensity\\countries.csv')
countries.show()

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




