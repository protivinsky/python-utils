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

# find sensible GDP vs population vs CO2eq (or CO2) data vs time ?

root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
df = pd.read_csv(root + '\\data_all.csv')
df.show_csv()



df['edgar432_co2'] = df['edgar432_CO2_excl_short-cycle_org_C']
df['edgar432_co2_w_short'] = df['edgar432_co2'] + df['edgar432_CO2_org_short-cycle_C']
df['edgar432_co2eq'] = df['edgar432_CO2_excl_short-cycle_org_C'] + 25 * df['edgar432_CH4'] + 298 * df['edgar432_N2O']
df['edgar432_co2eq_w_short'] = df['edgar432_co2eq'] + df['edgar432_CO2_org_short-cycle_C']

data = df[['code', 'year', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'edgar432_co2eq']] \
    .rename(columns={'year': 'year_data', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp_ppp',
                     'edgar432_co2eq': 'co2eq'})

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
pop_years['pop'].sum()  # 6_994_758_962
pop_years[pop_years['count'] < 23]['pop'].sum()  # 103_474_221
pop_years[pop_years['count'] == 23]['pop'].sum()  # 6_891_284_741

pop_years[pop_years['count'] == 23]['pop']
countries.dtypes

countries = pd.merge(countries, pop_years)

countries.final_region.drop_duplicates()

data

regions = pd.merge(data, countries[countries['count'] == 23][['code', 'final_region']])
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

Chart(figs, cols=2, title='All regions').show()
all_chart = Chart(figs, cols=2, title='All regions')


plt.rcParams['figure.figsize'] = 8, 5
figs = []
for x in ['ghg_per_cap', 'ghg_per_gdp', 'gdp_per_cap', 'pop', 'gdp_ppp', 'co2eq']:
    fig, ax = plt.subplots()
    sns.lineplot('year_data', x, data=world, marker='o')
    ax.set_title(titles[x] + ' (world)')
    figs.append(fig)

Chart(figs, cols=3, title='World').show()

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
regions_chart.show()

rep = Selector([all_chart, world_chart, regions_chart], 'Emissions intensity')
rep.show()


Chart(figs, cols=3, title='World').show()

world_chart.show()
all_chart.show()





ghg_per_cap = sns.lineplot('year_data', 'g')





