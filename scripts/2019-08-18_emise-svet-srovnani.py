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

root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
path_wb = root + '\\worldbank\\API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_49299.csv'
path_wb_countries = root + '\\worldbank\\Metadata_Country_API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_49299.csv'
path_oecd = root + '\\oecd\\AIR_GHG_19082019230046941.csv'
path_gca = root + '\\global-carbon-atlas\\export_20190819_2250.csv'

# data loading and sanitization
wb = pd.read_csv(path_wb, skiprows=4)
wb.show()
wb.dtypes
wb.columns
np.sum(np.isfinite(wb['Unnamed: 63']))  # 0

# valid years
years = [y for y in range(1960, 2019) if np.sum(np.isfinite(wb[str(y)]))]
wb['Indicator Name'].unique()  # 'Total greenhouse gas emissions (kt of CO2 equivalent)'
wb['Indicator Code'].unique()  # 'EN.ATM.GHGT.KT.CE'

wb = wb.rename(columns={'Country Name': 'country', 'Country Code': 'code'})
wb = wb[['country', 'code'] + [str(y) for y in years]].copy()

# great, now I just have to figure out country / region structure
wb_countries = pd.read_csv(path_wb_countries)
wb_countries.dtypes

cols = {'Country Code': 'code', 'Region': 'region', 'IncomeGroup': 'income_group', 'SpecialNotes': 'notes',
        'TableName': 'name'}

wb_countries = wb_countries[[c for c in cols]].rename(columns=cols)
wb_countries['is_country'] = ~pd.isna(wb_countries.region)

wb = pd.merge(wb, wb_countries[['code', 'is_country']])

wb[wb['is_country']]['2012'].sum()
wb.set_index('code').loc['WLD', '2012']

# sum check
for y in range(2000, 2013):
    total = wb[wb['is_country']][str(y)].sum()
    wld = wb.set_index('code').loc['WLD', str(y)]
    print('WLD = {:.4g}, TOTAL = {:.4g}'.format(wld, total))

# WLD is bigger everywhere, by about 1.5-3 M kT

# check income groups and regions

wb = pd.merge(wb, wb_countries[['code', 'income_group', 'region']])

# ok, this is almost exact
regions = wb[wb['is_country']].groupby('region')['2012'].sum()
wb.set_index('country').loc[regions.index, '2012']

regions = wb[wb['is_country']].groupby('region')['2011'].sum()
wb.set_index('country').loc[regions.index, '2011']

# but world is somewhat larger
wb.set_index('country').loc[regions.index, '2012'].sum()
wb.set_index('code').loc['WLD', '2012']

regions = wb[wb['is_country']].groupby('income_group')['2012'].sum()
wb.set_index('country').loc[regions.index, '2012']

# REGIONS, 2012 ---
# IN DATASET:
# East Asia & Pacific           1.882291e+07
# Europe & Central Asia         9.398207e+06
# Latin America & Caribbean     5.746908e+06
# Middle East & North Africa    1.464935e+06
# North America                 7.371537e+06
# South Asia                    3.648610e+06
# Sub-Saharan Africa            4.601155e+06
#
# AGGREGATED:
# East Asia & Pacific           1.849706e+07
# Europe & Central Asia         9.398207e+06
# Latin America & Caribbean     5.746908e+06
# Middle East & North Africa    1.464935e+06
# North America                 7.371537e+06
# South Asia                    3.648610e+06
# Sub-Saharan Africa            4.601155e+06

# INCOME GROUPS, 2012 ---
# IN DATASET:
# High income            1.618139e+07
# Low income             3.115755e+06
# Lower middle income    8.983602e+06
# Upper middle income    2.277352e+07
#
# AGGREGATED:
# High income            1.585554e+07
# Low income             3.115755e+06
# Lower middle income    8.983602e+06
# Upper middle income    2.277352e+07

# The difference is about 3e5 kT CO2 in both cases, in high income East Asia & Pacific.
# Some of groups is smaller than total world for both.

np.sum(wb['is_country'])  # 217 countries
# can I get some official WB list of countries and codes?
# https://wits.worldbank.org/wits/wits/witshelp/content/codes/country_codes.htm

wb_codes = pd.read_csv(root + '\\worldbank\\country-codes.csv').iloc[:-1]
wb_codes

co2_countries = wb['code'].values

wb_codes[~wb_codes['code'].isin(co2_countries)]
# ok, for instance Taiwan and Romania are not included
#   - actually, Romania is included, only with different code
#   - based on GCA, Taiwan is about 2.7e5 kT CO2 -- that sounds about right (and GCA is underestimate, as it is
#       CO2 only)

# can I explain the world difference?
# sum check
for y in range(2000, 2013):
    total = wb[wb['is_country']][str(y)].sum()
    wld = wb.set_index('code').loc['WLD', str(y)]
    print('WLD = {:.4g}, TOTAL = {:.4g}'.format(wld, total))

y = 2011
diff = wb.set_index('code').loc['WLD', str(y)] - wb[wb['is_country']][str(y)].sum()
wb[np.abs(wb[str(y)] - diff) < 1e5][['country', 'code', str(y)]]

wb[['country', 'code', '2012']].show()


wb[~wb['is_country']][['country', 'code', '2012']].show()

foo = ['CEB', 'OSS', 'SST', 'MNA']
wb[wb['code'].isin(foo)][str(y)].sum()

# is there a country not having recent data? but shouldn't that mess also regions?
# yes, unless they were just plainly computed

wb[wb['is_country'] & (pd.isna(wb['2012']))]

inner = wb[['code'] + [str(y) for y in years]].set_index('code').T.fillna(method='ffill').T
inner = pd.merge(inner.reset_index(), wb_countries[['code', 'is_country']])
inner[inner['is_country']]['2012'].sum()


for y in range(2000, 2013):
    total = inner[inner['is_country']][str(y)].sum()
    wld = inner.set_index('code').loc['WLD', str(y)]
    print('WLD = {:.4g}, TOTAL = {:.4g}'.format(wld, total))

inner

gca = pd.read_csv(path_gca, skiprows=1, sep=';')
gca.show_csv()
gca = gca.rename(columns={'Unnamed: 0': 'year'})

gca_countries = gca.columns[1:]
wb_countries.dtypes

wb_countries_only = wb_countries[wb_countries['is_country']][['code', 'name']]


import nltk

wb_ = np.sort(wb_countries_only['name'].values)
gca_ = np.sort(gca_countries.values)

wb_[0]
gca_[0]

nltk.edit_distance(wb_[0], gca_[1])

# this is not superfast, but doable
dist = np.empty(shape=(len(wb_), len(gca_)), dtype=np.int_)
for i in range(len(wb_)):
    for j in range(len(gca_)):
        dist[i, j] = nltk.edit_distance(wb_[i], gca_[j])

country_map = pd.DataFrame({'wb': wb_, 'gca': gca_[np.argmin(dist, axis=1)]})
country_map.show_csv()

country_map = pd.read_csv(root + '\\country_mapping.csv')[['wb', 'gca']]
country_map = pd.merge(wb_countries_only.rename(columns={'name': 'wb'}), country_map)

gca17 = gca.set_index('year').T.reset_index()[['index', '2017']] \
    .rename(columns={'index': 'gca', '2017': 'gca_emission'})

emi = pd.merge(country_map, inner[['code', '2012']].rename(columns={'2012': 'wb_emission'}))
emi = pd.merge(emi, gca17)

fig, ax = plt.subplots()
sns.scatterplot(x='wb_emission', y='gca_emission', data=emi)
ax.set(xscale='log', yscale='log')
ax.show()

# fairly nice - get population and GDP data, join it and see what's coming out of it

mapping = {}
for i in range(len(wb_)):
    mapping[wb_[i]] = dist

np.idxmin


path = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani' \
    '\\data\\API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_49299.csv'

df = pd.read_csv(path, skiprows=4)
df.dtypes



