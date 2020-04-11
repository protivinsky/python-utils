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

import world_bank_data as wb

wb.get_topics()
wb.get_sources().show_csv()
wb.get_indicators(topic=19).show_csv()  # this is pretty cool!
wb.get_indicators(topic=3).show_csv()

# EN.ATM.GHGT.KT.CE	Total greenhouse gas emissions (kt of CO2 equivalent)
# EN.ATM.CO2E.KD.GD	CO2 emissions (kg per 2010 US$ of GDP)
# EN.ATM.CO2E.KT	CO2 emissions (kt)
# EN.ATM.CO2E.PC	CO2 emissions (metric tons per capita)
# EN.ATM.CO2E.PP.GD	CO2 emissions (kg per PPP $ of GDP)
# EN.ATM.CO2E.PP.GD.KD	CO2 emissions (kg per 2011 PPP $ of GDP)
# EN.CLC.GHGR.MT.CE	GHG net emissions/removals by LUCF (Mt of CO2 equivalent)

# SP.POP.TOTL	Population, total

# NY.GDP.MKTP.CD	GDP (current US$)
# NY.GDP.MKTP.KD	GDP (constant 2010 US$)
# NY.GDP.MKTP.PP.CD	GDP, PPP (current international $)
# NY.GDP.MKTP.PP.KD	GDP, PPP (constant 2011 international $)
# NY.GDP.PCAP.CD	GDP per capita (current US$)
# NY.GDP.PCAP.KD	GDP per capita (constant 2010 US$)
# NY.GDP.PCAP.PP.CD	GDP per capita, PPP (current international $)
# NY.GDP.PCAP.PP.KD	GDP per capita, PPP (constant 2011 international $)

wb.get_countries().show()
wb.get_regions().show()
wb.get_series('SP.POP.TOTL', id_or_value='id')
wb.get_series('SP.POP.TOTL').reset_index()

# looks simple - so I need:
#   - GCA country to WB code conversion
#   - then I can just join everything and I should have all available years, so should be able to do ASOF over countries

# country mapping
root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
path_gca = root + '\\global-carbon-atlas\\export_20190819_2250.csv'
country_map = pd.read_csv(root + '\\country_mapping.csv')[['wb', 'gca']]

country_map = pd.merge(country_map, wb.get_countries()['name'].rename('wb').reset_index())
country_map.show_csv()
country_map = country_map.rename(columns={'id': 'code'})
country_map.to_csv(root + '\\country_map.csv')

# ready to merge everything ---
country_map = pd.read_csv(root + '\\country_map.csv')
country_map['gca'] = country_map['gca'].fillna('')

gca = pd.read_csv(path_gca, skiprows=1, sep=';', skipfooter=2, engine='python')
gca = gca.rename(columns={'Unnamed: 0': 'year'})
gca = gca.set_index('year')
gca.columns = gca.columns.rename('gca')
gca = gca.stack().rename('co2_emissions').reset_index()
gca = pd.merge(gca, country_map[['gca', 'code']])
gca.dtypes  # this should be good enough to merge it with WB data

# construct big wb datasets with all indicators I need
indicators = ['EN.ATM.GHGT.KT.CE', 'EN.ATM.CO2E.KD.GD', 'EN.ATM.CO2E.KT', 'EN.ATM.CO2E.PC', 'EN.ATM.CO2E.PP.GD',
              'EN.ATM.CO2E.PP.GD.KD', 'EN.CLC.GHGR.MT.CE', 'SP.POP.TOTL', 'NY.GDP.MKTP.CD', 'NY.GDP.MKTP.KD',
              'NY.GDP.MKTP.PP.CD', 'NY.GDP.MKTP.PP.KD', 'NY.GDP.PCAP.CD', 'NY.GDP.PCAP.KD', 'NY.GDP.PCAP.PP.CD',
              'NY.GDP.PCAP.PP.KD']

df = wb.get_series(indicators[0], id_or_value='id').reset_index().drop(columns='Series')
for ind in indicators[1:]:
    df = pd.merge(df, wb.get_series(ind, id_or_value='id').reset_index().drop(columns='Series'))

df.to_parquet(root + '\\wb_raw.parquet')
df = df.rename(columns={'Country': 'code', 'Year': 'year'})
df['year'] = np.int_(df['year'])
df.dtypes
gca.dtypes
df = pd.merge(df, gca.drop(columns=['gca']), how='left')
df = df.rename(columns={'co2_emissions': 'gca_co2'})

df.to_parquet(root + '\\wb_with_gca.parquet')

df = pd.read_parquet(root + '\\wb_with_gca.parquet')

# I need to create some countries metadata ---

codes = df['code'].unique()
countries = wb.get_countries()
countries = countries[countries['region'] != 'Aggregates']
countries = countries.drop(columns=['adminregion', 'incomeLevel', 'lendingType', 'capitalCity'])
countries = countries.reset_index().rename(columns={'id': 'code', 'iso2Code': 'iso'})
countries.show_csv()

countries = countries.rename(columns={'name': 'wb_name'})
pop = df[df['year'] == 2018][['code', 'SP.POP.TOTL']].rename(columns={'SP.POP.TOTL': 'population'}) \
    .reset_index(drop=True)
countries = pd.merge(countries, pop, how='left')

countries.to_parquet(root + '\\countries_raw.parquet')
countries.to_csv(root + '\\countries_raw.csv', index=False)
# ok, sensible starting point:
#   - look if you can get better names based on iso
#   - and have a think about sensible groups

ciselnik = pd.read_csv(root + '\\ciselnik_zemi.csv')
ciselnik = ciselnik[ciselnik.columns[:7]]
ciselnik = ciselnik.iloc[:250]
ciselnik.cz_full
ciselnik.dtypes
countries.dtypes
ciselnik.show_csv()

countries = pd.merge(ciselnik.rename(columns={'iso3': 'code'}), countries.drop(columns=['iso']), how='left')
countries.show_csv()

countries['iso_numeric'] = np.int_(countries['iso_numeric'])

wiki = pd.read_csv(root + '\\wiki_countries.csv')
countries = pd.merge(countries, wiki[['wiki_country', 'region']].rename(columns={'wiki_country': 'wb_name',
    'region': 'wiki_region'}), how='left')
countries = pd.merge(countries, pd.DataFrame({'edgar_name': edgar_countries}), left_on='en_short',
                     right_on='edgar_name', how='left')
countries.sort_values('en_short').to_csv(root + '\\countries_new.csv', encoding='utf-8-sig', index=False)

# now it should be fully fixed
countries = pd.read_csv(root + '\\countries_new.csv')

# check edgar ---
# join czech regions ---
# and try some plotting ---

len(edgar_countries.unique())
countries['edgar_name'] = countries['edgar_name'].fillna('')
len(countries[countries['edgar_name'] != '']['edgar_name'].unique())
countries.groupby('edgar_name')[['code']].count().sort_values('code', ascending=False)

edgar_countries[~edgar_countries.isin(countries['edgar_name'])]
pd.DataFrame({'edgar': edgar_countries}).sort_values('edgar').show_csv()
# yeah, there was one bug, fixed now

countries = pd.read_csv(root + '\\countries_new.csv')
countries = countries.rename(columns={'region': 'wb_region', 'wiki_region': 'en_region'})
regions = countries[['en_region']].drop_duplicates().sort_values('en_region').reset_index(drop=True).iloc[:6]
regions['cz_region'] = ['Afrika', 'Arabské státy', 'Asie a Oceánie', 'Evropa', 'Severní Amerika', 'Jižní Amerika']
countries = pd.merge(countries, regions, how='left')
countries.to_csv(root + '\\countries_new.csv', encoding='utf-8-sig', index=False)

countries.to_csv(root + '\\countries.csv', encoding='utf-8-sig', index=False)

# ok, now I have fixed countries


# load EDGAR and join the countries
path_edgar = root + '\\edgar\\EDGARv5.0_FT2017_fossil_CO2_booklet2018.xls'
edgar = pd.read_excel(path_edgar, sheet_name='fossil_CO2_totals_by_country', skipfooter=5)

edgar_countries = edgar['country_name']
iso_countries = countries['en_short']
wb_countries = countries['wb_name']

import nltk

dist = np.empty(shape=(len(edgar_countries), len(wb_countries)), dtype=np.int_)
for i in range(len(edgar_countries)):
    for j in range(len(wb_countries)):
        dist[i, j] = nltk.edit_distance(edgar_countries[i], wb_countries[j])

country_map = pd.DataFrame({'edgar': edgar_countries.values, 'wb': wb_countries[np.argmin(dist, axis=1)].values})
country_map.show_csv()

pd.DataFrame({'wb': wb_countries}).show_csv()


df.dtypes
# 'EN.ATM.GHGT.KT.CE'
#   - this is messy, try it again with CO2 only
data = df[['code', 'year', 'gca_co2', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD']] \
    .rename(columns={'year': 'year_data', 'gca_co2': 'co2', 'SP.POP.TOTL': 'pop', 'NY.GDP.MKTP.PP.KD': 'gdp'})
data['co2_per_capita'] = 1_000_000 * data['co2'] / data['pop']  # CO2eq tonnes per capita
data['co2_per_dollar'] = 1_000_000_000 * data['co2'] / data['gdp']  # CO2eq kg per dollar
data['co2_Mt'] = data['co2']
data = data.dropna(subset=['co2'])

codes = pd.DataFrame({'code': np.sort(data['code'].unique())})
codes['year'] = np.int_(2017)

res = pd.merge_asof(codes, data.drop(columns=['co2']).sort_values('year_data'), by='code', left_on='year',
                    right_on='year_data')
res = pd.merge(res, countries[['code', 'cz_short', 'cz_region']])
res.show_csv()
res.to_csv(root + '\\emissions.csv', encoding='utf-8-sig', index=False)
# check outliers, maybe describe better cols ---

df[(df['code'] == 'CAF') & (df['year'] == 2012)].T
df[(df['code'] == 'CAF')]['EN.ATM.GHGT.KT.CE']


sns.scatterplot('EN.ATM.CO2E.KT', 'EN.ATM.GHGT.KT.CE', data=df[df['year'] == 2012]).show()

df12 = df[df['year'] == 2012].copy()
df12['ratio'] = df12['EN.ATM.GHGT.KT.CE'] / df12['EN.ATM.CO2E.KT']

sns.distplot(df12[np.isfinite(df12['ratio'])]['ratio'], kde=False).show()

pd.merge(df12[df12['ratio'] > 10], countries[['code', 'cz_short', 'en_short', 'en_region']]).show_csv()

df[(df['year'] == 2010) & (df['code'] == 'CAF')].T


# basic data checks ---
plots = []
for y in range(2000, 2018):
    logger(y)
    fig, ax = plt.subplots()
    sns.scatterplot('EN.ATM.CO2E.KT', 'gca_co2', data=df[df['year'] == y])
    ax.set_title(str(y))
    plots.append(fig)

Chart(plots, title='CO2 - WB vs GCA').show()  # ok, data are almost identical

plots = []
for y in range(2000, 2018):
    logger(y)
    fig, ax = plt.subplots()
    sns.scatterplot('EN.ATM.CO2E.KT', 'EN.ATM.GHGT.KT.CE', data=df[df['year'] == y])
    ax.set_title(str(y))
    plots.append(fig)

Chart(plots, title='WB - CO2 vs GHG').show()  # largely similar, however there is some noise



wb.get_series('SP.POP.TOTL', id_or_value='id')


gca[['gca']].drop_duplicates().sort_values('gca').show_csv()
len(np.sort(pd.merge(gca, country_map[['gca', 'code']])['code'].unique()))

(country_map['gca'].drop_duplicates() != '').sum()
country_map['gca'].fillna('')

country_map.groupby('gca')['code'].count().reset_index().sort_values('code', ascending=False)

gca_countries = country_map['gca'][country_map['gca'] != ''].reset_index(drop=True)
gca_countries2 = np.sort(pd.merge(gca, country_map[['gca', 'code']])['gca'].unique())
len(gca_countries2)
pd.DataFrame({'countries1': np.sort(gca_countries.values)}).show_csv()
pd.DataFrame({'countries2': np.sort(gca_countries2)}).show_csv()

wb.get_countries()['name'].rename('wb').reset_index().show_csv()

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



