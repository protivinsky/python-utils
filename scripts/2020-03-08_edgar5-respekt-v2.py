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


root = 'D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\data'
edgar_files = ['CH4', 'CO2_excl_short-cycle_org_C', 'CO2_org_short-cycle_C', 'N2O']
ef = edgar_files[0]

data = []

for ef in edgar_files:
    logger(ef)
    ey = 2018 if ef == 'CO2_excl_short-cycle_org_C' else 2015
    frame = pd.read_excel(f'{root}\\edgar_v5.0\\v50_{ef}_1970_{ey}.xls', sheet_name='TOTALS BY COUNTRY',
                          header=9)
    frame = frame[['ISO_A3'] + list(range(1970, ey + 1))].rename(columns={'ISO_A3': 'code'}).set_index('code')
    data.append(frame)
    # data.append(np.sum(frame.T, axis=1).rename(ef))

df = data[1]

# so here I have edgar CO2 up to 2018
# what do I want to do with it?

countries = pd.read_csv('D:\\projects\\fakta-o-klimatu\\work\\emission-intensity\\countries.csv')
countries.show()
countries = countries.rename(columns={'country_name': 'country', 'final_region': 'cont', 'final_region_en': 'cont_en',
    'second_chart_region': 'region'}).drop(columns=['world_bank_region', 'wiki_region', 'final_region_full'])

regions = countries[['code', 'final_region']].rename(columns={'final_region': 'region'})
selected = ['Čína', 'Evropská unie', 'Indie', 'Rusko', 'Spojené státy americké']
regions = regions[regions.region.isin(selected)].copy()
# what about Great Britain?
regions = regions.query('code != "GBR"').reset_index(drop=True).copy()
regions.shape
regions.show()

df = pd.merge(regions, df.reset_index())
df.show()

cze = df.iloc[[0]].copy()
cze.loc[0, 'region'] = 'Česká republika'

co2 = pd.concat([df, cze]).drop(columns=['code']).set_index('region').groupby('region').apply(lambda x: x.sum(axis=0)) \
    .sort_index()

import world_bank_data as wb
pop = wb.get_series('SP.POP.TOTL', id_or_value='id')
pop = pop.unstack().reset_index().drop(columns=['Series']).rename(columns={'Country': 'code'})
pop = pd.merge(regions, pop)
pop.show()
cze = pop.query('code == "CZE"').copy()
cze.loc[0, 'region'] = 'Česká republika'

pop = pd.concat([pop, cze]).drop(columns=['code']).set_index('region').groupby('region').apply(lambda x: x.sum(axis=0))
pop = pop[[str(i) for i in range(1970, 2019)]].sort_index()

pop.columns = [int(i) for i in pop.columns]

co2_per_capita = 1e3 * co2 / pop
co2 = co2 / 1e6

covered = co2[2018].sum()
world = data[1][2018].sum()

covered / world

output = 'D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\regiony\\'

co2.to_csv(output + 'regiony_co2.csv')
pop.to_csv(output + 'regiony_pop.csv')
co2_per_capita.to_csv(output + 'regiony_co2_per_capita.csv')


df = pd.merge(regions, df.reset_index(), how='right')
df['region'] = df['region'].fillna('Ostatní')
co2 = df.drop(columns=['code']).set_index('region').groupby('region').apply(lambda x: x.sum(axis=0)).sort_index()
co2 = co2 / 1e6

ax = co2.T.plot.area(lw=0)
ax.set(xlabel='Rok', ylabel='Gt CO2', title='Světové roční emise CO2')
ax.show()

co2.to_csv(output + 'regiony_co2.csv')


