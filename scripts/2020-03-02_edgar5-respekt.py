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

df = pd.DataFrame(data).T.loc[:2015]
df['co2eq'] = (df['CO2_excl_short-cycle_org_C'] + df['CH4'] * 28 + df['N2O'] * 265) / 1e6

import world_bank_data as wb
pop = wb.get_series('SP.POP.TOTL').loc['World', :, :].reset_index()
pop['Year'] = np.int_(pop.Year)
pop = pop.set_index('Year').loc[1970:2018].drop(columns='Series').reset_index()

df = pd.merge(df, pop.set_index('Year'), left_index=True, right_index=True)
df['co2eq_per_capita'] = 1e9 * df['co2eq'] / df['SP.POP.TOTL']

df['co2_per_capita'] = 1e9 * df['CO2_excl_short-cycle_org_C'] / df['SP.POP.TOTL']
df = df.reset_index().rename(columns={'SP.POP.TOTL': 'population', 'index': 'year',
    'CO2_excl_short-cycle_org_C': 'co2'})[['year', 'population', 'co2', 'co2_per_capita']]
df['co2'] = df['co2'] / 1e6
df['co2_per_capita'] = df['co2_per_capita'] / 1e6
df.to_csv('D:\\projects\\fakta-o-klimatu\\work\\respekt-data\\emise_co2_1970-2018.csv', index=False)

plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['figure.subplot.top'] = 0.94

plt.subplots()
ax_ghg = sns.lineplot(df['year'], df['co2'], marker='o')
ax_ghg.set(xlabel='Rok', ylabel='Gt CO2')
ax_ghg.set_title('Světové emise 1970-2018')
ax_ghg.show()

plt.subplots()
ax_pop = sns.lineplot(pop['Year'], pop['SP.POP.TOTL'] / 1e9, marker='o', color='firebrick')
ax_pop.set(xlabel='Rok', ylabel='Populace (mld)')
ax_pop.set_title('Světová populace 1970-2015')

plt.subplots()
ax_pc = sns.lineplot(df['year'], df['co2eq_per_capita'], marker='o', color='darkolivegreen')
ax_pc.set(xlabel='Rok', ylabel='t CO2eq na osobu')
ax_pc.set_title('Roční emise na osobu 1970-2015')

Chart([ax_ghg, ax_pop, ax_pc], title='Světové emise 1970-2015').show()

0.924 / 49

plt.subplots()
ax_pc = sns.lineplot(df['year'], df['co2_per_capita'], marker='o', color='darkolivegreen')
ax_pc.set(xlabel='Rok', ylabel='t CO2 na osobu')
ax_pc.set_title('Roční emise na osobu 1970-2018')
ax_pc.show()


# select EU and some other players




