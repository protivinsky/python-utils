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


# --- CHARTS GENERATION ---
# Can I generate the usual charts from Python?

root = 'D:\\projects\\fakta-o-klimatu\\work\\423-emisni-intenzita'
colors = {
    'Asie': ('#c32b2a', '#f17073', '#da7c7b', 'f6a6a8'),
    'Severní Amerika': ('#2d2e73', '#3e4ea0', '#7d7ea8', '#8791c4'),
    'Evropa': ('#562D84', '#7666AC', '#967db3', '#aaa0cc'),
    'Jižní a Střední Amerika': ('#4591CE', '#4AAEE2', '#8cbbe1', '#8fcded'),
    'Afrika': ('#CE8529', '#F7C653', '#e1b37a', '#fadc94'),
    'Rusko': ('#49BFB5', '#', '#8ed7d1', '#'),
    'Austrálie a Nový Zéland': ('#0E9487', '#', '#6abdb5', '#')
}

colors['Afrika'][2]
conts = list(colors.keys())

plt.rcParams['figure.figsize'] = 12.5, 7
plt.rcParams['hatch.linewidth'] = 0.8

plt.rcParams['figure.figsize'] = 16, 9
plt.rcParams['hatch.linewidth'] = 0.8

df = pd.read_csv(root + '\\data.csv', decimal=',')
df.dtypes
df.show()
type(df['gdp'].iloc[3])


df['pop'] = np.float_(df['pop'].apply(lambda x: x.replace('\xa0', '')))
df['gdp'] = np.float_(df['gdp'].apply(lambda x: x.replace('\xa0', '')))
df['gdp_per_capita'] = np.float_(df['gdp_per_capita'].apply(lambda x: x.replace('\xa0', '')))
df['ghg_per_gdp'] = np.float_(df['ghg_per_gdp'])

df.dtypes
df.show()


# 1. emise svet na osobu ---
df1 = pd.read_csv(root + '\\data1.csv', decimal=',')

df1['pop'] = np.float_(df1['pop'].apply(lambda x: x.replace('\xa0', '')))
df1['gdp'] = np.float_(df1['gdp'].apply(lambda x: x.replace('\xa0', '')))
df1['gdp_per_capita'] = np.float_(df1['gdp_per_capita'].apply(lambda x: x.replace('\xa0', '')))
df1['ghg_per_gdp'] = np.float_(df1['ghg_per_gdp'])

gdf1 = df1.groupby('cont')

sns.scatterplot('gdp_per_capita', 'ghg_per_capita', data=df1, hue='cont').show()

fig, ax = plt.subplots()
# c = 'Asie'
x = 0
xgap = 1e7
ygap = 0.4

for c in conts:
    cdf = df1.loc[gdf1.groups[c]].sort_values('ghg_per_capita', ascending=False).reset_index(drop=True)
    cs = colors[c]
    for i, row in cdf.iterrows():
        x = x + xgap
        plt.text(x + 0.5 * row['pop'] - 4 * xgap, row['ghg_per_capita'] + ygap, row['region'], rotation=45, fontsize=9)
        rec = mpl.patches.Rectangle((x, 0), row['pop'], row['ghg_per_capita'], ec=None, fc=cs[i % 2])
        ax.add_patch(rec)
        x = x + row['pop']
    x = x + 20 * xgap

ax.set(xlim=(0, 9.4e9), ylim=(0, 38))
ax.set(xlabel='Population', ylabel='t CO2eq per capita', title='CO2eq emissions per capita (2015)')
ax.show()

fig_per_capita = fig


# 2. emise svet na hdp ---
df2 = pd.read_csv(root + '\\data2.csv', decimal=',')

df2['pop'] = np.float_(df2['pop'].apply(lambda x: x.replace('\xa0', '')))
df2['gdp'] = np.float_(df2['gdp'].apply(lambda x: x.replace('\xa0', '')))
df2['gdp_per_capita'] = np.float_(df2['gdp_per_capita'].apply(lambda x: x.replace('\xa0', '')))
df2['ghg_per_gdp'] = np.float_(df2['ghg_per_gdp'])
gdf2 = df2.groupby('cont')

fig, ax = plt.subplots()
# c = 'Asie'
x = 0
xgap = 1e11
ygap = 20

for c in conts:
    cdf = df2.loc[gdf2.groups[c]].sort_values('ghg_per_gdp', ascending=False).reset_index(drop=True)
    cs = colors[c]
    for i, row in cdf.iterrows():
        x = x + xgap
        plt.text(x + 0.5 * row['gdp'] - 4 * xgap, row['ghg_per_gdp'] + ygap, row['region'], rotation=45, fontsize=9)
        plt.rcParams['hatch.color'] = cs[i % 2 + 2]
        rec = mpl.patches.Rectangle((x, 0), row['gdp'], row['ghg_per_gdp'], ec=None, fc=cs[i % 2], hatch='///')
        ax.add_patch(rec)
        x = x + row['gdp']
    x = x + 20 * xgap

ax.set(xlim=(0, 130e12), ylim=(0, 1300))
ax.set(xlabel='GDP', ylabel='g CO2eq per $', title='CO2eq emissions per GDP (2015)')
ax.show()



fig_per_gdp = fig


'{:.3g}'.format(1.23456)

# 3. emise svet na gdp vs ghg per capita ---
# a) horizontal
df1.gdp_per_capita.max()
df1.dtypes

plt.rcParams['figure.figsize'] = 16, 9
fig, ax = plt.subplots()
# c = 'Asie'
x = 0
xgap = 15
ygap = 1000

for c in conts:
    cdf = df1.loc[gdf1.groups[c]].sort_values('gdp_per_capita', ascending=False).reset_index(drop=True)
    cs = colors[c]
    n = cdf.shape[0] - 1
    for i, row in cdf.iterrows():
        x = x + xgap
        plt.text(x + 0.5 * row['ghg_per_gdp'] - 4 * xgap, row['gdp_per_capita'] + ygap, row['region'], rotation=45, fontsize=10)
        plt.rcParams['hatch.color'] = cs[i % 2 + 2]
        rec = mpl.patches.Rectangle((x, 0), row['ghg_per_gdp'], row['gdp_per_capita'], ec=None, fc=cs[i % 2], hatch='xxxx')
        ax.add_patch(rec)
        yy = 0.5 + (n / 2 - i) / 20 if n > 0 else 0.5
        plt.text(x + 0.5 * row['ghg_per_gdp'], yy * row['gdp_per_capita'], '{:.2g}'.format(row['ghg_per_capita']),
                 fontsize=10, va='center', ha='center', color='white', fontweight='bold', backgroundcolor=(0, 0, 0, 0.4))
        x = x + row['ghg_per_gdp']
    x = x + 20 * xgap

ax.set(xlim=(0, 2.1e4), ylim=(0, 7.5e4))
title = 'GDP per capita vs GHG per GDP; area is GHG per capita (2015)'
ax.set(xlabel='GHG per GDP (g CO2eq per $)', ylabel='GDP per capita ($)', title=title)

Chart(ax).show()
Chart(ax, format='svg').show()

fig_relative = fig

Chart([fig_per_capita, fig_relative, fig_per_gdp], title='Emissions per capita and per GDP', cols=2, format='svg').show()
Selector([fig_per_capita, fig_relative, fig_per_gdp], title='Emissions per capita and per GDP').show()

fig.savefig(root + '\\per_capita.svg')
fig.savefig(root + '\\per_capita.pdf')

436 * 107
124 / 72
5.1 / 13

# df1.dtypes
# colors_for_conts = pd.DataFrame({'cont': conts, 'color': [colors[c][0] for c in conts]})
# df1 = pd.merge(df1, colors_for_conts)

sns.scatterplot('ghg_per_gdp', 'gdp_per_capita', data=df1, color='red', size='pop').show()
norm = plt.Normalize(df1['pop'].min(), df1['pop'].max())
sns.scatterplot('ghg_per_gdp', 'gdp_per_capita', data=df1, color='red', size='pop', sizes=(20, 1000), size_norm=norm).show()



# doplnena velikost populace
gdf1 = df1.groupby('cont')
norm = plt.Normalize(df1['pop'].min(), df1['pop'].max())

fig, ax = plt.subplots()
for c in conts:
    cdf = df1.loc[gdf1.groups[c]].sort_values('ghg_per_capita', ascending=False).reset_index(drop=True)
    # sns.scatterplot('ghg_per_gdp', 'gdp_per_capita', data=cdf, color=colors[c][0], label=c, s=40)
    sns.scatterplot('ghg_per_gdp', 'gdp_per_capita', data=cdf, color=colors[c][0], label=c, size='pop',
                    sizes=(20, 1000), size_norm=norm, legend=False, alpha=0.8)
    for i, row in cdf.iterrows():
        #plt.text(row['ghg_per_gdp'], row['gdp_per_capita'] + 1000, row['region'], color=colors[c][0], ha='center', va='center')
        plt.text(row['ghg_per_gdp'], row['gdp_per_capita'] + 100 + np.sqrt(row['pop']) / 20, row['region'],
                 color=colors[c][0], ha='center', va='bottom')

xmax = df1['ghg_per_gdp'].max() * 1.05
ax.set(xlim=(0, xmax), ylim=(0, df1['gdp_per_capita'].max() * 1.05))

# ax.show()

# add hyperbolas
xs = np.linspace(10, 1100, 200)
i = 3
# for i in list(range(1, 11)) + list(range(12, 22, 2)) + list(range(25, 45, 5)):
# for i in list(range(1, 41)):
for i in list(range(2, 41, 2)):
    ys = 1e6 * i / xs
    sns.lineplot(xs, ys, lw=0.6, alpha=0.15, color='black')
    plt.text(1e6 * i / 6e4, 6e4, str(i), color='black', ha='center', va='center')

plt.text(1e6 / 6e4 - 3, 6.25e4, 'Hyperboly jsou emise na osobu (t CO2 eq)', color='black', ha='left', va='center')
ax.set(xlabel='Emise na HDP (g CO2eq / $)', ylabel='GDP na osobu ($)', title='Emise na HDP a na osobu')
ax.show()

fig.savefig(root + '\\emise-hdp-per-capita-hyper-pop.pdf')
fig.savefig(root + '\\emise-hdp-per-capita-hyper-pop.svg')

# puvodni graf
gdf1 = df1.groupby('cont')

fig, ax = plt.subplots()
for c in conts:
    cdf = df1.loc[gdf1.groups[c]].sort_values('ghg_per_capita', ascending=False).reset_index(drop=True)
    sns.scatterplot('ghg_per_gdp', 'gdp_per_capita', data=cdf, color=colors[c][0], label=c)
    for i, row in cdf.iterrows():
        plt.text(row['ghg_per_gdp'], row['gdp_per_capita'] + 1000, row['region'], color=colors[c][0], ha='center', va='center')

xmax = df1['ghg_per_gdp'].max() * 1.05
ax.set(xlim=(0, xmax), ylim=(0, df1['gdp_per_capita'].max() * 1.05))

#ax.show()

# add hyperbolas
xs = np.linspace(10, 1100, 200)
i = 3
# for i in list(range(1, 11)) + list(range(12, 22, 2)) + list(range(25, 45, 5)):
# for i in list(range(1, 41)):
for i in list(range(2, 41, 2)):
    ys = 1e6 * i / xs
    sns.lineplot(xs, ys, lw=0.6, alpha=0.2, color='black')
    plt.text(1e6 * i / 6e4, 6e4, str(i), color='black', ha='center', va='center')

plt.text(1e6 / 6e4 - 3, 6.25e4, 'Hyperboly jsou emise na osobu (t CO2 eq)', color='black', ha='left', va='center')
ax.set(xlabel='Emise na HDP (g CO2eq / $)', ylabel='GDP na osobu ($)', title='Emise na HDP a na osobu')
ax.show()











fig.savefig(root + '\\per_hdp.pdf')
df1.show()
df2.show()


plt.rcParams['figure.figsize'] = 16, 9
fig, ax = plt.subplots()
# c = 'Asie'
x = 0
xgap = 15
ygap = 1000

for c in conts:
    cdf = df1.loc[gdf1.groups[c]].sort_values('gdp_per_capita', ascending=False).reset_index(drop=True)
    cs = colors[c]
    n = cdf.shape[0] - 1
    for i, row in cdf.iterrows():
        x = x + xgap
        plt.text(x + 0.5 * row['ghg_per_gdp'] - 4 * xgap, row['gdp_per_capita'] + ygap, row['region'], rotation=45, fontsize=10)
        plt.rcParams['hatch.color'] = cs[i % 2 + 2]
        rec = mpl.patches.Rectangle((x, 0), row['ghg_per_gdp'], row['gdp_per_capita'], ec=None, fc=cs[i % 2], hatch='xxxx')
        ax.add_patch(rec)
        yy = 0.5 + (n / 2 - i) / 20 if n > 0 else 0.5
        plt.text(x + 0.5 * row['ghg_per_gdp'], yy * row['gdp_per_capita'], '{:.2g}'.format(row['ghg_per_capita']),
                 fontsize=10, va='center', ha='center', color='white', fontweight='bold', backgroundcolor=(0, 0, 0, 0.4))
        x = x + row['ghg_per_gdp']
    x = x + 20 * xgap

ax.set(xlim=(0, 2.1e4), ylim=(0, 7.5e4))
# title = 'GDP per capita vs GHG per GDP; area is GHG per capita (2015)'
# ax.set(xlabel='GHG per GDP (g CO2eq per $)', ylabel='GDP per capita ($)', title=title)
ax.set(xlabel='Emise na HDP (g CO2eq / $)', ylabel='GDP na osobu ($)',
       title='Emisní intenzity (čísla v obdélnících jsou emise na osobu v t CO2eq)')

Chart(ax).show()
Chart(ax, format='svg').show()






    cdf = df1.loc[gdf1.groups[c]].sort_values('ghg_per_capita', ascending=False).reset_index(drop=True)
    cs = colors[c]
    for i, row in cdf.iterrows():
        x = x + xgap
        plt.text(x + 0.5 * row['pop'] - 4 * xgap, row['ghg_per_capita'] + ygap, row['region'], rotation=45, fontsize=9)
        rec = mpl.patches.Rectangle((x, 0), row['pop'], row['ghg_per_capita'], ec=None, fc=cs[i % 2])
        ax.add_patch(rec)
        x = x + row['pop']
    x = x + 20 * xgap

ax.set(xlim=(0, 9.4e9), ylim=(0, 38))
ax.set(xlabel='Population', ylabel='t CO2eq per capita', title='CO2eq emissions per capita (2015)')
ax.show()








3.595 * 258.4
4329 / 7188

25.631
25.631 / 46.907

7188 * 25631 / 4329
