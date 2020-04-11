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

df = pd.read_csv('D:\\projects\\fakta-o-klimatu\\work\\111-emise-svet-srovnani\\cum-pop.csv')

plt.rcParams['figure.figsize'] = 16, 9

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

'befeleme'.contains('fe')

'hefe' in 'befeleme'

x = 'befeleme'
x[2] = 'o'


def f():
    '''popis funkce'''
    pass

f.__doc__

10510785 / 7058449725  # 0.15 %
0.14 / 45.96

13.04 * 7058449725 / 1e9


4163369957 / 7058449725  # 59 %

7058449725 * 5.79 / 1e9
