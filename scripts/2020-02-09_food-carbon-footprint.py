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

# -----------------------------
# --- FOOD CARBON FOOTPRINT ---

# Can I parse the LCA meta model database? ---

root = 'D:\\projects\\fakta-o-klimatu\\work\\389-food-carbon-footprint'
path = root + '\\LCA+Meta-Analysis_database-static.xlsx'
df = pd.read_excel(path, header=2)
df.columns

labels = df[np.isfinite(df['Unnamed: 0'])][['Unnamed: 0', 'Reference']]

df['Unnamed: 0'] = df['Unnamed: 0'].fillna(method='ffill')

ghg_cols = {
    'GHG Emis \n(kg CO2 eq)': 'ghg_total',
    'LUC Burn': 'luc_burn',
    'LUC C Stock': 'luc_stock',
    'Feed': 'feed',
    'Farm': 'farm',
    'Prcssing': 'processing',
    'Tran & Str': 'transport',
    'Packging': 'packaging',
    'Ret\nail': 'retail',
    'Loss.1': 'loss'
}
w_col = 'Weight'
g_col = 'Unnamed: 0'

data = df[[g_col, w_col, *ghg_cols]].rename(columns={w_col: 'weight', g_col: 'product', **ghg_cols})
data['processing'] = data['processing'].replace(to_replace='-', value=0)

data.show()
data.dtypes

data['luc'] = data['']


ghgs = list(ghg_cols.values())

df.dtypes.reset_index().show()
df.columns[65:76]

data
data.dtypes

data.groupby('product')['weight'].sum()
data.groupby('product')['weight'].sum()

for c in ghgs:
    data[f'{c}_w'] = data[c] * data['weight']

avgs = data.groupby('product')[[f'{c}_w' for c in ghgs]].sum().reset_index().rename(columns={f'{c}_w':c for c in ghgs})
labels.columns = ['product', 'label']
labels = labels.iloc[:-1]

avgs = pd.merge(avgs, labels)

avgs = avgs[['label', *ghgs]].copy()
avgs['luc'] = avgs.luc_burn + avgs.luc_stock

avgs.show()

plain = data.groupby('product')[ghgs].mean().reset_index()
plain = pd.merge(plain, labels)
plain = plain[['label', *ghgs]].copy()
plain['luc'] = plain.luc_burn + plain.luc_stock
plain.show()

plain['ghg'] = plain.eval('luc + feed + farm + transport + processing + packaging + retail')
avgs['ghg'] = avgs.eval('luc + feed + farm + transport + processing + packaging + retail')
avgs.show()

sns.barplot('')

cols = 'luc feed farm transport processing packaging retail'.split()

plt.rcParams['figure.figsize'] = 12, 9
plt.rcParams['figure.subplot.left'] = 0.2

sums = avgs.set_index('label')[cols].sum(axis=1).sort_values()
ax = avgs.set_index('label').loc[sums.index][cols].plot.barh(stacked=True)
for x, y in enumerate(sums):
    ax.text(y + 3, x, f'{y:.3g}', va='center')
#ax.text(50, 5, 'here')
ax.show()

second_file = 'aaq0216_DataS2.xls'
path2 = root + '\\' + second_file
df2 = pd.read_excel(path2, sheet_name='Results - Global Totals', header=2)
df2.show()

ghg_cols = ['LUC', 'Feed', 'Farm', 'Processing', 'Transport', 'Packging', 'Retail']

df2 = df2.iloc[:43].set_index('Product')[ghg_cols]

sums = df2.sum(axis=1).sort_values()
ax = df2.loc[sums.index].plot.barh(stacked=True)
for x, y in enumerate(sums):
    ax.text(y + 3, x, f'{y:.3g}', va='center')
ax.set_title('Total GHG based on LCA, FBS 1kg weight')
ax.show()

# THIS IS GOOD! ---
# exactly the same results as on OurWorldInData

foo = """Beef (beef herd) 16 10 22 2.1 0 0 0
Lamb & Mutton 5.7 3.7 11 0.7 0 0 0
Beef (dairy herd) 13 8.6 18 1.7 0 0 0
Buffalo 2.7 1.8 2.8 0.4 0 0 0
Crustaceans (farmed) 4.3 2.1 1.1 0.2 0 0 0
Cheese 8.3 8.0 27 1.7 0 0 0
Pig Meat 45 28 112 4.5 0 0 0
Fish (farmed) 18 7.4 12 1.7 0 0 0
Poultry Meat 39 26 51 4.5 0 0 0
Eggs 24 24 34 2.6 0 0 0
Fish (capture) 21 8.4 13 1.9 0 0 0
Crustaceans (capture) 8.5 4.2 2.1 0.4 0 0 0
Tofu 3.5 3.2 2.5 0.5 53 40 8.4
Groundnuts 4.3 3.5 21 0.9 5.9 36 1.6
Other Pulses 16 15 51 3.1 55 188 12
Nuts 6.0 2.7 16 0.5 4.7 28 0.8
Peas 2.3 2.1 7.2 0.5 7.9 27 1.8
Milk 185 171 105 6.1 0 0 0
Butter, Cream & Ghee 4.8 4.6 29 0.1 0 0 0
Soymilk 10 9.1 5.1 0.3 185 104 6.1
Cassava 55 45 44 0.4 45 44 0.4
Rice 148 134 494 9.3 146 538 10
Oatmeal 1.6 1.0 2.6 0.1 1.1 2.9 0.1
Potatoes 115 90 66 1.3 90 66 1.3
Wheat & Rye (Bread) 182 166 471 14 181 513 15
Maize (Meal) 47 28 127 3.1 31 138 3.3
Cereals & Oilcr. Misc. 39 34 93 3.2 37 101 3.5
Palm Oil 6.6 6.7 52 0 7.5 59 0
Soybean Oil 10 10 77 0 11 87 0
Olive Oil 1.2 1.3 10 0 1.5 11 0
Rapeseed Oil 4.0 4.1 33 0 4.7 37 0
Sunflower Oil 3.8 3.8 31 0 4.3 35 0
Oils Misc. 5.1 4.7 41 0 5.2 47 0
Animal Fats 4.2 3.8 27 0 0 0 0
Tomatoes 55 37 6.7 0.4 44 8.0 0.4
Brassicas 28 25 6.2 0.3 30 7.5 0.4
Onions & Leeks 29 23 8.7 0.3 28 10 0.4
Root Vegetables 13 11 2.8 0.2 14 3.4 0.2
Other Vegetables 241 213 53 2.9 256 64 3.4
Aquatic Plants 5.0 4.4 1.8 0.1 5.3 2.1 0.1
Berries 11 7.5 4.3 0 9.0 5.1 0.1
Bananas 42 29 19 0.2 35 23 0.3
Apples 25 22 9.2 0.1 27 11 0.1
Citrus 48 40 11 0.2 47 13 0.2
Other Fruit 77 58 26 0.3 69 32 0.4
Cane Sugar 50 41 145 0 41 145 0
Beet Sugar 10 7.9 28 0 7.9 28 0
Sweeteners & Honey 8.3 6.7 20 0 6.7 20 0
Beer 72 63 28 0.3 63 28 0.3
Wine 9.1 8.0 5.3 0 8.0 5.3 0
Dark Chocolate 1.7 0.6 3.0 0.1 0.6 3.0 0.1
Coffee 3.1 1.7 0.7 0.1 1.7 0.7 0.1
Stimul. & Spices Misc. 5.3 3.5 6.8 0.4 3.5 6.8 0.4""".split(sep='\n')

pat = '^([^0-9]*) ([0-9.]*) ([0-9.]*) '

res = []
for x in foo:
    m = re.match(pat, x)
    res.append((m.group(1), m.group(2), m.group(3)))

conv = pd.DataFrame(res, columns=['Product', 'fbs', 'retail'])
conv['fbs'] = np.float_(conv.fbs)
conv['retail'] = np.float_(conv.retail)
conv['ratio'] = conv['fbs'] / conv['retail']

conv['Product'] = conv.Product \
    .replace('Beef (beef herd)', 'Bovine Meat (beef herd)') \
    .replace('Beef (dairy herd)', 'Bovine Meat (dairy herd)') \
    .replace('Citrus', 'Citrus Fruit') \
    .replace('Berries', 'Berries & Grapes')

bar = pd.merge(df2.reset_index(), conv)
for c in ghg_cols:
    bar[c] = bar[c] * bar.ratio
bar = bar.set_index('Product')[ghg_cols]

sums = bar.sum(axis=1).sort_values()
ax = bar.loc[sums.index].plot.barh(stacked=True)
for x, y in enumerate(sums):
    ax.text(y + 3, x, f'{y:.3g}', va='center')
ax.show()

# ok, this is not bad
# compare with the initial, including all

data.dtypes
data['luc'] = data['luc_burn'] + data['luc_stock']
data['processing'] = data['processing'].replace(to_replace='-', value=0)

data.farm
data = data.rename(columns={'feed': 'feed_orig', 'farm': 'farm_orig'})
data['farm'] = data.apply(lambda x: x.feed_orig if np.isnan(x.farm_orig) else x.farm_orig, axis=1)
data['feed'] = data.apply(lambda x: 0.0 if np.isnan(x.farm_orig) else x.feed_orig, axis=1)

ghgs = ['luc', 'feed', 'farm', 'processing', 'transport', 'packaging', 'retail', 'loss']
for c in ghgs:
    data[f'{c}_w'] = data[c] * data['weight']

avgs = data.groupby('product')[[f'{c}_w' for c in ghgs]].sum().reset_index().rename(columns={f'{c}_w':c for c in ghgs})
labels.columns = ['product', 'label']
labels = labels.iloc[:-1]

avgs = pd.merge(avgs, labels)


avgs = avgs.drop(columns=['product']).set_index('label')
sums = avgs.sum(axis=1).sort_values()
ax = avgs.loc[sums.index].plot.barh(stacked=True)
for x, y in enumerate(sums):
    ax.text(y + 3, x, f'{y:.3g}', va='center')
ax.set_title('Total GHG based on LCA, retail 1kg weight')
ax.show()









in_df2 = set(df2.index)
in_conv = set(conv.Product)

in_df2.difference(in_conv)
in_conv.difference(in_df2)

bar.group(3)

f = pd.ExcelFile(path2)
foo = f.sheet_names[2]
df2 = pd.read_excel(path2, sheet_name=foo, header=2)

df2 = pd.read_excel(path, sheet_name=2, header=2)

f.book.sheet_by_index(2)


{'Barley (Beer)',
 'Berries & Grapes',
 'Bovine Meat (beef herd)',
 'Bovine Meat (dairy herd)',
 'Citrus Fruit'}










