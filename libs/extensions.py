import os
import pandas as pd
import matplotlib as mpl
import tempfile
from datetime import datetime
from libs.utils import create_stamped_temp
from libs.plots import Chart


def pd_dataframe_show(self, num_rows=1_000, format='csv', **kwargs):
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = tempfile.gettempdir() + '/dataframes'
    os.makedirs(path, exist_ok=True)
    path = '{}/{}.{}'.format(path, stamp, format)
    getattr(self.iloc[:num_rows], 'to_{}'.format(format))(path, **kwargs)
    os.startfile(path)

    # if num_rows > 0 and self.shape[0] > num_rows:
    #     self.iloc[:num_rows].to_parquet(path)
    # else:
    #     getattr(self, 'to_{}'.format(format))(path)
    #     #self.to_parquet(path)
    # os.startfile(path)


def figure_show(self, title=None):
    Chart(self, title=title).show()
    # dir = create_stamped_temp('reports')
    # fig_path = '{}/fig.png'.format(dir)
    # rep_path = '{}/page.html'.format(dir)
    # self.savefig(fig_path)
    # plt.close('all')
    # html = '<html><head><title>{}</title></head><body><h2>{}</h2><div><img src="{}"></body></html>'.format(title, title, fig_path)
    # file = open(rep_path, 'w')
    # file.write(html)
    # file.close()
    # os.startfile(rep_path)

def axes_show(self, title=None):
    figure_show(self.get_figure(), title)


pd.DataFrame.show = pd_dataframe_show
pd.DataFrame.show_csv = lambda self, **kwargs: pd_dataframe_show(self, format='csv', index=False, encoding='utf-8-sig',
                                                                 **kwargs)

# here I am overwriting the default show method, do I mind?
mpl.figure.Figure.show = figure_show
mpl.axes.Axes.show = axes_show

