import os
import pandas as pd
import matplotlib as mpl
import tempfile
from libs.utils import create_stamped_temp, get_stamp, create_temp_dir
import reportree as rt
import docx

os.environ['TEMP'] = 'D:/temp'
tempfile.tempdir = 'D:/temp'


def _open(path, program=None):
    # this should be at least somewhat cross-platform
    if os.name == 'nt':
        os.startfile(path)
    else:
        os.system(f'{program} {path} >/dev/null 2>&1 &')


def pd_dataframe_show(self, title=None, num_rows=1_000, format='parquet', **kwargs):
    title = title or 'data'
    path = os.path.join(create_stamped_temp('dataframes'), f'{title}.{format}')
    getattr(self.iloc[:num_rows], f'to_{format}')(path, **kwargs)
    _open(path, 'tad')


def docx_document_show(self, **kwargs):
    path = create_temp_dir('docs')
    self.save(os.path.join(path, get_stamp() + '.docx'), **kwargs)
    # can this work on linux at all?
    _open(path)


def rtree_show(t: rt.IRTree, entry='index.htm'):
    path = create_stamped_temp('reports')
    t.save(path, entry=entry)
    _open(os.path.join(path, entry), 'firefox')


rt.IRTree.show = rtree_show


def figure_show(self, title=None):
    rt.Leaf(self, title=title).show()


def axes_show(self, title=None):
    figure_show(self.get_figure(), title)


pd.DataFrame.show = pd_dataframe_show
pd.DataFrame.show_csv = lambda self, **kwargs: pd_dataframe_show(self, format='csv', index=False, encoding='utf-8-sig',
                                                                 **kwargs)
# here I am overwriting the default show method, do I mind?
mpl.figure.Figure.show = figure_show
mpl.axes.Axes.show = axes_show

docx.document.Document.show = docx_document_show
