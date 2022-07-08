import numpy as np, pandas as pd, pyarrow.parquet as pq

def set_options(fp):

    max_rows = 500
    max_cols = 25
    edgeitems = 30
    linewidth = 900
    threshold = 10000
    all_cols = False

    float_p = '%.' + str(fp) + 'f'

    ff = lambda x: float_p % x

    pd.options.display.float_format = ff
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)

    if all_cols:
        pd.set_option('display.expand_frame_repr', False)

    np.set_printoptions(formatter={'float': ff}, edgeitems=edgeitems, linewidth=linewidth, threshold=threshold)

def get_dirac(dirac_dir):

    if dirac_dir[0] != '/':
        sl_1 = '/'
    else:
        sl_1 = ''

    if dirac_dir[-1] != '/':
        sl_2 = '/'
    else:
        sl_2 = ''

    dir_path  = '/var/lib/alpha/quantum/dirac' + sl_1 + dirac_dir + sl_2

    return dir_path

def get_fp(fn,dirac_dir,f_ext='parquet'):

    if f_ext is not None:
        fp = get_dirac(dirac_dir) + fn + '.' + f_ext

    if f_ext is None:
        fp = get_dirac(dirac_dir) + fn

    return fp

def get_bars(fp,y=0,m=0.,d=0.,cols=None,reset_idx=False,add_idxes=False):

    tab = pq.read_table(fp)
    bars = tab.to_pandas()

    if y != 0:
        bars = bars[bars['bar_year'] == y]

    if m != 0:
        bars = bars[bars['bar_month'] == m]

    if d != 0:
        bars = bars[bars['bar_day'] == d]

    if reset_idx:
        bars = bars.sort_values('bar_id')
        bars = bars.reset_index(drop=True)

    if cols is not None:
        bars = bars[cols]

    if add_idxes:
        bars['bar_idx'] = bars.reset_index(drop=True).index.values.astype(np.float64)
        bars['day_idx'] = bars['bar_idx'].floordiv(23400.)
        bars['intra_idx'] = bars['bar_idx'] - (bars['day_idx'] * 23400.)

    return bars

# --- float of time converted to ns float --- #
def ti_to_float_ns(t):

    return float(round(t - int(t),9))
