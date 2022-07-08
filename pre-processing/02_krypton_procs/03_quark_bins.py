import sys, os, pytz, warnings, math, operator
warnings.filterwarnings("ignore")
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa
from pyarrow import parquet as pq

est = pytz.timezone('US/Eastern')

config = [['SPY_1S',1],
          ['SPY_15S',15],
          ['SPY_30S',30],
          ['SPY_1T',60],
          ['SPY_5T',300]]

fp_dir = '/oxygen/_proc/quark/'

for c in config:

    data_fn = c[0]
    res = c[1]

    # --- Input --- #
    fp = plasma.get_fp(data_fn, fp_dir)
    tab = pq.read_table(fp)
    bars = tab.to_pandas()

    bars[['y_long_tg',
          'y_long_to',
          'y_long_sl',
          'y_short_tg',
          'y_short_to',
          'y_short_sl',]] = 0

    bars.loc[bars['long_close_ty'] == 1, 'y_long_tg'] = 1
    bars.loc[bars['long_close_ty'] == 2, 'y_long_sl'] = 1
    bars.loc[bars['long_close_ty'] == 3, 'y_long_to'] = 1

    bars.loc[bars['short_close_ty'] == 1, 'y_short_tg'] = 1
    bars.loc[bars['short_close_ty'] == 2, 'y_short_sl'] = 1
    bars.loc[bars['short_close_ty'] == 3, 'y_short_to'] = 1

    # --- Output --- #
    fp_out = plasma.get_fp(data_fn,fp_dir)
    tab_out = pa.Table.from_pandas(bars, preserve_index=True)
    pa.parquet.write_table(tab_out,fp_out)

