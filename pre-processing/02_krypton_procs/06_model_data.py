import sys, warnings, pytz
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, carbon, pandas as pd, numpy as np, pyarrow as pa

from pyarrow import parquet as pq

est = pytz.timezone('US/Eastern')

reduce_on = False

st_year = 0
st_month = 0
st_day = 0

max_days = 0

config = [['SPY_1S',1],
          ['SPY_15S',15],
          ['SPY_30S',30],
          ['SPY_1T',60],
          ['SPY_5T',300]]

for c in config:

    data_fn = c[0]
    res = c[1]

    if res == 60:
        freq = '1T'

    bars_pd = int(23400 / res)

    neon_fp = plasma.get_fp(data_fn, '/oxygen/krypton/neon/')
    neon_tab = pq.read_table(neon_fp)
    bars = neon_tab.to_pandas()

    # -- reduce year/month -- #
    if reduce_on:
        bars = bars[bars['bar_year'] >= st_year]
        bars = bars[bars['bar_month'] >= st_month]

        # --- bars indexing  --- #
        bars['bar_idx'] = bars.reset_index(drop=True).index.values.astype(np.float64)
        bars['day_idx'] = bars['bar_idx'].floordiv(bars_pd)
        bars['intra_idx'] = bars['bar_idx'] - (bars['day_idx'] * 23400.)

    # -- reduce days --#
    if reduce_on:
        bars = bars[bars['day_idx'] >= st_day]
        bars = bars[bars['day_idx'] <= int(st_day + max_days)]

    # --- bars indexing  --- #
    bars['bar_idx'] = bars.reset_index(drop=True).index.values.astype(np.float64)
    bars['day_idx'] = bars['bar_idx'].floordiv(bars_pd)
    bars['intra_idx'] = bars['bar_idx'] - (bars['day_idx'] * bars_pd)

    # --- output --- #
    out_fp = plasma.get_fp(data_fn,'/oxygen/model_data/')
    out_tab = pa.Table.from_pandas(bars, preserve_index=True)
    pa.parquet.write_table(out_tab,out_fp)

