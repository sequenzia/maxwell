import sys, warnings, pytz
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, carbon, pandas as pd, numpy as np, pyarrow as pa

from pyarrow import parquet as pq

est = pytz.timezone('US/Eastern')

config = [['SPY_1T',60]]

fp_dir = '/oxygen/krypton/meson/'

for c in config:

    data_fn = c[0]
    res = c[1]

    # --- input --- #
    fp = plasma.get_fp(data_fn, fp_dir)
    tab = pq.read_table(fp)
    bars = tab.to_pandas()

    # --- group blocks -- #
    bars = carbon.set_group_blocks(bars)

    # --- set LAG prices -- #
    bars = carbon.set_lags(bars, res)

    # --- set TR -- #
    bars = carbon.set_tr(bars)

    # --- set moving avgs -- #
    bars = carbon.set_mov_avgs(bars, res)

    # --- set vol avgs -- #
    bars = carbon.set_vol_avgs(bars, res)

    # --- set ATR avgs -- #
    bars = carbon.set_atrs(bars, res)

    # --- set rate of change -- #
    bars = carbon.set_rocs(bars)

    # --- set z-scores -- #
    bars = carbon.set_zscores(bars, res)

    # --- drop first year --- #
    bars = bars[bars.bar_year > bars['bar_year'].min()]

    bars = bars.rename(columns={'bar_tp': 'BAR_TP',
                                'bar_vwap': 'BAR_VWAP',
                                'bar_vol': 'BAR_VOL'})

    # --- output --- #
    fp_out = plasma.get_fp(data_fn,'/oxygen/krypton/neon/')
    tab_out = pa.Table.from_pandas(bars, preserve_index=True)
    pa.parquet.write_table(tab_out,fp_out)
