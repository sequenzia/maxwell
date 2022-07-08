import sys, os, pytz, math
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa
from pyarrow import parquet as pq
from numba import cuda, jit, types

est = pytz.timezone('US/Eastern')

st_yr = 2004
ed_yr = 2018
yr_counter = 0

for y in range(st_yr,ed_yr):
    
    print(y, '--\n')

    bars_fn = 'spy_bars_' + str(y)
    bars_fp = plasma.get_fp(bars_fn,'hydrogen/electron/bars/pre_tf','.parquet')
    bars_tab = pq.read_table(bars_fp)

    bars = bars_tab.to_pandas()
    bars = bars.set_index('bar_dt')

    # -- YB -- #
    bars['tf_yb'] = 0.
    bars.loc[bars['bar_month'] < 7, 'tf_yb'] = 1.
    bars.loc[bars['bar_month'] >= 7, 'tf_yb'] = 2.

    # -- YQ -- #
    bars['tf_yq'] = 0.

    bars.loc[bars['bar_month'].isin(['1','2','3']), 'tf_yq'] = 1.
    bars.loc[bars['bar_month'].isin(['4','5','6']), 'tf_yq'] = 2.
    bars.loc[bars['bar_month'].isin(['7','8','9']), 'tf_yq'] = 3.
    bars.loc[bars['bar_month'].isin(['10','11','12']), 'tf_yq'] = 4.

    # -- YM -- #
    bars['tf_ym'] = bars['bar_month']

    # -- MB -- #
    bars['tf_mb'] = 0
    bars.loc[bars['bar_day'] < 16, 'tf_mb'] = 1.
    bars.loc[bars['bar_day'] >= 16, 'tf_mb'] = 2.

    # -- WD -- #
    bars['tf_wd'] = bars.index.dayofweek.astype(np.float64) + 1

    # -- DH -- #
    bars['tf_dh'] = bars['bar_hour']

    bars.loc[(bars['bar_hour'] == float(9)) & (bars['bar_min'] >= float(30)), 'tf_dh'] = 9.5
    bars.loc[(bars['bar_hour'] == float(15)) & (bars['bar_min'] >= float(30)), 'tf_dh'] = 15.5

    # -- DP -- #
    bars['tf_dp'] = 0

    bars.loc[(bars['bar_time'] <= float(120000)), 'tf_dp'] = 1.
    bars.loc[(bars['bar_time'] > float(120000)) & (bars['bar_time'] <= float(130000)), 'tf_dp'] = 2.
    bars.loc[(bars['bar_time'] > float(130000)), 'tf_dp'] = 3.

    if yr_counter == 0:

        tab_out_full = pa.Table.from_pandas(bars)
    
    else:

        tab_out_yr = pa.Table.from_pandas(bars)
        tab_out_full = pa.concat_tables([tab_out_full,tab_out_yr])

    yr_counter += 1

out_fp = plasma.get_fp('spy_bars','hydrogen/electron/bars/')

pa.parquet.write_table(tab_out_full,out_fp)