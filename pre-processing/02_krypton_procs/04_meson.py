import sys, os, pytz
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa
from pyarrow import parquet as pq

cuda_devices = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

in_dir = '/oxygen/krypton/quark/'
out_dir = '/oxygen/krypton/meson/'

data_fn = 'SPY_1T'
res = 60

# --- input --- #
in_fp = plasma.get_fp(data_fn, in_dir)
tab = pq.read_table(in_fp)
neon_bars = tab.to_pandas()

_cols = ['bar_idx',
         'bar_id',
         'bar_ts',
         'bar_fd',
         'bar_time',
         'bar_date',
         'bar_sec',
         'bar_min',
         'bar_hour',
         'bar_day',
         'bar_month',
         'bar_year',
         'bar_eod',
         'day_idx',
         'intra_idx',
         'eod_idx',
         'bar_op',
         'bar_hp',
         'bar_lp',
         'bar_cp',
         'bar_tp',
         'bar_vol',
         'bar_vwap',
         'ptd',
         'ptd_cp',
         'day_op',
         'day_lp',
         'day_cp',
         'day_hp',
         'tf_yq',
         'tf_ym',
         'tf_yb',
         'tf_mb',
         'tf_wd',
         'tf_dh',
         'tf_dp',
         'dt_count',
         'tt_count',
         'ut_count',
         'ft_count',
         'period_atr',
         'bar_r']

all_bars = neon_bars[_cols].copy()

# ----- setup is close ----- #

mod_split_1 = 104
mod_split_2 = 284

# -- set bar_dt -- #
all_bars['bar_dt'] = all_bars.index

# -- set up close info -- #
all_bars['is_close'] = False

all_bars.loc[(all_bars['intra_idx'] == mod_split_1) & (all_bars['day_idx'] > 0), 'is_close'] = True
all_bars.loc[(all_bars['intra_idx'] == mod_split_2) & (all_bars['day_idx'] > 0), 'is_close'] = True

all_bars[all_bars['is_close'] == True]

# -- days back bars -- #
def db_bars(_bars, _hold_config):

    open_cols = ['bar_idx', 'bar_date', 'bar_dt', 'bar_time', 'day_idx' , 'intra_idx', 'bar_tp']

    close_cols = ['bar_idx', 'bar_date', 'bar_dt', 'bar_time', 'day_idx', 'intra_idx', 'bar_tp']

    # -- create close bars -- #
    close_bars = _bars[_bars['is_close'] == True][close_cols]

    for i in range(len(hold_config)):

        _mins = _hold_config[i]['hold_bars']
        _hd_nm = _hold_config[i]['nm']
        
        L1 = _hold_config[i]['cls_rngs']['L1']
        L2 = _hold_config[i]['cls_rngs']['L2']

        S1 = _hold_config[i]['cls_rngs']['S1']
        S2 = _hold_config[i]['cls_rngs']['S2']

        # ------------ set a variables ------------ #

        db_bar_idx = _hd_nm + '_bar_idx'
        db_bar_dt = _hd_nm + '_bar_dt'
        db_bar_date = _hd_nm + '_bar_date'
        db_bar_time = _hd_nm + '_bar_time'
        db_day_idx = _hd_nm + '_day_idx'
        db_intra_idx = _hd_nm + '_intra_idx'

        db_pr = _hd_nm + '_pr'
        db_net = _hd_nm + '_net'
        db_pct = _hd_nm + '_pct'

        db_mins = _hd_nm + '_mins'
        db_mkt_mins = _hd_nm + '_mkt_mins'
        db_true_mins = _hd_nm + '_true_mins'

        db_cls = _hd_nm + '_cls'
        
        db_L1 = _hd_nm + '_L1'
        db_L2 = _hd_nm + '_L2'
        
        db_S1 = _hd_nm + '_S1'
        db_S2 = _hd_nm + '_S2'

        db_N0 = _hd_nm + '_N0'

        # -- set open_idx -- #
        close_bars['open_idx'] = close_bars['bar_idx'] - _mins

        # -- merge bars with close bars -- #
        close_bars = close_bars.merge(_bars[open_cols], left_on='open_idx', right_on='bar_idx', suffixes=('_CB', '_OB'))

        # -- remame merged cols -- #
        close_bars = close_bars.rename(columns={'bar_idx_CB': 'bar_idx',
                                                'bar_dt_CB': 'bar_dt',
                                                'bar_date_CB': 'bar_date',
                                                'bar_time_CB': 'bar_time',
                                                'day_idx_CB': 'day_idx',
                                                'intra_idx_CB': 'intra_idx',
                                                'bar_tp_CB': 'bar_tp',
                                                'is_close_CB': 'is_close',

                                                'bar_idx_OB': db_bar_idx,
                                                'bar_dt_OB': db_bar_dt,
                                                'bar_date_OB': db_bar_date,
                                                'bar_time_OB': db_bar_time,
                                                'day_idx_OB': db_day_idx,

                                                'intra_idx_OB': db_intra_idx,
                                                'bar_tp_OB': db_pr})

        #-- delete open_idx -- #
        close_bars = close_bars.drop(['open_idx'], axis=1)

        close_bars[db_net] = close_bars['bar_tp'] - close_bars[db_pr]
        close_bars[db_pct] = ((close_bars['bar_tp'] - close_bars[db_pr]) / close_bars[db_pr]) * 100

        # --- close classes --- #

        close_bars[db_cls] = 0

        close_bars.loc[(close_bars[db_pct] >= L1) & (close_bars[db_pct] < L2), db_cls] = 1
        close_bars.loc[close_bars[db_pct] >= L2, db_cls] = 2

        close_bars.loc[(close_bars[db_pct] <= S1) & (close_bars[db_pct] > S2), db_cls] = -1
        close_bars.loc[close_bars[db_pct] <= S2, db_cls] = -2

        # --- L1, L2, S1, S2 --- #

        close_bars[db_L1] = 0
        close_bars[db_L2] = 0

        close_bars[db_S1] = 0
        close_bars[db_S2] = 0

        close_bars[db_N0] = 1

        close_bars.loc[close_bars[db_cls] == 1, db_L1] = 1
        close_bars.loc[close_bars[db_cls] == 2, db_L2] = 1

        close_bars.loc[close_bars[db_cls] == -1, db_S1] = 1
        close_bars.loc[close_bars[db_cls] == -2, db_S2] = 1

        close_bars.loc[close_bars[db_cls] != 0, db_N0] = 0

        # --- time components --- #
        close_bars[db_mins] = _mins

        close_bars[db_mkt_mins] = close_bars['bar_idx'] - close_bars[db_bar_idx]
        close_bars[db_true_mins] = (close_bars['bar_dt'] - close_bars[db_bar_dt]).astype('timedelta64[s]') / 60

    close_bars = close_bars.rename(columns={'bar_idx': 'close_bar_idx',
                                            'bar_date': 'close_bar_date',
                                            'bar_time': 'close_bar_time',
                                            'day_idx': 'close_day_idx',
                                            'intra_idx': 'close_intra_idx',
                                            'bar_tp': 'close_bar_pr'})

    return close_bars

# -- hold configs -- #

hold_config = [
        {'hold_bars': 390, 'nm': 'DB1', 'cls_rngs': {'L1': .25,
                                                     'L2': .75,
                                                     'S1': -.25,
                                                     'S2': -.75}},

        {'hold_bars': 780, 'nm': 'DB2', 'cls_rngs': {'L1': .40,
                                                     'L2': 1.15,
                                                     'S1': -.40,
                                                     'S2': -1.15}},

        {'hold_bars': 1170, 'nm': 'DB3', 'cls_rngs': {'L1': .55,
                                                      'L2': 1.45,
                                                      'S1': -.55,
                                                      'S2': -1.45}},

        {'hold_bars': 1560, 'nm': 'DB4', 'cls_rngs': {'L1': .65,
                                                      'L2': 1.60,
                                                      'S1': -.65,
                                                      'S2': -1.60}},

        {'hold_bars': 1950, 'nm': 'DB5', 'cls_rngs': {'L1': .72,
                                                      'L2': 1.75,
                                                      'S1': -.72,
                                                      'S2': -1.75}},

        {'hold_bars': 2340, 'nm': 'DB6', 'cls_rngs': {'L1': .77,
                                                      'L2': 1.82,
                                                      'S1': -.77,
                                                      'S2': -1.82}},

        {'hold_bars': 2730, 'nm': 'DB7', 'cls_rngs': {'L1': .85,
                                                      'L2': 2.0,
                                                      'S1': -.85,
                                                      'S2': -2.0}}]

# ---- run ---- #
close_bars = db_bars(all_bars, hold_config)

# ---- merge bars ---- #
bars = all_bars.merge(close_bars, left_on='bar_idx', right_on='close_bar_idx', how='left')

# --- update bar_st & bar_ed --- #

bars['bar_st'] = (bars['bar_dt_x'] + pd.Timedelta(seconds=res*-1)).dt.strftime('%H%M%S').astype(float)

bars = bars.rename(columns={'bar_time': 'bar_ed'})

# --- drop cols & fill --- #

bars = bars.drop(['DB1_bar_dt',
                  'DB2_bar_dt',
                  'DB3_bar_dt',
                  'DB4_bar_dt',
                  'DB5_bar_dt',
                  'DB6_bar_dt',
                  'DB7_bar_dt'], axis=1)

cols_out = ['bar_idx',
            'bar_id',
            'bar_ts',
            'bar_fd',
            'bar_st',
            'bar_ed',
            'bar_date',
            'bar_sec',
            'bar_min',
            'bar_hour',
            'bar_day',
            'bar_month',
            'bar_year',
            'bar_eod',
            'day_idx',
            'intra_idx',
            'eod_idx',
            'bar_op',
            'bar_hp',
            'bar_lp',
            'bar_cp',
            'bar_tp',
            'bar_vol',
            'bar_vwap',
            'ptd',
            'ptd_cp',
            'day_op',
            'day_lp',
            'day_cp',
            'day_hp',
            'tf_yq',
            'tf_ym',
            'tf_yb',
            'tf_mb',
            'tf_wd',
            'tf_dh',
            'tf_dp',
            'dt_count',
            'tt_count',
            'ut_count',
            'ft_count',
            'period_atr',
            'bar_r',
            'is_close',
            'close_bar_idx',
            'close_bar_date',
            'close_bar_time',
            'close_day_idx',
            'close_intra_idx',
            'close_bar_pr',
            'DB1_bar_idx',
            'DB1_bar_date',
            'DB1_bar_time',
            'DB1_day_idx',
            'DB1_intra_idx',
            'DB1_pr',
            'DB1_net',
            'DB1_pct',
            'DB1_cls',
            'DB1_S2',
            'DB1_S1',
            'DB1_N0',
            'DB1_L1',
            'DB1_L2',
            'DB1_mins',
            'DB1_mkt_mins',
            'DB1_true_mins',
            'DB2_bar_idx',
            'DB2_bar_date',
            'DB2_bar_time',
            'DB2_day_idx',
            'DB2_intra_idx',
            'DB2_pr',
            'DB2_net',
            'DB2_pct',
            'DB2_cls',
            'DB2_S2',
            'DB2_S1',
            'DB2_N0',
            'DB2_L1',
            'DB2_L2',
            'DB2_mins',
            'DB2_mkt_mins',
            'DB2_true_mins',
            'DB3_bar_idx',
            'DB3_bar_date',
            'DB3_bar_time',
            'DB3_day_idx',
            'DB3_intra_idx',
            'DB3_pr',
            'DB3_net',
            'DB3_pct',
            'DB3_cls',
            'DB3_S2',
            'DB3_S1',
            'DB3_N0',
            'DB3_L1',
            'DB3_L2',
            'DB3_mins',
            'DB3_mkt_mins',
            'DB3_true_mins',
            'DB4_bar_idx',
            'DB4_bar_date',
            'DB4_bar_time',
            'DB4_day_idx',
            'DB4_intra_idx',
            'DB4_pr',
            'DB4_net',
            'DB4_pct',
            'DB4_cls',
            'DB4_S2',
            'DB4_S1',
            'DB4_N0',
            'DB4_L1',
            'DB4_L2',
            'DB4_mins',
            'DB4_mkt_mins',
            'DB4_true_mins',
            'DB5_bar_idx',
            'DB5_bar_date',
            'DB5_bar_time',
            'DB5_day_idx',
            'DB5_intra_idx',
            'DB5_pr',
            'DB5_net',
            'DB5_pct',
            'DB5_cls',
            'DB5_S2',
            'DB5_S1',
            'DB5_N0',
            'DB5_L1',
            'DB5_L2',
            'DB5_mins',
            'DB5_mkt_mins',
            'DB5_true_mins',
            'DB6_bar_idx',
            'DB6_bar_date',
            'DB6_bar_time',
            'DB6_day_idx',
            'DB6_intra_idx',
            'DB6_pr',
            'DB6_net',
            'DB6_pct',
            'DB6_cls',
            'DB6_S2',
            'DB6_S1',
            'DB6_N0',
            'DB6_L1',
            'DB6_L2',
            'DB6_mins',
            'DB6_mkt_mins',
            'DB6_true_mins',
            'DB7_bar_idx',
            'DB7_bar_date',
            'DB7_bar_time',
            'DB7_day_idx',
            'DB7_intra_idx',
            'DB7_pr',
            'DB7_net',
            'DB7_pct',
            'DB7_cls',
            'DB7_S2',
            'DB7_S1',
            'DB7_N0',
            'DB7_L1',
            'DB7_L2',
            'DB7_mins',
            'DB7_mkt_mins',
            'DB7_true_mins']

bars = bars[cols_out].fillna(0.00)

# --- checks --- #

msg = f""

for _idx in range(7):

    _db = hold_config[_idx]['nm']

    db_idx = _db + '_idx'
    db_pr = _db + '_pr'
    db_net = _db + '_net'
    db_pct = _db + '_pct'
    db_cls = _db + '_cls'

    db_N0 = _db + '_N0'

    db_S2 = _db + '_S2'
    db_S1 = _db + '_S1'

    db_L1 = _db + '_L1'
    db_L2 = _db + '_L2'

    db_cls_rngs = hold_config[_idx]['cls_rngs']

    S2 = db_cls_rngs['S2']
    S1 = db_cls_rngs['S1']

    L1 = db_cls_rngs['L1']
    L2 = db_cls_rngs['L2']

    _close_bars = bars[bars['is_close'] == True]

    total_bars = _close_bars.shape[0]

    bars_N0 = _close_bars[_close_bars[db_cls] == 0].shape[0]
    bars_L1 = _close_bars[_close_bars[db_cls] == 1].shape[0]
    bars_L2 = _close_bars[_close_bars[db_cls] == 2].shape[0]
    bars_S1 = _close_bars[_close_bars[db_cls] == -1].shape[0]
    bars_S2 = _close_bars[_close_bars[db_cls] == -2].shape[0]

    bars_N0_ = _close_bars[_close_bars[db_N0] == 1].shape[0]

    bars_L1_ = _close_bars[_close_bars[db_L1] == 1].shape[0]
    bars_L2_ = _close_bars[_close_bars[db_L2] == 1].shape[0]

    bars_S1_ = _close_bars[_close_bars[db_S1] == 1].shape[0]
    bars_S2_ = _close_bars[_close_bars[db_S2] == 1].shape[0]


    (bars_N0 + bars_L1 + bars_L2 + bars_S1 + bars_S2) - total_bars

    (bars_N0_ + bars_L1_ + bars_L2_ + bars_S1_ + bars_S2_) - total_bars

    bars_N0_pct = bars_N0_ / total_bars

    bars_L1_pct = bars_L1_ / total_bars
    bars_L2_pct = bars_L2_ / total_bars

    bars_S1_pct = bars_S1_ / total_bars
    bars_S2_pct = bars_S2_ / total_bars

    msg += f"\n\n  ---------------- {_db}  ----------------\n"

    msg += f"\n S2: {bars_S2_pct}"
    msg += f"\n S1: {bars_S1_pct}"

    msg += f"\n\n N0: {bars_N0_pct}\n"

    msg += f"\n L1: {bars_L1_pct}"
    msg += f"\n L2: {bars_L2_pct}"

print(f"{msg}")

# ---- OUTPUT ---- #
out_fp = plasma.get_fp(data_fn,out_dir)
tab_out = pa.Table.from_pandas(bars[cols_out], preserve_index=True)
pa.parquet.write_table(tab_out, out_fp)
