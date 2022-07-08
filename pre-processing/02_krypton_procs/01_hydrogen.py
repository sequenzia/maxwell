import sys, os, pytz
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, carbon, pandas as pd, numpy as np, pyarrow as pa
from pyarrow import parquet as pq

est = pytz.timezone('US/Eastern')

st_year = 2004

fn = 'spy_bars'
fp = plasma.get_fp(fn,'hydrogen/electron/bars/')

bars = plasma.get_bars(fp, df_type=1, add_idxes=1)

bars = bars[bars['bar_year'] >= st_year]

# -- 0 vol to 1 -- #
bars.loc[bars['bar_vol'] == 0, 'bar_vol'] = 1

# -- set first ptd close to .01 -- #
bars.loc[bars['ptd_cp'] == 0, 'ptd_cp'] = .01

# -- anch vol means -- #
bars['bar_vol_3D'] = bars[['bar_vol']].groupby([pd.Grouper(freq='3D')]).transform(carbon.aa_mean)
bars['bar_vol_3H'] = bars[['bar_vol']].groupby([pd.Grouper(freq='3H')]).transform(carbon.aa_mean)

# -- update fake vols to avg of 3D & 3H -- #
bars.loc[bars.is_fake == 1, 'bar_vol'] = (bars['bar_vol_3D'] + bars['bar_vol_3H']) / 2

# -- drop anch vol cols -- #
bars = bars.drop(columns=['bar_vol_3D','bar_vol_3H'])

# -- add bar_tp -- #
bars['bar_tp'] = bars[['bar_hp', 'bar_lp', 'bar_cp']].mean(axis=1)

# -- bar_tpv: tp * vol -- #
bars['bar_tpv'] = pd.Series(bars['bar_tp'] * bars['bar_vol'])

# -- vwap vol -- #
bar_vwap = bars[['bar_vol','bar_tpv']]. \
    groupby([pd.Grouper(freq='1D')]).transform(carbon.aa_sum, sh=1).fillna(0)

# -- daily vwap -- #
bars['bar_vwap'] = pd.Series(bar_vwap['bar_tpv'] / bar_vwap['bar_vol']).fillna(0)

# -- fill first bar of day with vwap from next bar -- #
bars.loc[bars['intra_idx'] == 0, 'bar_vwap'] = bars['bar_vwap'].shift(periods=-1,freq='s')

bars['eod_idx'] = ((bars['day_idx'] + 1) * 23400) - 1

bars['bar_st_dt'] = bars.index - pd.Timedelta(seconds=1)
bars['bar_st'] = float((bars.index - pd.Timedelta(minutes=1)).strftime('%H%M%S'))








cols = ['bar_id',
        'bar_idx',
        'day_idx',
        'intra_idx',
        'bar_fd',
        'bar_date',
        'bar_time',
        'bar_year',
        'bar_month',
        'bar_day',
        'bar_hour',
        'bar_min',
        'bar_sec',
        'bar_op',
        'bar_hp',
        'bar_lp',
        'bar_cp',
        'bar_tp',
        'bar_vol',
        'bar_vwap',
        'bar_eod',
        'ptd',
        'ptd_cp',
        'day_op',
        'day_hp',
        'day_lp',
        'day_cp',
        'eod_idx',
        'ft_count',
        'ut_count',
        'dt_count',
        'tt_count',
        'tf_yb',
        'tf_yq',
        'tf_ym',
        'tf_mb',
        'tf_wd',
        'tf_dh',
        'tf_dp',
        'is_fake']

bars_1S = bars[cols]




def rebar_save(_bars_in):

    # ......... bars_1T ......... #
    bars_1T = carbon.rebar(bars_1S, '1T')

    # --- set end time --- #
    bars_1T = bars_1T.rename(columns={'bar_time': 'bar_ed'})






# --- set st time --- #











# # --- Output -- #
# nm_1 = 'SPY_1S'
# fp_1 = plasma.get_fp(nm_1,'/oxygen/krypton/hydrogen/')
# tab_1 = pa.Table.from_pandas(bars_1S, preserve_index=True)
# pa.parquet.write_table(tab_1,fp_1)
#
# # ......... bars_15S ......... #
# bars_15S = carbon.rebar(bars_1S,'15S')
#
# nm_2 = 'SPY_15S'
# fp_2 = plasma.get_fp(nm_2,'/oxygen/krypton/hydrogen/')
# tab_2 = pa.Table.from_pandas(bars_15S, preserve_index=True)
# pa.parquet.write_table(tab_2,fp_2)
#
# # ......... bars_30S ......... #
# bars_30S = carbon.rebar(bars_1S,'30S')
#
# nm_3 = 'SPY_30S'
# fp_3 = plasma.get_fp(nm_3,'/oxygen/krypton/hydrogen/')
# tab_3 = pa.Table.from_pandas(bars_30S, preserve_index=True)
# pa.parquet.write_table(tab_3,fp_3)

# ......... bars_1T ......... #

nm_4 = 'SPY_1T'
fp_4 = plasma.get_fp(nm_4,'/oxygen/krypton/hydrogen/')
tab_4 = pa.Table.from_pandas(bars_1T, preserve_index=True)
pa.parquet.write_table(tab_4,fp_4)

# ......... bars_5T ......... #
bars_5T = carbon.rebar(bars_1S,'5T')

nm_5 = 'SPY_5T'
fp_5 = plasma.get_fp(nm_5,'/oxygen/krypton/hydrogen/')
tab_5 = pa.Table.from_pandas(bars_5T, preserve_index=True)
pa.parquet.write_table(tab_5,fp_5)

# ......... bars_15T ......... #
bars_15T = carbon.rebar(bars_1S,'15T')

nm_6 = 'SPY_15T'
fp_6 = plasma.get_fp(nm_6,'/oxygen/krypton/hydrogen/')
tab_6 = pa.Table.from_pandas(bars_15T, preserve_index=True)
pa.parquet.write_table(tab_6,fp_6)

