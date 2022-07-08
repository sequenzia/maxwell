from numpy.core.fromnumeric import ptp
import pandas as pd
import numpy as np

def calc_periods(res,
                 periods,
                 units='P'):

    periods_len = periods

    day_secs = 23400

    if units == 'D':
        to_secs = periods * day_secs
        periods_len = int(to_secs / res)

    if units == 'H':
        to_secs = periods * 60 * 60
        periods_len = int(to_secs / res)

    if units == 'T':
        to_secs = periods * 60
        periods_len = int(to_secs / res)

    return periods_len

def rolling_fn(data,
               win,
               mp=0,
               agg='mean',
               shift=1,
               fillna=0):

               _roll = data.rolling(window=win,min_periods=mp)
               _agg = getattr(_roll,agg)

               return _agg().shift(shift).fillna(fillna)

def wwma(data,
         periods,
         min_periods=0,
         shift=1,
         fillna=0):

    return data.ewm(alpha=1/periods, min_periods=min_periods, adjust=False).mean().shift(shift).fillna(fillna)

def win_data(data,
             res,
             periods,
             units='P',
             ma_type='sma',
             col='bar_tp',
             agg='mean',
             shift=1,
             fillna=0):

    _periods = calc_periods(res,periods,units)

    if ma_type == 'ema':
        return wwma(data[col],_periods,_periods,shift,fillna)

    else:
        return rolling_fn(data[col],_periods,_periods,agg,shift,fillna)

def col_avg(bars,
            res,
            periods,
            col,
            units='P',
            scale=1.0):

    return win_data(data=bars,
                    res=res,
                    periods=periods,
                    units=units,
                    ma_type='sma',
                    col=col) * scale

# -- Anch Sum -- #
def aa_sum(x,sh=0):
    x = x.expanding().sum().shift(sh)
    return x

# -- Anch Avg -- #
def aa_mean(x,sh=0):
    x = x.expanding().mean().shift(sh)
    return x

# -- Anch Std -- #
def aa_std(x,ddof=1,sh=0):
    x = x.expanding().std(ddof=ddof).shift(sh)
    return x

# -- Rebar -- #
def rebar(bars, res):

    secs_pd = 23400
    td_secs = pd.to_timedelta(res).seconds

    bars_pd = secs_pd / td_secs

    z_bars = bars.resample(res, closed='right', label='right').aggregate({'bar_id':'last',
                                                                          'bar_idx':'last',
                                                                          'day_idx':'last',
                                                                          'intra_idx':'last',
                                                                          'bar_fd':'last',
                                                                          'bar_date':'last',
                                                                          'bar_st':'first',
                                                                          'bar_ed': 'last',
                                                                          'bar_year':'last',
                                                                          'bar_month':'last',
                                                                          'bar_day':'last',
                                                                          'bar_hour':'last',
                                                                          'bar_min':'last',
                                                                          'bar_sec':'last',
                                                                          'bar_op':'first',
                                                                          'bar_hp':'max',
                                                                          'bar_lp':'min',
                                                                          'bar_cp':'last',
                                                                          'bar_vol':'sum',
                                                                          'bar_eod':'last',
                                                                          'ptd':'last',
                                                                          'ptd_cp':'last',
                                                                          'day_op':'first',
                                                                          'day_hp':'max',
                                                                          'day_lp':'min',
                                                                          'day_cp':'last',
                                                                          'eod_idx':'last',
                                                                          'ft_count':'sum',
                                                                          'ut_count':'sum',
                                                                          'dt_count':'sum',
                                                                          'tt_count':'sum',
                                                                          'tf_yb':'last',
                                                                          'tf_yq':'last',
                                                                          'tf_ym':'last',
                                                                          'tf_mb':'last',
                                                                          'tf_wd':'last',
                                                                          'tf_dh':'last',
                                                                          'tf_dp':'last'}).dropna()

    z_bars['is_fake'] = 0

    # -- add bar_tp -- #
    z_bars['bar_tp'] = z_bars[['bar_hp', 'bar_lp', 'bar_cp']].mean(axis=1)

    # -- bar_tpv: tp * vol -- #
    z_bars['bar_tpv'] = pd.Series(z_bars['bar_tp'] * z_bars['bar_vol'])

    # -- vwap vol -- #
    bar_vwap = z_bars[['bar_vol','bar_tpv']].\
        groupby([pd.Grouper(freq='1D')]).transform(aa_sum, sh=1).fillna(method='bfill')

    # -- daily vwap -- #
    z_bars['bar_vwap'] = pd.Series(bar_vwap['bar_tpv'] / bar_vwap['bar_vol']).fillna(method='bfill')

    z_bars = z_bars.drop(columns=['bar_tpv'])

    # --- z_bars indexing  --- #
    z_bars['bar_idx'] = z_bars.reset_index(drop=True).index.values.astype(np.float64)
    z_bars['day_idx'] = z_bars['bar_idx'].floordiv(bars_pd)
    z_bars['intra_idx'] = z_bars['bar_idx'] - (z_bars['day_idx'] * bars_pd)

    z_bars['eod_idx'] = ((z_bars['day_idx'] + 1) * bars_pd) - 1

    cols = ['bar_id',
            'bar_idx',
            'day_idx',
            'intra_idx',
            'bar_fd',
            'bar_date',
            'bar_st',
            'bar_ed',
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

    return z_bars[cols]

def lag_pr(bars,
           shift=1,
           pr_type='bar_tp'):

    return bars[pr_type].shift(shift).fillna(0)

def bar_chng(bars,
             shift=1,
             pr_type='bar_cp'):

    if shift > 0:
        return (bars[pr_type] - bars[pr_type].shift(shift)).fillna(0)

    if shift < 0:
        return (bars[pr_type].shift(shift*-1) - bars[pr_type]).fillna(0)

# ---- TA Funcs ---- #

def macd(bars,
         res,
         scale='P'):

    data = bars.copy()

    data['SMA_12'] = win_data(data,res,12,scale,'sma')
    data['SMA_26'] = win_data(data,res,26,scale,'sma')
    data['MACD_L'] = data['SMA_12'] - data['SMA_26']

    data['MACD_SL'] = win_data(data,res,9,scale,'ema','MACD_L')
    data['MACD_H'] = data['MACD_L'] - data['MACD_SL']

    data = data.drop(columns=['SMA_12','SMA_26'])

    return data

def rsi(bars,
        pr_type='bar_tp',
        timeperiod=14,
        fillna=0):

    bars['RSI'] = ta.RSI(bars[pr_type], timeperiod=timeperiod).fillna(fillna)

    bars[['RSI_L','RSI_S','RSI_N']] = 0

    bars.loc[bars['RSI'] >= 70, 'RSI_L'] = bars['RSI'] - 70
    bars.loc[bars['RSI'] <= 30, 'RSI_S'] = bars['RSI'] - 30
    bars.loc[(bars['RSI'] > 30) & (bars['RSI'] < 70), 'RSI_N'] = (100 - bars['RSI'])

    return bars

def z_bands(bars,
            res,
            pr_type='bar_tp',
            ma_periods=5,
            std_periods=5,
            ma_units='D',
            std_units='D'):

    data = bars.copy()

    nm_base = 'ZSC_SMA_' + str(ma_periods) + str(ma_units)

    nm_avg = nm_base + '_avg'
    nm_std = nm_base + '_std'
    nm_dif = nm_base + '_dif'

    data[nm_avg] = win_data(data, res, periods=ma_periods, units=ma_units, ma_type='sma', col=pr_type, agg='mean')

    data[nm_std] = win_data(data, res, periods=std_periods, units=std_units, ma_type='sma', col=pr_type, agg='std')

    data[nm_dif] = 0

    data.loc[data[nm_avg] != 0, nm_dif] = pd.Series(data['bar_tp'] - data[nm_avg])

    data[nm_base] = 0

    data.loc[data[nm_std] != 0, nm_base] = data[nm_dif] / data[nm_std]

    data = data.drop(columns=[nm_avg,nm_std,nm_dif])

    return data

def atr(bars,
        res,
        periods,
        units='P',
        scale=1.0):

    _periods = calc_periods(res, periods, units)

    data = bars.copy()

    high = data['bar_hp']
    low = data['bar_lp']
    close = data['bar_cp']

    data['tr_HL'] = abs(high - low)
    data['tr_HC'] = abs(high - close.shift())
    data['tr_LC'] = abs(low - close.shift())

    tr = data[['tr_HL', 'tr_HC', 'tr_LC']].max(axis=1)

    return wwma(tr, _periods, _periods) * scale

def add_rtp_cols(bars, cols):

    for col in cols:

        n_col = col + '_RTP'

        if col == 'bar_vwap':
            n_col = 'VWAP_RTP'

        col_type = col[:3]

        bars[n_col] = 0

        if col_type == 'ATR':
            bars.loc[bars[col] != 0, n_col] = (bars[col] / bars['bar_tp']) * 100

        if col_type != 'ATR':
            bars.loc[bars[col] != 0, n_col] = ((bars['bar_tp'] - bars[col]) / bars['bar_tp']) * 100

    return bars




def calc_roc(org_val, new_val):

    return ((new_val / org_val) -1) * 100


def calc_pct(org_val, new_val):

    net_chng = new_val - org_val

    return (net_chng / org_val) * 100


# -- set_group_blocks -- #
def set_group_blocks(bars):

    bars['block_idx'] = 0.

    block_times = [
          [093001.00, 100000.00],
          [100001.00, 103000.00],
          [103001.00, 110000.00],
          [110001.00, 113000.00],
          [113001.00, 120000.00],
          [120001.00, 123000.00],
          [123001.00, 130000.00],
          [130001.00, 133000.00],
          [133001.00, 140000.00],
          [140001.00, 143000.00],
          [143001.00, 150000.00],
          [150001.00, 153000.00],
          [153001.00, 160000.00]]

    for block_idx, block in enumerate(block_times):

        block_idx = block_idx + 1.0
        block_st = block[0]
        block_ed = block[1]

        bars.loc[(bars['bar_st'] >= block_st) & (bars['bar_ed'] <= block_ed), 'block_idx'] = block_idx

    return bars

# -- set lag prices -- #
def set_lags(bars, res):

    _col = 'bar_tp'

    _15T = calc_periods(res,15,'T')
    _60T = calc_periods(res,60,'T')
    _195T = calc_periods(res,195,'T')
    _1D = calc_periods(res,1,'D')
    _5D = calc_periods(res,5,'D')
    _15D = calc_periods(res,15,'D')
    _30D = calc_periods(res,30,'D')

    bars['LAG_15T'] = lag_pr(bars, _15T, _col)
    bars['LAG_60T'] = lag_pr(bars, _60T, _col)
    bars['LAG_195T'] = lag_pr(bars, _195T, _col)
    bars['LAG_1D'] = lag_pr(bars, _1D, _col)
    bars['LAG_5D'] = lag_pr(bars, _5D, _col)
    bars['LAG_15D'] = lag_pr(bars, _15D, _col)
    bars['LAG_30D'] = lag_pr(bars, _30D, _col)

    return bars

# -- set true range -- #
def set_tr(bars):

    high = bars['bar_hp']
    low = bars['bar_lp']
    close = bars['bar_cp']

    bars['tr_HL'] = abs(high - low)
    bars['tr_HC'] = abs(high - close.shift())
    bars['tr_LC'] = abs(low - close.shift())

    bars['bar_tr'] = bars[['tr_HL', 'tr_HC', 'tr_LC']].max(axis=1)

    # -- set any bar_tr 0 to .009 -- #
    bars.loc[bars['bar_tr'] == 0, 'bar_tr'] = 0.0099999999999909051

    return bars.drop(columns=['tr_HL','tr_HC','tr_LC'])

# -- set mov avgs -- #
def set_mov_avgs(bars, res):

    bars['SMA_15T'] = win_data(bars,res,15,'T','sma')
    bars['SMA_60T'] = win_data(bars,res,60,'T','sma')
    bars['SMA_195T'] = win_data(bars,res,195,'T','sma')
    bars['SMA_1D'] = win_data(bars,res,1,'D','sma')
    bars['SMA_5D'] = win_data(bars,res,5,'D','sma')
    bars['SMA_15D'] = win_data(bars,res,15,'D','sma')
    bars['SMA_30D'] = win_data(bars,res,30,'D','sma')

    bars['EMA_15T'] = win_data(bars,res,15,'T','ema')
    bars['EMA_60T'] = win_data(bars,res,60,'T','ema')
    bars['EMA_195T'] = win_data(bars,res,195,'T','ema')
    bars['EMA_1D'] = win_data(bars,res,1,'D','ema')
    bars['EMA_5D'] = win_data(bars,res,5,'D','ema')
    bars['EMA_15D'] = win_data(bars,res,15,'D','ema')
    bars['EMA_30D'] = win_data(bars,res,30,'D','ema')

    return bars

# -- set volume avgs -- #
def set_vol_avgs(bars, res):

    bars['VOL_15T'] = col_avg(bars=bars,
                              res=res,
                              periods=15,
                              units='T',
                              col='bar_vol')

    bars['VOL_60T'] = col_avg(bars=bars,
                              res=res,
                              periods=60,
                              units='T',
                              col='bar_vol')

    bars['VOL_195T'] = col_avg(bars=bars,
                              res=res,
                              periods=195,
                              units='T',
                              col='bar_vol')

    bars['VOL_1D'] = col_avg(bars=bars,
                              res=res,
                              periods=1,
                              units='D',
                              col='bar_vol')

    bars['VOL_5D'] = col_avg(bars=bars,
                              res=res,
                              periods=5,
                              units='D',
                              col='bar_vol')

    bars['VOL_15D'] = col_avg(bars=bars,
                              res=res,
                              periods=15,
                              units='D',
                              col='bar_vol')

    bars['VOL_30D'] = col_avg(bars=bars,
                              res=res,
                              periods=30,
                              units='D',
                              col='bar_vol')

    return bars

# -- set active true rngs -- #
def set_atrs(bars, res):

    bars['ATR_15T'] = col_avg(bars=bars,
                              res=res,
                              periods=15,
                              units='T',
                              col='bar_tr')

    bars['ATR_60T'] = col_avg(bars=bars,
                              res=res,
                              periods=60,
                              units='T',
                              col='bar_tr')

    bars['ATR_195T'] = col_avg(bars=bars,
                              res=res,
                              periods=195,
                              units='T',
                              col='bar_tr')

    bars['ATR_1D'] = col_avg(bars=bars,
                              res=res,
                              periods=1,
                              units='D',
                              col='bar_tr')

    bars['ATR_5D'] = col_avg(bars=bars,
                              res=res,
                              periods=5,
                              units='D',
                              col='bar_tr')

    bars['ATR_15D'] = col_avg(bars=bars,
                              res=res,
                              periods=15,
                              units='D',
                              col='bar_tr')

    bars['ATR_30D'] = col_avg(bars=bars,
                              res=res,
                              periods=30,
                              units='D',
                              col='bar_tr')

    return bars

# -- set rate of chng -- #
def set_rocs(bars):

    bars['ROC_VWAP'] = calc_roc(org_val=bars['bar_vwap'], new_val=bars['bar_tp'])
    
    bars['ROC_15T'] = calc_roc(org_val=bars['LAG_15T'], new_val=bars['bar_tp'])
    bars['ROC_60T'] = calc_roc(org_val=bars['LAG_60T'], new_val=bars['bar_tp'])
    bars['ROC_195T'] = calc_roc(org_val=bars['LAG_195T'], new_val=bars['bar_tp'])
    bars['ROC_1D'] = calc_roc(org_val=bars['LAG_1D'], new_val=bars['bar_tp'])
    bars['ROC_5D'] = calc_roc(org_val=bars['LAG_5D'], new_val=bars['bar_tp'])
    bars['ROC_15D'] = calc_roc(org_val=bars['LAG_15D'], new_val=bars['bar_tp'])
    bars['ROC_30D'] = calc_roc(org_val=bars['LAG_30D'], new_val=bars['bar_tp'])
    
    bars['ROC_SMA_15T'] = calc_roc(org_val=bars['SMA_15T'], new_val=bars['bar_tp'])
    bars['ROC_SMA_60T'] = calc_roc(org_val=bars['SMA_60T'], new_val=bars['bar_tp'])
    bars['ROC_SMA_195T'] = calc_roc(org_val=bars['SMA_195T'], new_val=bars['bar_tp'])
    bars['ROC_SMA_1D'] = calc_roc(org_val=bars['SMA_1D'], new_val=bars['bar_tp'])
    bars['ROC_SMA_5D'] = calc_roc(org_val=bars['SMA_5D'], new_val=bars['bar_tp'])
    bars['ROC_SMA_15D'] = calc_roc(org_val=bars['SMA_15D'], new_val=bars['bar_tp'])
    bars['ROC_SMA_30D'] = calc_roc(org_val=bars['SMA_30D'], new_val=bars['bar_tp'])

    return bars

# -- set z scores -- #
def set_zscores(bars, res):

    bars = z_bands(bars, res, ma_periods=15, std_periods=15, ma_units='T', std_units='T')
    bars = z_bands(bars, res, ma_periods=60, std_periods=60, ma_units='T', std_units='T')
    bars = z_bands(bars, res, ma_periods=195, std_periods=195, ma_units='T', std_units='T')
    bars = z_bands(bars, res, ma_periods=1, std_periods=1, ma_units='D', std_units='D')
    bars = z_bands(bars, res, ma_periods=5, std_periods=5, ma_units='D', std_units='D')
    bars = z_bands(bars, res, ma_periods=15, std_periods=15, ma_units='D', std_units='D')
    bars = z_bands(bars, res, ma_periods=30, std_periods=30, ma_units='D', std_units='D')

    return bars


