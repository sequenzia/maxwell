import sys, os, pytz, warnings, math, operator
warnings.filterwarnings("ignore")
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa, cudf as cd, cupy as cp, numba as nb
from pyarrow import parquet as pq
import concurrent.futures
from contextlib import redirect_stdout

# --- Periods CUDA --- #
@nb.cuda.jit()
def periods_cuda(bars, periods_out, config):

    # - i/j threads & size
    i_thread = nb.cuda.grid(1)
    i_size = nb.cuda.gridsize(1)

    # - Total bars in dataset -
    bars_period = config[12]
    total_bars = config[15]
    total_periods = config[16]

    # -- Loop all bars -- #
    for i_idx in range(i_thread, total_bars, i_size):

        # -- set start idx -- #
        st_idx = (i_idx / bars_period)

        # -- if start idx falls on period st -- #
        if st_idx == math.floor(st_idx):

            period_st_idx = i_idx
            period_ed_idx = i_idx + bars_period

            # -- loop all bars in period -- #
            for period_idx in range(period_st_idx, period_ed_idx):

                if period_idx < total_bars:

                    # -- Variables for current bar -- #
                    cur_bar_ts = bars[period_idx,1]
                    cur_bar_time = bars[period_idx,2]
                    cur_bar_date = bars[period_idx,3]

                    cur_bar_op = bars[period_idx,6]
                    cur_bar_hp = bars[period_idx,7]
                    cur_bar_lp = bars[period_idx,8]
                    cur_bar_cp = bars[period_idx,9]


                    # -- if first bars in period -- #
                    if period_idx == period_st_idx:

                        # -- set period st vars -- #
                        period_date = cur_bar_date
                        period_sidx = period_idx
                        period_sts = cur_bar_ts

                        period_op = cur_bar_op
                        period_hp = cur_bar_hp
                        period_lp = cur_bar_lp

                    # -- catch period high -- #
                    if cur_bar_hp > period_hp:
                        period_hp = cur_bar_hp

                    # -- catch period low -- #
                    if cur_bar_lp < period_lp:
                        period_lp = cur_bar_lp

                    # -- set period end bars-- #
                    if period_idx == period_ed_idx - 1:

                        period_cp = cur_bar_cp
                        period_eidx = period_idx
                        period_ets = cur_bar_ts

                        # ---------- period to bars ---------- #
                        pb_sidx = int(period_idx + 1)
                        pb_eidx = int(pb_sidx + (bars_period - 1))

                        pb_sts = bars[pb_sidx,0]
                        pb_ets = bars[pb_eidx,0]

                        pb_st = cur_bar_ts + 1
                        pb_ed = pb_st + (bars_period - 1)

                        # ---------- out ---------- #
                        out_idx = int(period_idx / bars_period)

                        periods_out[out_idx,0] = period_date

                        periods_out[out_idx,1] = period_sidx
                        periods_out[out_idx,2] = period_eidx

                        periods_out[out_idx,3] = period_sts
                        periods_out[out_idx,4] = period_ets

                        periods_out[out_idx,5] = pb_sidx
                        periods_out[out_idx,6] = pb_eidx

                        periods_out[out_idx,7] = pb_sts
                        periods_out[out_idx,8] = pb_ets

                        periods_out[out_idx,9]	= period_op
                        periods_out[out_idx,10]	= period_hp
                        periods_out[out_idx,11]	= period_lp
                        periods_out[out_idx,12] = period_cp

# --- TR CUDA --- #
@nb.cuda.jit()
def tr_cuda(periods):

    # - i/j threads & size
    i_thread = nb.cuda.grid(1)
    i_size = nb.cuda.gridsize(1)

    # - Total Periods
    total_periods = periods.shape[0]

    # - Loop All Periods
    for i_idx in range(i_thread,total_periods,i_size):

        # - Set previous period idx but overide to 0 for first - #
        if i_idx > 0:
            pi_idx = i_idx - 1
        else:
            pi_idx = 0

        tr_sum = 0

        # - Period HP/LP - #
        period_hp = periods[i_idx,10]
        period_lp = periods[i_idx,11]

        # - Previous Period Close Price - #
        period_pcp = periods[pi_idx,12]

        # - Period High minus Low - #
        period_hp_lp = abs(period_hp - period_lp)

        # - If first period of the day set PCP periods to 0 - #
        # - Previous period start date <> current period start date - #
        if periods[pi_idx,0] != periods[i_idx,0]:

            period_hp_pcp = 0
            period_lp_pcp = 0

        else:

            # - Period High minus Pre Close Price - #
            period_hp_pcp = abs(period_hp - period_pcp)

            # - Period Low minus Pre Close Price - #
            period_lp_pcp = abs(period_lp - period_pcp)

        # - TR is max of 3 range variables - #
        period_tr = max(period_hp_lp,period_hp_pcp,period_lp_pcp)

        # -- output -- #
        periods[i_idx,13] = period_pcp
        periods[i_idx,14] = period_hp_lp
        periods[i_idx,15] = period_hp_pcp
        periods[i_idx,16] = period_lp_pcp
        periods[i_idx,17] = period_tr

# --- ATR CUDA --- #
@nb.cuda.jit()
def atr_cuda(periods, periods_atr):

    # - i/j threads & size
    i_thread = nb.cuda.grid(1)
    i_size = nb.cuda.gridsize(1)

    total_periods = periods.shape[0]

    # -- Loop all periods -- #
    for i_idx in range(i_thread, total_periods, i_size):

        i_sum = 0
        i_count = 0

        # -- period st/ed idx
        p_sidx = i_idx
        p_eidx = p_sidx + periods_atr

        # -- loop all bars in period -- #
        for p_idx in range(p_sidx, p_eidx):

            # -- set out idx to next period -- #
            p_out_idx = p_idx + 1

            # -- period tr -- #
            i_sum += periods[p_idx, 17]

            i_count += 1

            # -- if last bar in period -- #
            if p_idx == p_eidx - 1:

                # -- calc avg -- #
                i_avg = i_sum / i_count

                # -- outut -- #
                periods[p_out_idx,18] = i_avg

# --- ATR TO BARS CUDA --- #
@nb.cuda.jit()
def atr_bars_cuda(bars, periods):

    # - i/j threads & size
    i_thread = nb.cuda.grid(1)
    i_size = nb.cuda.gridsize(1)

    # - Total bars in dataset -
    total_bars = bars.shape[0]
    total_periods = periods.shape[0]

    # -- Loop all periods -- #
    for p_idx in range(i_thread, total_periods, i_size):

        pb_sidx = int(periods[p_idx,5])
        pb_eidx = int(periods[p_idx,6])

        # -- get period atr -- #
        period_atr = periods[p_idx, 18]

        # -- loop bars in period to bars -- #
        for pb_idx in range(pb_sidx, pb_eidx + 1):

            # -- set bar atr to period atr -- #
            bars[pb_idx, 11] = period_atr

# --- Bars Out Setup CUDA --- #
@nb.cuda.jit()
def bars_out_cuda(bars, config):

    # - i/j threads & size
    i_thread = nb.cuda.grid(1)
    i_size = nb.cuda.gridsize(1)

    total_bars = bars.shape[0]

    r_fa = config[4]
    bars_period = config[12]
    mins_period = config[2]

    max_days = config[11]
    max_bars = config[19]

    total_periods = config[16]

    long_tpt = int(config[9])
    short_tpt = int(config[10])

    long_tg_fa = config[5]
    long_sl_fa = config[6]
    short_tg_fa = config[7]
    short_sl_fa = config[8]

    last_idx = bars[-1,0]

    # -- Loop all bars -- #
    for i_idx in range(i_thread, total_bars, i_size):

        # -- get bar details -- #
        period_atr = bars[i_idx, 11]
        bar_r = period_atr * r_fa

        bar_long_trp = bars[i_idx, long_tpt]
        bar_short_trp = bars[i_idx, short_tpt]

        bar_long_tg_r = bar_r * long_tg_fa
        bar_long_sl_r = bar_r * long_sl_fa
        bar_short_tg_r = bar_r * short_tg_fa
        bar_short_sl_r = bar_r * short_sl_fa

        bar_long_tg_pr = bar_long_trp + bar_long_tg_r
        bar_long_sl_pr = bar_long_trp - bar_long_sl_r
        bar_short_tg_pr = bar_short_trp - bar_short_tg_r
        bar_short_sl_pr = bar_short_trp + bar_short_sl_r

        max_idx = i_idx + max_bars

        trade_on = 1

        if max_idx > last_idx or period_atr == 0:

            bar_r = 0

            bar_long_trp = 0
            bar_short_trp = 0
            bar_long_tg_r = 0
            bar_long_sl_r = 0
            bar_short_tg_r = 0
            bar_short_sl_r = 0

            bar_long_tg_pr = 0
            bar_long_sl_pr = 0
            bar_short_tg_pr = 0
            bar_short_sl_pr = 0

            trade_on = 0

        # ---------- Save Output ---------- #

        bars[i_idx, 12] = bar_r # -- bar R -- #

        bars[i_idx, 13] = bar_long_trp # -- long trade price -- #
        bars[i_idx, 14] = bar_short_trp # -- short trade price -- #

        bars[i_idx, 15] = bar_long_tg_r # -- long target R -- #
        bars[i_idx, 16] = bar_long_sl_r # -- long stop loss R -- #

        bars[i_idx, 17] = bar_short_tg_r # -- short target R -- #
        bars[i_idx, 18] = bar_short_sl_r # -- short stop loss R -- #

        bars[i_idx, 19] = bar_long_tg_pr # -- long target price -- #
        bars[i_idx, 20] = bar_long_sl_pr # -- long stop loss price -- #

        bars[i_idx, 21] = bar_short_tg_pr # -- short target price -- #
        bars[i_idx, 22] = bar_short_sl_pr # -- short stop loss price -- #

        bars[i_idx, 23] = max_idx # -- max exit idx -- #

        bars[i_idx, -1] = trade_on # -- is_tradable -- #

# --- Walk CUDA --- #
@nb.cuda.jit()
def walk_cuda(bars, config):

    # - i/j threads & size
    i_thread = nb.cuda.grid(1)
    i_size = nb.cuda.gridsize(1)

    bars_size = bars.shape[0]

    last_idx = bars[-1,0]

    # .............. Bars Loop .......... #
    for i_idx in range(i_thread, bars_size, i_size):

        trade_on = bars[i_idx, -1]

        bar_r = bars[i_idx, 12]

        if trade_on:

            bar_ts = bars[i_idx,1]
            bar_time = bars[i_idx,2]
            bar_date = bars[i_idx,3]

            bar_op = bars[i_idx,6]
            bar_hp = bars[i_idx,7]
            bar_lp = bars[i_idx,8]
            bar_cp = bars[i_idx,9]
            bar_tp = bars[i_idx,10]

            bar_long_trp = bars[i_idx,13]
            bar_short_trp = bars[i_idx,14]

            bar_long_tg_r = bars[i_idx, 15]
            bar_long_sl_r = bars[i_idx,16]
            bar_short_tg_r = bars[i_idx,17]
            bar_short_sl_r = bars[i_idx,18]

            bar_long_tg = bars[i_idx,19]
            bar_long_sl = bars[i_idx,20]
            bar_short_tg = bars[i_idx,21]
            bar_short_sl = bars[i_idx,22]

            max_idx = bars[i_idx, 23]

            # -------- Range Loop Setup -------- #

            rng_bar_ts = bar_ts
            rng_bar_time = bar_time
            rng_bar_date = bar_date

            rng_bar_op = bar_op
            rng_bar_hp = bar_hp
            rng_bar_lp = bar_lp
            rng_bar_cp = bar_cp

            # -------- Long -------- #

            rng_long_tg_hit = 0
            rng_long_tg_idx = 0
            rng_long_tg_ts = 0
            rng_long_tg_time = 0
            rng_long_tg_date = 0

            rng_long_tg_pr = 0
            rng_long_tg_bc = 0

            rng_long_sl_hit = 0
            rng_long_sl_idx = 0
            rng_long_sl_ts = 0
            rng_long_sl_time = 0
            rng_long_sl_date = 0

            rng_long_sl_pr = 0
            rng_long_sl_bc = 0

            # -------- Short -------- #

            rng_short_tg_hit = 0
            rng_short_tg_idx = 0
            rng_short_tg_ts = 0
            rng_short_tg_time = 0
            rng_short_tg_date = 0

            rng_short_tg_pr = 0
            rng_short_tg_bc = 0

            rng_short_sl_hit = 0
            rng_short_sl_idx = 0
            rng_short_sl_ts = 0
            rng_short_sl_time = 0
            rng_short_sl_date = 0

            rng_short_sl_pr = 0
            rng_short_sl_bc = 0

            # -------- Close -------- #

            rng_long_close_ty = 0
            rng_long_close_idx = 0
            rng_long_close_ts = 0
            rng_long_close_time = 0
            rng_long_close_date = 0

            rng_long_close_pr = 0
            rng_long_close_bc = 0
            rng_long_close_net = 0
            rng_long_close_pct = 0
            rng_long_close_r = 0

            rng_short_close_ty = 0
            rng_short_close_idx = 0
            rng_short_close_ts = 0
            rng_short_close_time = 0
            rng_short_close_date = 0

            rng_short_close_pr = 0
            rng_short_close_bc = 0
            rng_short_close_net = 0
            rng_short_close_pct = 0
            rng_short_close_r = 0

            rng_counter = 0

            # ........ Rng Loop ........ #
            for rng_idx in range(i_idx, max_idx+1):

                # - Rng Bar Current Values - #
                rng_bar_idx = bars[rng_idx,0]
                rng_bar_ts = bars[rng_idx,1]
                rng_bar_time = bars[rng_idx,2]
                rng_bar_date = bars[rng_idx,3]

                rng_bar_op = bars[rng_idx,6]
                rng_bar_hp = bars[rng_idx,7]
                rng_bar_lp = bars[rng_idx,8]
                rng_bar_cp = bars[rng_idx,9]
                rng_bar_tp = bars[rng_idx,10]

                # ......... Long TG ......... #

                # - If long tg has R and tg not already hit - #
                if (rng_long_tg_hit == 0) and (bar_long_tg_r > 0):

                    # - If rng bar gte long target - #
                    if rng_bar_hp >= bar_long_tg:

                        # - Flag TG Hit - #
                        rng_long_tg_hit = 1
                        rng_long_tg_idx = rng_bar_idx
                        rng_long_tg_ts = rng_bar_ts
                        rng_long_tg_time = rng_bar_time
                        rng_long_tg_date = rng_bar_date
                        rng_long_tg_pr = rng_bar_hp
                        rng_long_tg_bc = rng_counter

                # ......... Long SL ......... #

                # - If long sl has R and sl not already hit - #
                if (rng_long_sl_hit == 0) and (bar_long_sl_r > 0):

                    # - If cur rng bar lp is lte long sl
                    if rng_bar_lp <= bar_long_sl:

                        # - Flag SL Hit - #
                        rng_long_sl_hit = 1
                        rng_long_sl_idx = rng_bar_idx
                        rng_long_sl_ts = rng_bar_ts

                        rng_long_sl_time = rng_bar_time
                        rng_long_sl_date = rng_bar_date

                        rng_long_sl_pr = rng_bar_lp
                        rng_long_sl_bc = rng_counter

                # ......... Short TG ......... #

                # - If short tg has R and tg not already hit - #
                if (rng_short_tg_hit == 0) and (bar_short_tg_r > 0):

                    # - If rng bar lp lte short target - #
                    if rng_bar_lp <= bar_short_tg:

                        # - Flag TG Hit - #
                        rng_short_tg_hit = 1
                        rng_short_tg_idx = rng_bar_idx
                        rng_short_tg_ts = rng_bar_ts
                        rng_short_tg_time = rng_bar_time
                        rng_short_tg_date = rng_bar_date

                        rng_short_tg_pr = rng_bar_lp
                        rng_short_tg_bc = rng_counter

                # ......... Short SL ......... #

                # - If short sl has R and sl not already hit - #
                if (rng_short_sl_hit == 0) and (bar_short_sl_r > 0):

                    # - If cur rng bar hp is gte short sl
                    if rng_bar_hp >= bar_short_sl:

                        # - Flag SL Hit - #
                        rng_short_sl_hit = 1
                        rng_short_sl_idx = rng_bar_idx
                        rng_short_sl_ts = rng_bar_ts
                        rng_short_sl_time = rng_bar_time
                        rng_short_sl_date = rng_bar_date

                        rng_short_sl_pr = rng_bar_hp
                        rng_short_sl_bc = rng_counter

                # ......... Max Rng Hits ......... #

                if (rng_idx == max_idx):

                    # :::::::::: Long :::::::::: #

                    # -- Timeout: Long TG and SL not hit -- #
                    if (rng_long_tg_hit == 0) & (rng_long_sl_hit == 0):

                        rng_long_close_ty = 3
                        rng_long_close_idx = rng_bar_idx
                        rng_long_close_ts = rng_bar_ts
                        rng_long_close_time = rng_bar_time
                        rng_long_close_date = rng_bar_date

                        rng_long_close_pr = rng_bar_cp
                        rng_long_close_bc = rng_counter

                    # -- Long TG hit (SL not hit) -- #
                    if (rng_long_tg_hit == 1) and (rng_long_sl_hit == 0):

                        rng_long_close_ty = 1
                        rng_long_close_idx = rng_long_tg_idx
                        rng_long_close_ts = rng_long_tg_ts
                        rng_long_close_time = rng_long_tg_time
                        rng_long_close_date = rng_long_tg_date

                        rng_long_close_pr = rng_long_tg_pr
                        rng_long_close_bc = rng_long_tg_bc

                    # -- Long SL hit (TG not hit) -- #
                    if (rng_long_sl_hit == 1) and (rng_long_tg_hit == 0):

                        rng_long_close_ty = 2
                        rng_long_close_idx = rng_long_sl_idx
                        rng_long_close_ts = rng_long_sl_ts
                        rng_long_close_time = rng_long_sl_time
                        rng_long_close_date = rng_long_sl_date

                        rng_long_close_pr = rng_long_sl_pr
                        rng_long_close_bc = rng_long_sl_bc

                    # -- Both TG and SL hit -- #
                    if (rng_long_tg_hit == 1) and (rng_long_sl_hit == 1):

                        if rng_long_tg_idx < rng_long_sl_idx:

                            rng_long_close_ty = 1
                            rng_long_close_idx = rng_long_tg_idx
                            rng_long_close_ts = rng_long_tg_ts
                            rng_long_close_time = rng_long_tg_time
                            rng_long_close_date = rng_long_tg_date

                            rng_long_close_pr = rng_long_tg_pr
                            rng_long_close_bc = rng_long_tg_bc

                        if rng_long_tg_idx > rng_long_sl_idx:

                            rng_long_close_ty = 2
                            rng_long_close_idx = rng_long_sl_idx
                            rng_long_close_ts = rng_long_sl_ts
                            rng_long_close_time = rng_long_sl_time
                            rng_long_close_date = rng_long_sl_date

                            rng_long_close_pr = rng_long_sl_pr
                            rng_long_close_bc = rng_long_sl_bc

                    # -- Long Close Net/Pct/R -- #
                    rng_long_close_net = rng_long_close_pr - bar_long_trp
                    rng_long_close_pct = (rng_long_close_net / bar_long_trp) * 100
                    rng_long_close_r = rng_long_close_net / bar_long_sl_r

                    # :::::::::: Short :::::::::: #

                    # -- Timeout: Short TG and SL not hit -- #
                    if (rng_short_tg_hit == 0) & (rng_short_sl_hit == 0):

                        rng_short_close_ty = 3
                        rng_short_close_idx = rng_bar_idx
                        rng_short_close_ts = rng_bar_ts
                        rng_short_close_time = rng_bar_time
                        rng_short_close_date = rng_bar_date

                        rng_short_close_pr = rng_bar_cp
                        rng_short_close_bc = rng_counter

                    # -- Short TG hit (SL not hit) -- #
                    if (rng_short_tg_hit == 1) and (rng_short_sl_hit == 0):

                        rng_short_close_ty = 1
                        rng_short_close_idx = rng_short_tg_idx
                        rng_short_close_ts = rng_short_tg_ts
                        rng_short_close_time = rng_short_tg_time
                        rng_short_close_date = rng_short_tg_date

                        rng_short_close_pr = rng_short_tg_pr
                        rng_short_close_bc = rng_short_tg_bc

                    # -- Short SL hit (TG not hit) -- #
                    if (rng_short_sl_hit == 1) and (rng_short_tg_hit == 0):

                        rng_short_close_ty = 2
                        rng_short_close_idx = rng_short_sl_idx
                        rng_short_close_ts = rng_short_sl_ts
                        rng_short_close_time = rng_short_sl_time
                        rng_short_close_date = rng_short_sl_date

                        rng_short_close_pr = rng_short_sl_pr
                        rng_short_close_bc = rng_short_sl_bc

                    # -- Both TG and SL hit -- #
                    if (rng_short_tg_hit == 1) and (rng_short_sl_hit == 1):

                        if rng_short_tg_idx < rng_short_sl_idx:

                            rng_short_close_ty = 1
                            rng_short_close_idx = rng_short_tg_idx
                            rng_short_close_ts = rng_short_tg_ts
                            rng_short_close_time = rng_short_tg_time
                            rng_short_close_date = rng_short_tg_date

                            rng_short_close_pr = rng_short_tg_pr
                            rng_short_close_bc = rng_short_tg_bc

                        if rng_short_tg_idx > rng_short_sl_idx:

                            rng_short_close_ty = 2
                            rng_short_close_idx = rng_short_sl_idx
                            rng_short_close_ts = rng_short_sl_ts
                            rng_short_close_time = rng_short_sl_time
                            rng_short_close_date = rng_short_sl_date

                            rng_short_close_pr = rng_short_sl_pr
                            rng_short_close_bc = rng_short_sl_bc

                    # -- Short Close Net/Pct/R -- #
                    rng_short_close_net = bar_short_trp - rng_short_close_pr
                    rng_short_close_pct = (rng_short_close_net / bar_short_trp) * 100
                    rng_short_close_r = rng_short_close_net / bar_short_sl_r

                    # ......... Save Output ......... #

                    # - Long TG Values - #
                    bars[i_idx,24] = rng_long_tg_hit
                    bars[i_idx,25] = rng_long_tg_idx
                    bars[i_idx,26] = rng_long_tg_ts
                    bars[i_idx,27] = rng_long_tg_time
                    bars[i_idx,28] = rng_long_tg_date

                    # - Long SL Values - #
                    bars[i_idx,29] = rng_long_sl_hit
                    bars[i_idx,30] = rng_long_sl_idx
                    bars[i_idx,31] = rng_long_sl_ts
                    bars[i_idx,32] = rng_long_sl_time
                    bars[i_idx,33] = rng_long_sl_date

                    # - Short TG Values - #
                    bars[i_idx,34] = rng_short_tg_hit
                    bars[i_idx,35] = rng_short_tg_idx
                    bars[i_idx,36] = rng_short_tg_ts
                    bars[i_idx,37] = rng_short_tg_time
                    bars[i_idx,38] = rng_short_tg_date

                    # - Short SL Values - #
                    bars[i_idx,39] = rng_short_sl_hit
                    bars[i_idx,40] = rng_short_sl_idx
                    bars[i_idx,41] = rng_short_sl_ts
                    bars[i_idx,42] = rng_short_sl_time
                    bars[i_idx,43] = rng_short_sl_date

                    # - Long Close Values - #
                    bars[i_idx,44] = rng_long_close_ty
                    bars[i_idx,45] = rng_long_close_idx
                    bars[i_idx,46] = rng_long_close_ts
                    bars[i_idx,47] = rng_long_close_time
                    bars[i_idx,48] = rng_long_close_date

                    bars[i_idx,49] = rng_long_close_pr
                    bars[i_idx,50] = rng_long_close_bc
                    bars[i_idx,51] = rng_long_close_net
                    bars[i_idx,52] = rng_long_close_pct
                    bars[i_idx,53] = rng_long_close_r

                    # - Short Close Values - #
                    bars[i_idx,54] = rng_short_close_ty
                    bars[i_idx,55] = rng_short_close_idx
                    bars[i_idx,56] = rng_short_close_ts
                    bars[i_idx,57] = rng_short_close_time
                    bars[i_idx,58] = rng_short_close_date

                    bars[i_idx,59] = rng_short_close_pr
                    bars[i_idx,60] = rng_short_close_bc
                    bars[i_idx,61] = rng_short_close_net
                    bars[i_idx,62] = rng_short_close_pct
                    bars[i_idx,63] = rng_short_close_r

                rng_counter += 1

                # ------- Close Types ------- #
                # - 1: TG Hit - #
                # - 2: SL Hit - #
                # - 3: Timeout - #

# ------------------- setup ------------------- #

blocks = (6144)
threads = (256)

fp_name = 'SPY_1T'
fp_dir = '/oxygen/krypton/hydrogen/'

fp = plasma.get_fp(fp_name, fp_dir,'.parquet')

# --- cols --- #

in_cols = ['bar_idx', 'bar_ts', 'bar_time', 'bar_date', 'bar_year', 'bar_eod',
           'bar_op', 'bar_hp', 'bar_lp', 'bar_cp', 'bar_tp', 'period_atr']

periods_cols = ['period_date',
                'period_sidx', 'period_eidx',
                'period_sts', 'period_ets',
                'pb_sidx', 'pb_eidx',
                'pb_sts', 'pb_ets',
                'period_op', 'period_hp', 'period_lp', 'period_cp',
                'period_pcp', 'period_hp_lp', 'period_hp_pcp',
                'period_lp_pcp', 'period_tr', 'period_atr']

ext_cols = ['bar_r',
            'bar_long_trp',
            'bar_short_trp',

            'bar_long_tg_r',
            'bar_long_sl_r',
            'bar_short_tg_r',
            'bar_short_sl_r',

            'bar_long_tg',
            'bar_long_sl',
            'bar_short_tg',
            'bar_short_sl',

            'max_idx',

            'long_tg_hit',
            'long_tg_idx',
            'long_tg_ts',
            'long_tg_time',
            'long_tg_date',

            'long_sl_hit',
            'long_sl_idx',
            'long_sl_ts',
            'long_sl_time',
            'long_sl_date',

            'short_tg_hit',
            'short_tg_idx',
            'short_tg_ts',
            'short_tg_time',
            'short_tg_date',

            'short_sl_hit',
            'short_sl_idx',
            'short_sl_ts',
            'short_sl_time',
            'short_sl_date',

            'long_close_ty',
            'long_close_idx',
            'long_close_ts',
            'long_close_time',
            'long_close_date',

            'long_close_pr',
            'long_close_bc',
            'long_close_net',
            'long_close_pct',
            'long_close_r',

            'short_close_ty',
            'short_close_idx',
            'short_close_ts',
            'short_close_time',
            'short_close_date',

            'short_close_pr',
            'short_close_bc',
            'short_close_net',
            'short_close_pct',
            'short_close_r',

            'trade_on']

out_cols = in_cols + ext_cols

# --- config --- #
config = {'year':0,
          'bar_res': 60,
          'period_mins': 390,

          'atr_days': 15,
          'r_fa': 1.0,

          'long_tg_fa': 2.5,
          'long_sl_fa': 1.5,
          'short_tg_fa': 2.5,
          'short_sl_fa': 1.5,

          'long_tpt': 10,
          'short_tpt': 10,

          'max_days': 7}

# ..... GET BARS ..... #
bars_all = cd.io.parquet.read_parquet(fp)
bars_all['bar_ts'] = bars_all['bar_id']
bars_all = bars_all.assign(period_atr=0.)

bars_data = bars_all.copy()

bars_data = bars_data[in_cols]

# -- period/day config --#
total_bars = bars_data.shape[0]

bars_per_day = 23400 / config["bar_res"]
bars_per_min = 60 / config["bar_res"]

bars_per_period = float(config["period_mins"] * bars_per_min)
periods_per_day = bars_per_day / bars_per_period
total_periods = total_bars / bars_per_period
periods_atr = periods_per_day * config['atr_days']
bars_atr = bars_per_day * config['atr_days']

max_mins = float(config['max_days'] * 390)
max_bars = max_mins * bars_per_min

# -- send bars data to device -- #
bars_data = bars_data.as_gpu_matrix()

# -- update config data -- #
config.update({'bars_per_period': bars_per_period,
                'bars_per_day': bars_per_day,
                'periods_per_day': periods_per_day,
                'total_bars': total_bars,
                'total_periods': total_periods,
                'periods_atr': periods_atr,
                'max_mins': max_mins,
                'max_bars': max_bars})

# -- write config data to disk and get back config id -- #
config.update({'config_id':
                plasma.insert_quark_config(pd.DataFrame([config]),
                'quark_config', 'quark_config', nf=0)})

# -- send config to cuda -- #
config_data = nb.cuda.to_device(np.array(list(config.values())))

# -- setup blank periods data for output -- #
periods_data = nb.cuda.device_array([int(total_periods), len(periods_cols)])

# ... call PERIODS kernel ... #
# --- creates periods --- #
periods_cuda[blocks,threads](bars_data, periods_data, config_data); nb.cuda.synchronize()

# ... call TR kernel ... #
# --- calcs and adds true range to periods --- #
tr_cuda[blocks, threads](periods_data); nb.cuda.synchronize()

# ... call ATR kernel ... #
# --- calcs and adds avg true range to periods --- #
atr_cuda[blocks, threads](periods_data, periods_atr); nb.cuda.synchronize()

# ... call ATR TO BARS kernel ... #
# --- adds period ATR to bars --- #
atr_bars_cuda[blocks, threads](bars_data, periods_data); nb.cuda.synchronize()

# --- create bars out --- #
# --- concats bars_data with zeros area needed for extra cols --- #
bars_out_np = np.concatenate((bars_data,np.empty([int(bars_data.shape[0]),
                                                  len(ext_cols)])),axis=1)

# --- send bars_out to gpu --- #
bars_out = nb.cuda.to_device(bars_out_np)

# ... call BARS OUT kernel ... #
# --- calcs R, target and SL values --- #
bars_out_cuda[blocks, threads](bars_out, config_data); nb.cuda.synchronize()

# ... call WALK CUDA kernel ... #
# --- after all all required values are calculated walk the range  --- #
walk_cuda[blocks, threads](bars_out, config_data); nb.cuda.synchronize()

periods_data = periods_data.copy_to_host()
periods_df = pd.DataFrame(periods_data,columns=periods_cols)

bars_data = bars_data.copy_to_host()
bars_data_df = pd.DataFrame(bars_data,columns=in_cols)

bars_out = pd.DataFrame(bars_out.copy_to_host(), columns=out_cols)

bars_all = bars_all.to_pandas()

bars_cols = list(bars_out.columns)
all_cols = list(bars_all.columns)

l_cols = ['bar_idx'] + list(set(all_cols) - set(bars_cols))

bars_merge = bars_all[l_cols].merge(bars_out, left_on='bar_idx', right_on='bar_idx')

merge_cols = ['bar_idx',
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
              'bar_r',
              'bar_long_trp',
              'bar_short_trp',
              'bar_long_tg_r',
              'bar_long_sl_r',
              'bar_short_tg_r',
              'bar_short_sl_r',
              'bar_long_tg',
              'bar_long_sl',
              'bar_short_tg',
              'bar_short_sl',
              'max_idx',
              'long_tg_hit',
              'long_tg_idx',
              'long_tg_ts',
              'long_tg_time',
              'long_tg_date',
              'long_sl_hit',
              'long_sl_idx',
              'long_sl_ts',
              'long_sl_time',
              'long_sl_date',
              'short_tg_hit',
              'short_tg_idx',
              'short_tg_ts',
              'short_tg_time',
              'short_tg_date',
              'short_sl_hit',
              'short_sl_idx',
              'short_sl_ts',
              'short_sl_time',
              'short_sl_date',
              'long_close_ty',
              'long_close_idx',
              'long_close_ts',
              'long_close_time',
              'long_close_date',
              'long_close_pr',
              'long_close_bc',
              'long_close_net',
              'long_close_pct',
              'long_close_r',
              'short_close_ty',
              'short_close_idx',
              'short_close_ts',
              'short_close_time',
              'short_close_date',
              'short_close_pr',
              'short_close_bc',
              'short_close_net',
              'short_close_pct',
              'short_close_r',
              'is_fake',
              'trade_on']

bars_merge_out = delta.id_to_dt(bars_merge[merge_cols])
