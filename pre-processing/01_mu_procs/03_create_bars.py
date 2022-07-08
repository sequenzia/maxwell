import sys, os, pytz, math
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa, cudf as cd
from pyarrow import parquet as pq
from numba import cuda, jit, types

est = pytz.timezone('US/Eastern')

''' ::::::::::::: Converts trades to Bars :::::::::::::
    Loops over every trade #
    Corrects any missing bars 
'''

# ------------- mu trades to bars ----------------- #
@cuda.jit()
def run_t2b(trades_in,bars_out):

    i_thread = cuda.grid(1)
    i_size = cuda.gridsize(1)

    trades_size = trades_in.shape[0]
    st_ts = trades_in[0,0]

    last_out_idx = 0

    # - Loop every trade - #
    for t_idx in range(i_thread, trades_size+1, i_size):

        is_ft = 0
        is_ut = 0
        is_dt = 0
        
        # - t m1 vars (one trade back) - #
        tm1_idx = t_idx - 1
        tm1_sec = trades_in[tm1_idx,8]
        tm1_price = trades_in[tm1_idx,10]
        tm1_ts = trades_in[tm1_idx,0]

        # - t vars - #
        t_sec = trades_in[t_idx,8]
        t_price = trades_in[t_idx,10]
        t_vol = trades_in[t_idx,11]
        t_ts = trades_in[t_idx,0]
        t_mu_flag = trades_in[t_idx,12]

        # --- If the first trade set flat if not check ---- #
        if t_idx == 0:

            is_ft = 1

        else:

            if t_price == tm1_price:

                is_ft = 1

            if t_price > tm1_price:

                is_ut = 1

            if t_price < tm1_price:

                is_dt = 1
        
        # .............  First trade of second ............. #
        if t_sec != tm1_sec:

            # - Bar prices - #
            t_op = t_price
            t_hp = t_price
            t_lp = t_price
            t_cp = t_price
            t_vol = t_vol

            is_skip = 0  
            
            # - trade direction counts setup - #

            if is_ft == 1:

                ft_count = 1

            else:

                ft_count = 0

            if is_ut == 1:

                ut_count = 1

            else:

                ut_count = 0
            
            if is_dt == 1:

                dt_count = 1

            else:
                
                dt_count = 0

            tt_count = 1

            rng_st_idx = t_idx + 1

            if rng_st_idx < trades_size:

                # ------------- Rng till next second ------------ #
                for rng_idx in range(rng_st_idx,t_idx+1500):

                    if rng_idx > trades_size:

                        print(rng_idx,trades_size)

                        break
                    
                    # - M/P idx - #
                    rng_m1_idx = rng_idx - 1
                    rng_p1_idx = rng_idx + 1

                    # - Rng Seconds - #
                    rng_sec = trades_in[rng_idx,8]
                    rng_m1_sec = trades_in[rng_m1_idx,8]
                    rng_m1_sec_diff = rng_sec - rng_m1_sec

                    # -- Rng m1 vars -- #
                    rng_m1_day = trades_in[rng_m1_idx,5]
                    rng_m1_price = trades_in[rng_m1_idx,10]
                    rng_m1_ts = trades_in[rng_m1_idx,0]

                    # -- Rng p1 vars -- #
                    rng_p1_sec = trades_in[rng_p1_idx,8]

                    # -- Rng vars -- #
                    rng_date = trades_in[rng_idx,1]
                    rng_year = trades_in[rng_idx,3]
                    rng_month = trades_in[rng_idx,4]
                    rng_day = trades_in[rng_idx,5]
                    rng_hour = trades_in[rng_idx,6]
                    rng_min = trades_in[rng_idx,7]
                    rng_price = trades_in[rng_idx,10]
                    rng_vol = trades_in[rng_idx,11]

                    rng_ts = trades_in[rng_idx,0]
                    out_idx = int((rng_ts - st_ts) // 1000000000)             
                    
                    # - If increase set HP - #
                    if rng_price > t_hp:

                        t_hp = rng_price
                    
                    # - If decrease set LP - #
                    if rng_price < t_lp:

                        t_lp = rng_price

                    t_vol += rng_vol

                    # - Is flat trade - #
                    if rng_price == rng_m1_price:

                        # - Inc flat count - #
                        ft_count += 1

                    else:

                        # - Is up trade - #
                        if rng_price > rng_m1_price:

                            # - Inc up count - #
                            ut_count += 1

                        # - Is down trade - #
                        if rng_price < rng_m1_price:

                            # - Inc down count - #
                            dt_count += 1  

                    # - Inc total count - #
                    tt_count += 1
                    
                    # --------- Last trade of sec------------ #
                    if rng_sec != rng_p1_sec:

                        # - Set close price - #
                        t_cp = trades_in[rng_idx,10]

                        # --- Write to output --- #

                        bars_out[out_idx,9] = t_op
                        bars_out[out_idx,10] = t_hp
                        bars_out[out_idx,11] = t_lp
                        bars_out[out_idx,12] = t_cp
                        bars_out[out_idx,13] = t_vol

                        bars_out[out_idx,14] = ft_count
                        bars_out[out_idx,15] = ut_count
                        bars_out[out_idx,16] = dt_count
                        bars_out[out_idx,17] = tt_count
                        bars_out[out_idx,18] = 0.

                        break

# ---------------- mu trades to bars run proc --------------------------#
def t2b_proc(day_trades):

    blocks = (256)
    threads = (256)

    mb_blocks = (1)
    mb_threads = (1)

    out_cols = ['bar_fd',
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
                'bar_vol',

                'ft_count',
                'ut_count',
                'dt_count', 
                'tt_count',
                'is_fake']

    day_size = 23400

    day_st = pd.to_datetime({'year': [day_trades.iloc[0,3][0]],
                            'month': [day_trades.iloc[0,4][0]],
                            'day': [day_trades.iloc[0,5][0]],
                            'hour': int(9),
                            'minute': int(30),
                            'second': int(1)}).dt.tz_localize(est)
                        
    day_idx = pd.date_range(day_st[0],periods=day_size,freq='S')

    day_data = {'bar_fd': day_idx.strftime('%Y%m%d%H%M%S').tolist(),
                'bar_date': day_idx.strftime('%Y%m%d').tolist(), 
                'bar_time': day_idx.strftime('%H%M%S').tolist(), 
                'bar_year': day_idx.strftime('%Y').tolist(),
                'bar_month': day_idx.strftime('%m').tolist(),
                'bar_day': day_idx.strftime('%d').tolist(),
                'bar_hour': day_idx.strftime('%H').tolist(),
                'bar_min': day_idx.strftime('%M').tolist(),
                'bar_sec': day_idx.strftime('%S').tolist(),
                'bar_op': 0., 
                'bar_hp': 0.,
                'bar_lp': 0.,
                'bar_cp': 0.,
                'bar_vol': 0.,

                'ft_count': 0.,
                'ut_count': 0.,
                'dt_count': 0.,
                'tt_count': 0.,
                'is_fake':1.}

    day_df = pd.DataFrame(day_data,dtype=np.float)
    day_np = day_df.to_numpy(dtype=np.float64)
    d_bars_out = cuda.to_device(day_np)

    d_trades_in = day_trades.as_gpu_matrix()

    # .......................... Call Kernel ............................ #
    
    run_t2b[blocks, threads](d_trades_in,d_bars_out); cuda.synchronize()
    
    # --- Run through missing bars --- #
    missing_bars[mb_blocks, mb_threads](d_bars_out); cuda.synchronize()

    bars_out_df = pd.DataFrame(d_bars_out.copy_to_host(),columns=out_cols)

    return bars_out_df

@cuda.jit()
def missing_bars(bars):

    i_thread = cuda.grid(1)
    i_size = cuda.gridsize(1)

    bars_size = 23400

    # - Loop every trade - #
    for bar_idx in range(0, bars_size):

        bar_idx_b1 = bar_idx - 1
        bar_idx_f1 = bar_idx + 1

        is_fake = bars[bar_idx,18]
        b_time =bars[bar_idx,2]

        if is_fake == 1:

            if bar_idx == 0:
                
                n_op = bars[bar_idx_f1,9]
                n_hp = bars[bar_idx_f1,10]
                n_lp = bars[bar_idx_f1,11]
                n_cp = bars[bar_idx_f1,12]

            else:
                
                n_op = bars[bar_idx_b1,9]
                n_hp = bars[bar_idx_b1,10]
                n_lp = bars[bar_idx_b1,11]
                n_cp = bars[bar_idx_b1,12]

            bars[bar_idx,9] = n_op
            bars[bar_idx,10] = n_hp
            bars[bar_idx,11] = n_lp
            bars[bar_idx,12] = n_cp
            bars[bar_idx,13] = 0
            bars[bar_idx,14] = 0
            bars[bar_idx,15] = 0
            bars[bar_idx,16] = 0
            bars[bar_idx,17] = 0

yr_st = 2004
yr_ed = 2017

chk_bac_secs = 1
chk_fwd_secs = 1

cu_dev = cuda.get_current_device()

mu_df = pd.DataFrame()
out_df = pd.DataFrame()

# ------------- loop years ------------- #
for yr in range(yr_st, yr_ed):

    trades_fn = 'spy_trades_' + str(yr)
    trades_fp = plasma.get_fp(trades_fn,'hydrogen/muon/trades/','.parquet')

    trades_cols = ['trade_ts',
                    'trade_date',
                    'trade_time',
                    'trade_year',
                    'trade_month',
                    'trade_day',
                    'trade_hour',
                    'trade_min',
                    'trade_sec',
                    'trade_ns',
                    'trade_price',
                    'trade_volume',
                    'mu_flag']

    trades_tab = pq.read_table(trades_fp,columns=trades_cols)

    trades_df = trades_tab.to_pandas().astype(np.float64)

    days_df = trades_df[['trade_date']].groupby('trade_date',as_index=True).agg('min')

    days_len = days_df.shape[0]
    day_counter = 0
    data_df = pd.DataFrame()

    # -- Loop all days in year -- #
    for i in range(0,days_len):

        day_counter += 1

        d = days_df.index.values[i]

        if i > 0:
            ptd = days_df.index.values[i-1]
        else:
            ptd = 0

        print('\n----------------', d , '(', day_counter ,') ----------------\n')

        # -- Filter trades data down to day -- #
        day_trades = cu.from_pandas(trades_df[trades_cols][trades_df['trade_date'] == d])

        # -- Run T2B Proc -- #
        proc_df = t2b_proc(day_trades)
        proc_df['bar_dt'] = pd.to_datetime(proc_df['bar_fd'],
            format="%Y%m%d%H%M%S").dt.tz_localize(est)

        proc_df['bar_id'] = proc_df['bar_dt'].astype(np.int64) // 1e9
        proc_df['bar_id'] = proc_df['bar_id'].astype(np.float64)

        proc_df['bar_eod_dt'] = pd.to_datetime({'year': proc_df['bar_dt'].dt.year,
                                'month': proc_df['bar_dt'].dt.month,
                                'day': proc_df['bar_dt'].dt.day,
                                'hour': int(16)}).dt.tz_localize(est)

        proc_df['bar_eod'] = proc_df['bar_eod_dt'].astype(np.int64) // 1e9
        proc_df['bar_eod'] = proc_df['bar_eod'].astype(np.float64)

        proc_df['bar_ptd'] = ptd

        bars_cols = ['bar_dt','bar_id','bar_fd', 'bar_date', 'bar_time',
                    'bar_year', 'bar_month', 'bar_day','bar_hour', 'bar_min',
                    'bar_sec', 'bar_op', 'bar_hp', 'bar_lp','bar_cp', 'bar_vol',
                    'ft_count', 'ut_count', 'dt_count', 'tt_count','bar_ptd',
                    'bar_eod', 'is_fake']

        proc_df = proc_df[bars_cols]

        data_df = pd.concat([data_df,proc_df])

    data_df = data_df.sort_values('bar_id').reset_index(drop=True)

    bars_fn = 'spy_bars_' + str(yr)
    bars_fp = plasma.get_fp(bars_fn,'hydrogen/electron/bars/pre_tf')
    bars_tab = pa.Table.from_pandas(data_df, preserve_index=False)
    pa.parquet.write_table(bars_tab,bars_fp)