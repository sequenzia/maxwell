import sys, os, pytz, math
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa, cudf as cd
from pyarrow import parquet as pq
from numba import cuda, jit, types

est = pytz.timezone('US/Eastern')

trades_fp = plasma.get_fp('spy_trades','hydrogen/lepton/trades')
book_fp = plasma.get_fp('spy_book','hydrogen/lepton/book','.parquet')

# -------------------------------- Cols -------------------------------- #
trades_cols = ['trade_ts', 'trade_date', 'trade_time', 'trade_year', 'trade_month', 'trade_day', 'trade_hour', 'trade_min', 'trade_sec','trade_ns','trade_price', 'trade_volume','sales_cond']

book_cols = ['book_ts', 'book_time', 'bb_price', 'ba_price']

trades_in_cols = ['trade_idx','trade_ts','trade_price', 'tt_sidx','tt_eidx','tb_sidx','tb_eidx']

chk_cols = ['tt_rng_avg','tt_diff','tt_diff_pct',
            'tb_rng_avg','tb_diff','tb_diff_pct',
            'tt_is_bad','tb_is_bad','mu_flag','rng_idx',
            'old_price']

chk_out_cols = trades_in_cols + chk_cols

blocks = (1024)
threads = (256)

yr_st = 2004
yr_ed = 2017

chk_bac_secs = 1
chk_fwd_secs = 1

cu_dev = cuda.get_current_device()

mu_df = pd.DataFrame()
out_df = pd.DataFrame()

# ------------- loop years ------------- #
for yr in range(yr_st,yr_ed):

    # ------------- loop months ------------- #
    for mi in range(1,3):

        print('\n---------- Run Proc:',yr, mi,'----------\n')

        # ------------- trades data ------------- #
        trades_tab = pa.parquet.read_table(trades_fp,columns=trades_cols, filters=[('trade_year','=',float(yr)),
                                                                                   ('trade_month','=',float(mi))])
        trades_df = trades_tab.to_pandas()
        trades_df = trades_df.sort_values('trade_ts').reset_index(drop=True)[trades_cols]
        trades_df['trade_idx'] = trades_df.index
        trades_df['trade_dt'] = pd.to_datetime(trades_df['trade_ts'], unit='ns',utc=True).dt.tz_convert(est)

        # --- setup chk data to get base for checking trades --- #
        chk_group = trades_df[['trade_dt','trade_price']].set_index('trade_dt').resample('30S')

        chk_df = chk_group.mean()
        chk_df['pr_min'] = chk_group.min()
        chk_df['pr_max'] = chk_group.max()
        chk_df['pr_rng'] = chk_df['pr_max'] - chk_df['pr_min']
        chk_df['pr_rng_pct'] =  100*(chk_df['pr_rng'] / chk_df['trade_price'])
        chk_df = chk_df.dropna()

        chk_base = chk_df['pr_rng_pct'].mean() + chk_df['pr_rng_pct'].std()

        # ------------- book data ------------- #
        book_tab = pa.parquet.read_table(book_fp,columns=book_cols, filters=[
            ('book_year','=',float(yr)),
            ('book_month','=',float(mi))])

        book_df = book_tab.to_pandas().astype(np.float64)
        book_df = book_df.sort_values('book_ts').reset_index(drop=True)[book_cols]
        book_df['book_idx'] = book_df.index

        # -- link book to trades (all trade cols) -- #
        trades_df = pd.merge_asof(left=trades_df,
                                  right=book_df,
                                  left_on='trade_ts',
                                  right_on='book_ts',
                                  suffixes=('_tr','_bk'),
                                  allow_exact_matches=False)

        # ------------- chk setup ------------- #
        trades_df['chk_sts'] = trades_df['trade_ts'] - (1e9 * chk_bac_secs)
        trades_df['chk_sdt'] = pd.to_datetime(trades_df['chk_sts']
                                              ,unit='ns',utc=True).dt.tz_convert(est)

        trades_df['chk_ets'] = trades_df['trade_ts'] + (1e9 * chk_fwd_secs)
        trades_df['chk_edt'] = pd.to_datetime(trades_df['chk_ets']
                                              ,unit='ns',utc=True).dt.tz_convert(est)

        # ------------- trades_st_df ------------- #
        trades_st_df = pd.merge_asof(left=trades_df[['trade_idx','chk_sts']],
                                     right=trades_df[['trade_idx','trade_ts']],
                                     left_on='chk_sts',
                                     right_on='trade_ts',
                                     suffixes=('_l','_r'),
                                     direction='forward',
                                     allow_exact_matches=True)

        trades_st_df = trades_st_df[['trade_idx_l','trade_idx_r','trade_ts']]
        trades_st_df.columns = ['trade_idx','tt_sidx','tt_sts']

        # ------------- trades_ed_df ------------- #
        trades_ed_df = pd.merge_asof(left=trades_df[['trade_idx','chk_ets']],
                                     right=trades_df[['trade_idx','trade_ts']],
                                     left_on='chk_ets',
                                     right_on='trade_ts',
                                     suffixes=('_l','_r'),
                                     direction='backward',
                                     allow_exact_matches=True)

        trades_ed_df = trades_ed_df[['trade_idx_l','trade_idx_r','trade_ts']]
        trades_ed_df.columns = ['trade_idx','tt_eidx','tt_ets']

        # ------------- book_st_df ------------- #
        book_st_df = pd.merge_asof(left=trades_df[['trade_idx','chk_sts']],
                                   right=book_df[['book_idx','book_ts']],
                                   left_on='chk_sts',
                                   right_on='book_ts',
                                   suffixes=('_l','_r'),
                                   direction='forward',
                                   allow_exact_matches=True)

        book_st_df = book_st_df[['trade_idx','book_idx','book_ts']]
        book_st_df.columns = ['trade_idx','tb_sidx','tb_sts']

        # ------------- book_ed_df ------------- #
        book_ed_df = pd.merge_asof(left=trades_df[['trade_idx','chk_ets']],
                                   right=book_df[['book_idx','book_ts']],
                                   left_on='chk_ets',
                                   right_on='book_ts',
                                   suffixes=('_l','_r'),
                                   direction='backward',
                                   allow_exact_matches=True)

        book_ed_df = book_ed_df[['trade_idx','book_idx','book_ts']]
        book_ed_df.columns = ['trade_idx','tb_eidx','tb_ets']

        # ------------- full df : trades_st_df------------- #
        full_df = pd.merge(left=trades_df,
                           right=trades_st_df,
                           left_on='trade_idx',
                           right_on='trade_idx')

        # ------------- full df : trades_ed_df------------- #
        full_df = pd.merge(left=full_df,
                           right=trades_ed_df,
                           left_on='trade_idx',
                           right_on='trade_idx')

        # ------------- full df : book_st_df------------- #
        full_df = pd.merge(left=full_df,
                           right=book_st_df,
                           left_on='trade_idx',
                           right_on='trade_idx')

        # ------------- full df : trades_st_df------------- #
        full_df = pd.merge(left=full_df,
                           right=book_ed_df,
                           left_on='trade_idx',
                           right_on='trade_idx')

        trades_in_df = full_df[trades_in_cols].copy()

        # -- append chk cols -- #
        for col_idx in range(0,len(chk_cols)):

            col_val = chk_cols[col_idx]
            trades_in_df[col_val] = 0.

        # --- to gpu --- #
        trades_in = cuda.to_device(trades_in_df.to_numpy(dtype=np.float64))
        book_in = cuda.to_device(book_df.to_numpy(dtype=np.float64))

        # -------------- CUDA FUNCTIONS -------------------- #
        @cuda.jit()
        def chk_proc(trades_in,book_in):

            i_thread = cuda.grid(1)
            i_size = cuda.gridsize(1)

            # - total ticks - #
            trades_size = trades_in.shape[0]

            # - loop every tick - #
            for t_idx in range(i_thread, trades_size, i_size):

                trade_price = trades_in[t_idx][2]

                # ............ tt chk ............ #
                tt_sidx = int(trades_in[t_idx][3])
                tt_eidx = int(trades_in[t_idx][4])

                tt_rng_pr = 0
                tt_rng_counter = 0

                # -- tt rng chk loop -- #
                for tt_idx in range(tt_sidx,tt_eidx+1):

                    tt_rng_counter += 1
                    tt_rng_pr += trades_in[tt_idx][2]

                tt_rng_avg = tt_rng_pr / tt_rng_counter

                tt_diff = trade_price - tt_rng_avg
                tt_diff_pct = abs((tt_diff / trade_price) * 100)

                trades_in[t_idx][7] = tt_rng_avg
                trades_in[t_idx][8] = tt_diff
                trades_in[t_idx][9] = tt_diff_pct

                # ............ tb chk ............ #
                tb_sidx = int(trades_in[t_idx][5])
                tb_eidx = int(trades_in[t_idx][6])

                tb_rng_pr = 0
                tb_rng_counter = 0

                # -- tb rng chk loop -- #
                for tb_idx in range(tb_sidx,tb_eidx+1):

                    tb_rng_counter += 1

                    tb_rng_pr += (book_in[tb_idx][2] + book_in[tb_idx][3]) / 2

                tb_rng_avg = tb_rng_pr / tb_rng_counter

                tb_diff = trade_price - tb_rng_avg
                tb_diff_pct = abs((tb_diff / trade_price) * 100)

                trades_in[t_idx][10] = tb_rng_avg
                trades_in[t_idx][11] = tb_diff
                trades_in[t_idx][12] = tb_diff_pct

        @cuda.jit()
        def fix_pr(trades_in,chk_base):

            i_thread = cuda.grid(1)
            i_size = cuda.gridsize(1)

            # - total ticks - #
            trades_size = trades_in.shape[0]

            # - loop every tick - #
            for t_idx in range(i_thread, trades_size, i_size):

                # --- is tt bad -- #
                if trades_in[t_idx][9] >= chk_base:
                    tt_is_bad = True
                    trades_in[t_idx][13] = 1
                else:
                    tt_is_bad = False

                # --- is tb bad -- #
                if trades_in[t_idx][12] >= chk_base:
                    tb_is_bad = True
                    trades_in[t_idx][14] = 1
                else:
                    tb_is_bad = False

                # --- mu flag --- #
                if tt_is_bad & tb_is_bad:

                    mu_flag = 1
                    trades_in[t_idx][15] = 1

                    rng_pr = 0

                    for rng_idx in range(t_idx,0,-1):

                        if (trades_in[rng_idx][9] < chk_base) | \
                                (trades_in[rng_idx][12] < chk_base):

                            rng_pr = trades_in[rng_idx][2]

                            break

                    trades_in[t_idx][16] = rng_idx
                    trades_in[t_idx][17] = trades_in[t_idx][2]
                    trades_in[t_idx][2] = rng_pr

        # .............. call chk kernel  .............. #
        chk_proc[blocks, threads](trades_in,book_in); cuda.synchronize()
        fix_pr[blocks, threads](trades_in,chk_base); cuda.synchronize()

        chk_out_df = pd.DataFrame(trades_in.copy_to_host(),columns=chk_out_cols)

        l_cols = ['trade_idx','trade_ts','trade_date','trade_time',
                  'trade_year', 'trade_month','trade_day','trade_hour',
                  'trade_min', 'trade_sec','trade_ns','trade_volume',
                  'sales_cond']

        r_cols = ['trade_idx','trade_price','mu_flag']

        out_cols = ['trade_idx','trade_ts','trade_date','trade_time',
                    'trade_year', 'trade_month','trade_day','trade_hour',
                    'trade_min', 'trade_sec','trade_ns','trade_price',
                    'trade_volume','sales_cond','mu_flag']

        data_df = pd.merge(left=trades_df[l_cols],
                           right=chk_out_df[r_cols],
                           on='trade_idx')

        mu_df = pd.concat([mu_df,chk_out_df])
        out_df = pd.concat([out_df,data_df])

        del trades_tab
        del trades_df
        del chk_base
        del chk_group
        del chk_df
        del book_tab
        del book_df
        del trades_st_df
        del trades_ed_df
        del book_ed_df
        del book_st_df
        del trades_in
        del book_in
        del trades_in_df
        del full_df
        del chk_proc
        del chk_out_df
        del data_df

        cu_dev.reset()

mu_tab_out = pa.Table.from_pandas(mu_df,preserve_index=False)
trades_tab_out = pa.Table.from_pandas(out_df[out_cols],preserve_index=False)

fn_mu = 'spy_mu_' + str(yr)
fp_mu = plasma.get_fp(fn_mu,'hydrogen/muon/mu')
pa.parquet.write_table(mu_tab_out, fp_mu)

fn_out = 'spy_trades_' + str(yr)
fp_out = plasma.get_fp(fn_out,'hydrogen/muon/trades')
pa.parquet.write_table(trades_tab_out, fp_out)