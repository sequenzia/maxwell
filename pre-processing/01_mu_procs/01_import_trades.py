import sys, warnings, pytz
warnings.filterwarnings("ignore")
sys.path.append('/var/lib/alpha/quantum/_maxwell/utils/')

import plasma, pandas as pd, numpy as np, pyarrow as pa
from pyarrow import parquet as pq

est = pytz.timezone('US/Eastern')

yr = '2009'

fn = 'spy_trades_' + str(yr)
fp = plasma.get_fp(fn,'hydrogen/lepton/trades/imports','csv')
fp_out = plasma.get_fp('spy_trades_dev','hydrogen/lepton/trades')

cols = ['trade_date','trade_time','trade_price','trade_volume','sales_cond','alt_price','dt']

col_types = {"trade_date":pa.string(),
             "trade_time":pa.string(),
             "trade_price":pa.float64(),
             "trade_volume":pa.float32(),
             "sales_cond":pa.string(),
             "alt_price":pa.float64(),
             "dt":pa.string()}

convert_opts = csv.ConvertOptions(include_columns=cols,column_types=col_types)
trades_table = csv.read_csv(fp,convert_options=convert_opts)
trades_df = trades_table.to_pandas().fillna(0)

del trades_table

# -- Remove all 7 V trades -- #
trades_df = trades_df[trades_df['sales_cond'] != '7 V']

# -- Remove all U trades (out of sequence after hours) -- #
trades_df = trades_df[trades_df['sales_cond'] != 'U']

# -- Update Trade Price to Alt Price -- #
trades_df.loc[trades_df['alt_price'] != 0, 'trade_price'] = trades_df['alt_price']

# -- add cols -- #
trades_df['trade_dt'] = pd.to_datetime(trades_df['dt'], unit='ns').dt.tz_localize(est)
trades_df['trade_ts'] = trades_df['trade_dt'].values.astype(np.float64)
trades_df['trade_date'] = trades_df['trade_date'].str.replace('-','').astype(np.float64)
trades_df['trade_time'] = trades_df['trade_time'].str.replace(':','').astype(np.float64)
trades_df['trade_year'] = trades_df['trade_dt'].dt.year.astype(np.float64)
trades_df['trade_month'] = trades_df['trade_dt'].dt.month.astype(np.float64)
trades_df['trade_day'] = trades_df['trade_dt'].dt.day.astype(np.float64)
trades_df['trade_hour'] = trades_df['trade_dt'].dt.hour.astype(np.float64)
trades_df['trade_min'] = trades_df['trade_dt'].dt.minute.astype(np.float64)
trades_df['trade_sec'] = trades_df['trade_dt'].dt.second.astype(np.float64)
trades_df['trade_ns'] = trades_df['trade_time'].astype(np.float64).apply(plasma.ti_to_float_ns)

# -- drop cols -- #
trades_df = trades_df.drop('trade_dt',axis=1)
trades_df = trades_df.drop('dt',axis=1)

cols_out = ['trade_ts', 'trade_date', 'trade_time', 'trade_year', 'trade_month',
            'trade_day', 'trade_hour', 'trade_min', 'trade_sec','trade_ns',
            'trade_price', 'trade_volume','sales_cond']

tab_out = pa.Table.from_pandas(trades_df[cols_out],preserve_index=False)

pq.write_to_dataset(tab_out, fp_out, partition_cols=['trade_year','trade_month'])

