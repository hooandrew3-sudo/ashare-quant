import pandas as pd

for d in ['083ed44c76d3', '6467e03ad13b']:
    t = pd.read_parquet(f'artifacts/20260821_175428_{d}/trades.parquet')
    print(f'\n=== {d} ===')
    print(f'  total trades: {len(t)}')
    print(f'  by reason:')
    print(t['reason'].value_counts().to_string())
    print(f'  by side:')
    print(t['side'].value_counts().to_string())
    # Rebalance frequency by month
    t['date'] = pd.to_datetime(t['date'])
    t['ym'] = t['date'].dt.to_period('M')
    rebal = t[t['reason'] == 'rebalance'].groupby('ym').size()
    print(f'  rebalance-months: {len(rebal)}')
    print(f'  rebal-months distribution (max trades in a month):')
    print(rebal.describe().round(1).to_string())
