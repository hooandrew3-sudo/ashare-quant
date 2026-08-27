import pandas as pd

for d in ['083ed44c76d3', '6467e03ad13b']:
    p = f'artifacts/20260821_175428_{d}'
    m = pd.read_parquet(f'{p}/monthly.parquet')
    t = pd.read_parquet(f'{p}/trades.parquet')
    print(f'\n=== {d} ===')
    print(f'  monthly rows: {len(m)}, trades rows: {len(t)}')
    print(f'  monthly cols: {list(m.columns)}')
    print(f'  trades cols: {list(t.columns)}')
    if 'n_trades' in m.columns:
        active_months = (m['n_trades'] > 0).sum()
        print(f'  rebalance months (n_trades>0): {active_months}')
    print(f'  monthly head:')
    print(m.head(3).to_string())
    print(f'  trades head:')
    print(t.head(3).to_string())
