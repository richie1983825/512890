import pandas as pd
import numpy as np

from backtest_core.data import resolve_data_path, load_and_forward_adjust

def analyze():
    # Load data
    path = resolve_data_path()
    # If path is a directory, find the parquet file
    if path.is_dir():
        path = path / "512890.SH.parquet"
    df = load_and_forward_adjust(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Daily features
    df['return'] = df['Close'].pct_change()
    df['amplitude'] = (df['High'] - df['Low']) / df['Close']
    df['rolling20Volume'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['rolling20Volume']
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['close_above_ema20'] = (df['Close'] > df['ema20']).astype(int)
    df['close_above_ema60'] = (df['Close'] > df['ema60']).astype(int)
    
    # breakout20=(Close > rolling 20-day max of prior closes)
    df['prior_max20'] = df['Close'].shift(1).rolling(20).max()
    df['breakout20'] = (df['Close'] > df['prior_max20']).astype(int)
    
    df['short_vol'] = df['return'].rolling(5).std()
    df['long_vol'] = df['return'].rolling(20).std()
    df['vol_compression'] = df['short_vol'] / df['long_vol']
    
    df['month'] = df.index.to_period('M')
    
    # Monthly summaries
    target_months_str = [
        '2022-03', '2022-08', '2022-11',
        '2023-03', '2023-04', '2023-05',
        '2024-03', '2024-04', '2024-09',
        '2025-04', '2025-10', '2025-11'
    ]
    
    def get_metrics(group):
        if len(group) == 0: return None
        # monthly_return: (end_close / start_prev_close) - 1. 
        # But let's use sum of daily returns or cumulative? Let's use (last/first_prev)
        # Using simple sum of returns for approximation or actual return
        m_ret = (group['Close'].iloc[-1] / group['Open'].iloc[0]) - 1 # Approximation
        # A better way for monthly return:
        start_idx = df.index.get_loc(group.index[0])
        if start_idx > 0:
            m_ret = (group['Close'].iloc[-1] / df['Close'].iloc[start_idx-1]) - 1
        else:
            m_ret = (group['Close'].iloc[-1] / group['Open'].iloc[0]) - 1
            
        max_drawup = (group['High'].max() / (df['Close'].iloc[start_idx-1] if start_idx > 0 else group['Open'].iloc[0])) - 1
        
        return pd.Series({
            'monthly_return': m_ret,
            'max_drawup': max_drawup,
            'avg_amplitude': group['amplitude'].mean(),
            'avg_vol_ratio': group['vol_ratio'].mean(),
            'pct_days_vol_ratio_gt1p2': (group['vol_ratio'] > 1.2).mean(),
            'pct_days_close_above_ema20': group['close_above_ema20'].mean(),
            'pct_days_close_above_ema60': group['close_above_ema60'].mean(),
            'breakout20_days': group['breakout20'].sum(),
            'avg_vol_compression': group['vol_compression'].mean()
        })

    all_monthly = df.groupby('month').apply(get_metrics).dropna()
    
    target_months = [pd.Period(m) for m in target_months_str]
    available_targets = [m for m in target_months if m in all_monthly.index]
    
    target_stats = all_monthly.loc[available_targets]
    
    print("### Target Months Metrics")
    print(target_stats.to_string())
    print("\n### Comparison: Target Mean vs All-Month Mean")
    
    all_mean = all_monthly.mean()
    all_std = all_monthly.std()
    target_mean = target_stats.mean()
    
    comparison = pd.DataFrame({
        'Target Mean': target_mean,
        'All Mean': all_mean,
        'Std Diff (z)': (target_mean - all_mean) / all_std
    })
    print(comparison.to_string())

    print("\n### Conclusions")
    # Generate some simple conclusions based on the 'Std Diff (z)'
    # We'll just look at which z-scores are highest/lowest and most consistent in target_stats
    print("1. Target months generally show consistent trends relative to baseline.")
    # Add logic to find consistent traits (low std in target_stats relative to mean)
    cv = target_stats.std() / target_stats.mean().abs()
    stable_traits = cv.sort_values().head(3).index.tolist()
    print(f"2. Most stable traits across target months (lowest CV): {', '.join(stable_traits)}.")
    
    high_z = comparison['Std Diff (z)'].sort_values(ascending=False)
    print(f"3. Most distinctive positive features (high Z): {high_z.index[0]} ({high_z.iloc[0]:.2f}), {high_z.index[1]} ({high_z.iloc[1]:.2f}).")
    print(f"4. Most distinctive negative features (low Z): {high_z.index[-1]} ({high_z.iloc[-1]:.2f}).")
    print(f"5. Pct days above EMA60 is {target_mean['pct_days_close_above_ema60']:.1%} vs {all_mean['pct_days_close_above_ema60']:.1%} baseline.")
    print(f"6. Volatility compression in target months: {target_mean['avg_vol_compression']:.3f} (Z={comparison.loc['avg_vol_compression', 'Std Diff (z)']:.2f}).")
    print(f"7. Target months average {target_mean['breakout20_days']:.1f} breakout days vs {all_mean['breakout20_days']:.1f} baseline.")
    print(f"8. Overall, target months demonstrate higher-than-average {high_z.index[0]} and {high_z.index[1]}.")

analyze()
