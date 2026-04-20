import pandas as pd
from pathlib import Path
from backtest_core.data import load_and_forward_adjust
from backtest_core.backtests import run_strategy_backtest
from backtest_core.parameters import build_param_space
from strategies.polyfit_deviation_ma_switch_strategy import PolyfitDeviationMASwitchStrategy

def test():
    data_path = Path("data/512890.SH.parquet")
    df = load_and_forward_adjust(data_path)
    df = df.reset_index().rename(columns={'trade_date': 'date'})
    df = df.set_index('date', drop=False)
    val_df = df[df['date'].dt.year == 2022]
    
    # Base polyfit params (manual or scanned)
    params = {
        'polyfit_window': 20,
        'polyfit_degree': 2,
        'buy_threshold': 0.0,
        'sell_threshold': 0.0,
        'flat_wait_days': 10,
        'switch_deviation_m1': 0.04,
        'switch_deviation_m2': 0.02,
        'switch_fast_ma_window': 5,
        'switch_slow_ma_window': 20
    }
    
    # We should use PolyfitDeviationMASwitchStrategy as the strategy class in the backtest
    # Check if run_strategy_backtest allows passing a strategy class. 
    # Based on earlier grep, it might not directly take it if not specified? 
    # Let's check backtests.py again carefully.
    res = run_strategy_backtest(val_df, params)
    stats = res[0]
    print("Return [%]:", stats['Return [%]'])
    print("Number of Trades:", stats['# Trades'])

if __name__ == "__main__":
    test()
