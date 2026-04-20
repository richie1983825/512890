from backtest_core.data import load_and_forward_adjust, resolve_data_path
from backtest_core.workflows import build_rolling_splits_3y1y
base_data = load_and_forward_adjust(resolve_data_path())
windows = build_rolling_splits_3y1y(base_data)
w = windows[0]
print(f"Window type: {type(w)}")
print(f"Window length: {len(w)}")
for i, x in enumerate(w):
    print(f"Element {i} type: {type(x)}")
