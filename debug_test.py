import sys; sys.path.insert(0, '.')
from data.upi_data import load_upi_data
from models.forecasters import SeasonalNaive, ETSModel, HybridForecaster
from models.conformal import ConformalForecaster

df = load_upi_data()
series = df['transactions_mn']
train = series.iloc[:-12]
test  = series.iloc[-12:]
print('train end:', train.index[-1].date(), '  test start:', test.index[0].date())
print('pred will start from:', series.index[-12].date())

for Cls in [SeasonalNaive, ETSModel, HybridForecaster]:
    m = Cls()
    cf = ConformalForecaster(m, alpha=0.10, calib_frac=0.25)
    cf.fit(train, max_horizon=12)
    pred = cf.predict_with_intervals(12)
    print('pred index start:', pred.index[0].date())
    lo = pred['lower'].values
    hi = pred['upper'].values
    y = test.values
    inside = (y >= lo) & (y <= hi)
    print(f'{m.name}: coverage={inside.mean()*100:.1f}%')
    print('  sample interval:', round(lo[0],0), '-', round(hi[0],0), '  actual:', round(y[0],0))
