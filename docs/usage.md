# Usage

```python
import datetime as dt
import pandas as pd

pd.options.display.max_columns = 5
```

## #1 Data downloading

```python
from remodels.data.EntsoeApi import EntsoeApi

start_date = dt.date(2015, 1, 1)
end_date = dt.date(2023, 7, 1)
```

```python
# to use Entsoe API, you need a free account to obtain a security token (https://transparency.entsoe.eu/)
security_token = "your-token-here"

entsoeApi = EntsoeApi(security_token)
```

### #1.1 sample data - Germany

Data used by B. Uniejewski in his article "Smoothing Quantile Regression Averaging: A new approach to probabilistic forecasting of electricity prices"

```python
# downloading prices
# we need to download two time series for the desired period
prices_1 = entsoeApi.get_day_ahead_pricing(
    start_date,
    end_date,
    in_domain="10Y1001A1001A63L",
    resolution_preference=60,
)
prices_2 = entsoeApi.get_day_ahead_pricing(
    start_date,
    end_date,
    in_domain="10Y1001A1001A82H",
    resolution_preference=60,
)
prices = pd.concat([prices_1, prices_2])

# downloading load forecast
# load forecast is an additional variable that helps predict future prices
forecast_load_1 = entsoeApi.get_forecast_load(start_date, end_date, "10Y1001A1001A63L")
forecast_load_2 = entsoeApi.get_forecast_load(start_date, end_date, "10Y1001A1001A82H")
forecast_load_1 = forecast_load_1.resample("H").mean()
forecast_load_2 = forecast_load_2.resample("H").mean()
forecast_load = pd.concat([forecast_load_1, forecast_load_2])

#
germany_data = prices.join(forecast_load)
germany_data
```

### #1.2 sample data - Spain

Data used by B. Uniejewski in his article "Smoothing Quantile Regression Averaging: A new approach to probabilistic forecasting of electricity prices"

```python
prices_spain = entsoeApi.get_day_ahead_pricing(
    start_date, end_date, "10YES-REE------0", resolution_preference=60
)
forecast_load_spain = entsoeApi.get_forecast_load(
    start_date, end_date, "10YES-REE------0"
)
forecast_load_spain = forecast_load_spain.resample("H").mean()
spain_data = prices_spain.join(forecast_load_spain)
```

### #1.3 sample data - France

```python
prices_france = entsoeApi.get_day_ahead_pricing(
    start_date, end_date, "10YFR-RTE------C", resolution_preference=60
)
forecast_load_france = entsoeApi.get_forecast_load(
    start_date, end_date, "10YFR-RTE------C"
)
france_data = prices_france.join(forecast_load_france)
```

```python
germany_data.to_csv("germany_data.csv")
spain_data.to_csv("spain_data.csv")
france_data.to_csv("france_data.csv")
```

## #2 Point prediction model

```python
data = pd.read_csv("germany_data.csv", index_col=0, parse_dates=True)
data = data.rename(columns={"quantity": "load"})
data.head(5)
```

### #2.1 Adjusting data for Daylight Saving Time changes

```python
# useless - data is already properly adjusted
from remodels.transformers.DSTAdjuster import DSTAdjuster

data = DSTAdjuster().fit_transform(data)
```

#### Example data preparation - lags, additional variables

```python
data = data.assign(
    price_da_1D=lambda x: x["price_da"].shift(24),
    price_da_2D=lambda x: x["price_da"].shift(2 * 24),
    price_da_7D=lambda x: x["price_da"].shift(7 * 24),
    price_da_23_1D=lambda x: x.resample("D")["price_da"].transform("last").shift(24),
    min_price_da_1D=lambda x: x.resample("D")["price_da"].transform("min").shift(24),
    max_price_da_1D=lambda x: x.resample("D")["price_da"].transform("max").shift(24),
)
data.tail(5)
```

### #2.2 Variance Stabilizing Transformations

```python
from remodels.transformers.VSTransformers import ArcsinhScaler
from remodels.transformers.VSTransformers import BoxCoxScaler
from remodels.transformers.VSTransformers import ClippingScaler
from remodels.transformers.VSTransformers import LogClippingScaler
from remodels.transformers.VSTransformers import LogisticScaler
from remodels.transformers.VSTransformers import MLogScaler
from remodels.transformers.VSTransformers import PolyScaler
```

```python
# use VSTransformer directly
arcsinh_scaler = ArcsinhScaler()
transformed_data = arcsinh_scaler.fit_transform(data)
transformed_data.tail(5)

```

```python
# apply inverse transformation
arcsinh_scaler.inverse_transform(transformed_data)[0].tail(5)
```

```python
# some VSTransformers may require addtional arguments
# e.g. PolyScaler
# lamb: exponent used in the polynomial transformation

PolyScaler(lamb=0.125).fit_transform(data).tail(5)
```

```python
# later we will pass VSTransformer as an argument in out pipeline
```

### 2.3 Point Model definition & predictions

```python
from remodels.transformers import StandardizingScaler
from remodels.pointsModels import PointModel
from remodels.pipelines import RePipeline

# you can use any model represented by the class with .fit(X, y) and .predict(X) methods
from sklearn.linear_model import LinearRegression

# pipeline - to specify sequence of steps
pipeline = RePipeline(
    [
        ("standardScaler", StandardizingScaler()),
        ("vstScaler", PolyScaler()),
        ("LinearRegression", LinearRegression()),
    ]
)

# for point model, you have to specify mapping from hour ranges to the variables to be used in those hours
# our case is very simple - all variables for each hour
X_cols_to_pipeline = [
    "price_da_1D",
    "price_da_2D",
    "price_da_7D",
    "price_da_23_1D",
    "max_price_da_1D",
    "min_price_da_1D",
    "load",
]
y_col = "price_da"

variables_per_hour = {(0, 25): X_cols_to_pipeline}

pointModel = PointModel(
    pipeline=pipeline,
    variables_per_hour=variables_per_hour,
    y_column="price_da",
)
```

```python
# obtaining point predictions

# set start date and end date
start = dt.date(2017, 1, 1)
end = dt.date(2017, 1, 31)

# fit point model
pointModel.fit(data, start=start, end=end)

# get predictions for differend calibration windows
point_prediction_182 = pointModel.predict(calibration_window=182, inverse_predictions=True)
point_prediction_364 = pointModel.predict(calibration_window=364, inverse_predictions=True)
point_prediction_728 = pointModel.predict(calibration_window=728, inverse_predictions=True)

# print summary
pointModel.summary()
```

```python
point_predictions = pd.concat([
    point_prediction_182,
    point_prediction_364,
    point_prediction_728
], axis=1)

point_predictions
```

## #3 QRA model

```python
# for now, QRA models require numpy arrays
X = point_predictions.to_numpy()
y = point_predictions.join(data)["price_da"].to_numpy()
```

```python
# all QRA variants
from remodels.qra import QRA
from remodels.qra import QRM
from remodels.qra import LQRA
from remodels.qra import FQRA
from remodels.qra import FQRM
from remodels.qra import sFQRA
from remodels.qra import sFQRM
from remodels.qra import SQRA
from remodels.qra import SQRM
```

### #3.1 QRA models - direct usage

```python
# sample prediction
# three different quantiles
# output is returned as np.array
y_pred_q25 = QRA(quantile=0.25, fit_intercept=True).fit(X, y).predict(X)
y_pred_q50 = QRA(quantile=0.50, fit_intercept=True).fit(X, y).predict(X)
y_pred_q75 = QRA(quantile=0.75, fit_intercept=True).fit(X, y).predict(X)

y_pred_q50[:10]
```

```python
pd.DataFrame(
    dict(q25=y_pred_q25, q50=y_pred_q50, q75=y_pred_q75, y_true=y),
    index=point_predictions.index,
).tail(10)
```

```python
# some QRA models may require addtional parameters
# e.g. LQRA
# lambda_: LASSO regularization parameter
LQRA(quantile=0.50, lambda_=1, fit_intercept=True).fit(X, y).predict(X)[:10]
```

### #3.2 QRA Tester

```python
from remodels.qra.tester import QR_Tester
```

```python
# QR_Tester fits model on initial `calibration_window` period
# predicts next `prediction_window` values (every quantile)
# moves window and repeats

qra_model = LQRA(quantile=0.50, lambda_=1, fit_intercept=True)

results = QR_Tester(
    calibration_window=72,
    prediction_window=24,
    qr_model=qra_model,     # any QR model
    max_workers=4,          # multiprocessing max workers
).fit_predict(X, y)
```

```python
print("X.shape:", X.shape)
print("y.shape:", y.shape)
print("Y_pred.shape:", results.Y_pred.shape)  # without initial `calibration_window` period
```

```python
# all 99 percentiles
pd.DataFrame(results.Y_pred).tail(3)
```

### # 3.3 Probabilistic predictions metrics

```python
# from 3.2
results
```

```python
# average empirical coverage
# alpha: length of prediction interval
# e.g. alpha=50 --> prediction interval from 25 to 75 percentile

# desired: 50, obtained: 30 --> prediction intervals are too narrow
results.aec(alpha=50)
```

```python
# average empirical coverage per hour
results.ec_h(alpha=50)
```

```python
# mean absolute deviation of empirical coverage per hour
results.ec_mad(alpha=50)
```

```python
# Kupiec test for condidional coverage

# returns: number of hours that test is not rejected
# the higher, the better
results.kupiec_test(alpha=50, significance_level=0.05)
```

```python
# Christoffersen test

# returns: number of hours that test is not rejected
# the higher, the better
results.christoffersen_test(alpha=50, significance_level=0.05)
```

```python
# aggregate pinball score.
results.aps()
```
