# Usage

```{eval-rst}
.. container:: cell code

   .. code:: python

      import datetime as dt
      import pandas as pd

.. container:: cell markdown

   .. rubric:: #1 Data downloading
      :name: 1-data-downloading

.. container:: cell code

   .. code:: python

      from remodels.data.EntsoeApi import EntsoeApi

      start_date = dt.date(2015, 1, 1)
      end_date = dt.date(2023, 7, 1)

.. container:: cell code

   .. code:: python

      # to use Entsoe API, you need a free account to obtain a security token
      security_token = "7032e795-c8ae-4a50-aac8-a377b64b1c9e"

      entsoeApi = EntsoeApi(security_token)

.. container:: cell markdown

   .. rubric:: #1.1 sample data - Germany
      :name: 11-sample-data---germany

   Data used by B. Uniejewski in his article "Smoothing Quantile
   Regression Averaging: A new approach to probabilistic forecasting of
   electricity prices"

.. container:: cell code

   .. code:: python

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

   .. container:: output execute_result

      ::

                                    price_da  quantity
         datetime
         2015-01-04 23:00:00+00:00     22.34  50326.50
         2015-01-05 00:00:00+00:00     17.93  48599.50
         2015-01-05 01:00:00+00:00     15.17  47364.00
         2015-01-05 02:00:00+00:00     16.38  47292.25
         2015-01-05 03:00:00+00:00     17.38  48370.25
         ...                             ...       ...
         2023-07-01 17:00:00+00:00     82.36  48027.25
         2023-07-01 18:00:00+00:00     89.60  46469.00
         2023-07-01 19:00:00+00:00     92.79  44480.00
         2023-07-01 20:00:00+00:00     92.97  43819.00
         2023-07-01 21:00:00+00:00     89.86  41717.00

         [74399 rows x 2 columns]

.. container:: cell markdown

   .. rubric:: #1.2 sample data - Spain
      :name: 12-sample-data---spain

   Data used by B. Uniejewski in his article "Smoothing Quantile
   Regression Averaging: A new approach to probabilistic forecasting of
   electricity prices"

.. container:: cell code

   .. code:: python

      prices_spain = entsoeApi.get_day_ahead_pricing(
          start_date, end_date, "10YES-REE------0", resolution_preference=60
      )
      forecast_load_spain = entsoeApi.get_forecast_load(
          start_date, end_date, "10YES-REE------0"
      )
      forecast_load_spain = forecast_load_spain.resample("H").mean()
      spain_data = prices_spain.join(forecast_load_spain)

.. container:: cell markdown

   .. rubric:: #1.3 sample data - France
      :name: 13-sample-data---france

.. container:: cell code

   .. code:: python

      prices_france = entsoeApi.get_day_ahead_pricing(
          start_date, end_date, "10YFR-RTE------C", resolution_preference=60
      )
      forecast_load_france = entsoeApi.get_forecast_load(
          start_date, end_date, "10YFR-RTE------C"
      )
      france_data = prices_france.join(forecast_load_france)

.. container:: cell code

   .. code:: python

      germany_data.to_csv("germany_data.csv")
      spain_data.to_csv("spain_data.csv")
      france_data.to_csv("france_data.csv")

.. container:: cell markdown

   .. rubric:: #2 Point prediction model
      :name: 2-point-prediction-model

.. container:: cell code

   .. code:: python

      data = pd.read_csv("germany_data.csv", index_col=0, parse_dates=True)
      data = data.rename(columns={"quantity": "load"})
      data.head(5)

   .. container:: output execute_result

      ::

                                    price_da      load
         datetime
         2015-01-04 23:00:00+00:00     22.34  50326.50
         2015-01-05 00:00:00+00:00     17.93  48599.50
         2015-01-05 01:00:00+00:00     15.17  47364.00
         2015-01-05 02:00:00+00:00     16.38  47292.25
         2015-01-05 03:00:00+00:00     17.38  48370.25

.. container:: cell markdown

   .. rubric:: #2.1 Adjusting data for Daylight Saving Time changes
      :name: 21-adjusting-data-for-daylight-saving-time-changes

.. container:: cell code

   .. code:: python

      # useless - data is already properly adjusted
      from remodels.transformers.DSTAdjuster import DSTAdjuster

      data = DSTAdjuster().fit_transform(data)

.. container:: cell markdown

   .. rubric:: Example data preparation - lags, additional variables
      :name: example-data-preparation---lags-additional-variables

.. container:: cell code

   .. code:: python

      data = data.assign(
          price_da_1D=lambda x: x["price_da"].shift(24),
          price_da_2D=lambda x: x["price_da"].shift(2 * 24),
          price_da_7D=lambda x: x["price_da"].shift(7 * 24),
          price_da_23_1D=lambda x: x.resample("D")["price_da"].transform("last").shift(24),
          min_price_da_1D=lambda x: x.resample("D")["price_da"].transform("min").shift(24),
          max_price_da_1D=lambda x: x.resample("D")["price_da"].transform("max").shift(24),
      )
      data.tail(5)

   .. container:: output execute_result

      ::

                              price_da      load  price_da_1D  price_da_2D  \
         datetime
         2023-07-01 17:00:00     82.36  48027.25       117.96       166.25
         2023-07-01 18:00:00     89.60  46469.00       133.43       171.58
         2023-07-01 19:00:00     92.79  44480.00       130.74       154.73
         2023-07-01 20:00:00     92.97  43819.00       122.39       141.81
         2023-07-01 21:00:00     89.86  41717.00       109.47       121.11

                              price_da_7D  price_da_23_1D  min_price_da_1D  \
         datetime
         2023-07-01 17:00:00       129.95          108.91            88.46
         2023-07-01 18:00:00       141.44          108.91            88.46
         2023-07-01 19:00:00       150.72          108.91            88.46
         2023-07-01 20:00:00       130.46          108.91            88.46
         2023-07-01 21:00:00       118.31          108.91            88.46

                              max_price_da_1D
         datetime
         2023-07-01 17:00:00           138.97
         2023-07-01 18:00:00           138.97
         2023-07-01 19:00:00           138.97
         2023-07-01 20:00:00           138.97
         2023-07-01 21:00:00           138.97

.. container:: cell markdown

   .. rubric:: #2.2 Variance Stabilizing Transformations
      :name: 22-variance-stabilizing-transformations

.. container:: cell code

   .. code:: python

      from remodels.transformers.VSTransformers import ArcsinhScaler
      from remodels.transformers.VSTransformers import BoxCoxScaler
      from remodels.transformers.VSTransformers import ClippingScaler
      from remodels.transformers.VSTransformers import LogClippingScaler
      from remodels.transformers.VSTransformers import LogisticScaler
      from remodels.transformers.VSTransformers import MLogScaler
      from remodels.transformers.VSTransformers import PolyScaler

.. container:: cell code

   .. code:: python

      # use VSTransformer directly
      arcsinh_scaler = ArcsinhScaler()
      transformed_data = arcsinh_scaler.fit_transform(data)
      transformed_data.tail(5)

   .. container:: output execute_result

      ::

                              price_da       load  price_da_1D  price_da_2D  \
         datetime
         2023-07-01 17:00:00  5.104284  11.472671     5.463511     5.806649
         2023-07-01 18:00:00  5.188534  11.439688     5.586738     5.838205
         2023-07-01 19:00:00  5.223515  11.395942     5.566372     5.734839
         2023-07-01 20:00:00  5.225453  11.380970     5.500377     5.647648
         2023-07-01 21:00:00  5.191431  11.331811     5.388819     5.489863

                              price_da_7D  price_da_23_1D  min_price_da_1D  \
         datetime
         2023-07-01 17:00:00     5.560312         5.38369          5.17573
         2023-07-01 18:00:00     5.645035         5.38369          5.17573
         2023-07-01 19:00:00     5.708582         5.38369          5.17573
         2023-07-01 20:00:00     5.564229         5.38369          5.17573
         2023-07-01 21:00:00     5.466473         5.38369          5.17573

                              max_price_da_1D
         datetime
         2023-07-01 17:00:00         5.627418
         2023-07-01 18:00:00         5.627418
         2023-07-01 19:00:00         5.627418
         2023-07-01 20:00:00         5.627418
         2023-07-01 21:00:00         5.627418

.. container:: cell code

   .. code:: python

      # apply inverse transformation
      arcsinh_scaler.inverse_transform(transformed_data)[0].tail(5)

   .. container:: output execute_result

      ::

                              price_da      load  price_da_1D  price_da_2D  \
         datetime
         2023-07-01 17:00:00     82.36  48027.25       117.96       166.25
         2023-07-01 18:00:00     89.60  46469.00       133.43       171.58
         2023-07-01 19:00:00     92.79  44480.00       130.74       154.73
         2023-07-01 20:00:00     92.97  43819.00       122.39       141.81
         2023-07-01 21:00:00     89.86  41717.00       109.47       121.11

                              price_da_7D  price_da_23_1D  min_price_da_1D  \
         datetime
         2023-07-01 17:00:00       129.95          108.91            88.46
         2023-07-01 18:00:00       141.44          108.91            88.46
         2023-07-01 19:00:00       150.72          108.91            88.46
         2023-07-01 20:00:00       130.46          108.91            88.46
         2023-07-01 21:00:00       118.31          108.91            88.46

                              max_price_da_1D
         datetime
         2023-07-01 17:00:00           138.97
         2023-07-01 18:00:00           138.97
         2023-07-01 19:00:00           138.97
         2023-07-01 20:00:00           138.97
         2023-07-01 21:00:00           138.97

.. container:: cell code

   .. code:: python

      # some VSTransformers may require addtional arguments
      # e.g. PolyScaler
      # lamb: exponent used in the polynomial transformation

      PolyScaler(lamb=0.125).fit_transform(data).tail(5)

   .. container:: output execute_result

      ::

                              price_da      load  price_da_1D  price_da_2D  \
         datetime
         2023-07-01 17:00:00  0.603203  2.707741     0.680950     0.759117
         2023-07-01 18:00:00  0.621062  2.691912     0.708581     0.766497
         2023-07-01 19:00:00  0.628545  2.671017     0.703981     0.742442
         2023-07-01 20:00:00  0.628960  2.663892     0.689165     0.722421
         2023-07-01 21:00:00  0.621680  2.640592     0.664440     0.686817

                              price_da_7D  price_da_23_1D  min_price_da_1D  \
         datetime
         2023-07-01 17:00:00     0.702614        0.663313         0.618333
         2023-07-01 18:00:00     0.721825        0.663313         0.618333
         2023-07-01 19:00:00     0.736387        0.663313         0.618333
         2023-07-01 20:00:00     0.703497        0.663313         0.618333
         2023-07-01 21:00:00     0.681608        0.663313         0.618333

                              max_price_da_1D
         datetime
         2023-07-01 17:00:00         0.717811
         2023-07-01 18:00:00         0.717811
         2023-07-01 19:00:00         0.717811
         2023-07-01 20:00:00         0.717811
         2023-07-01 21:00:00         0.717811

.. container:: cell code

   .. code:: python

      # later we will pass VSTransformer as an argument in out pipeline

.. container:: cell markdown

   .. rubric:: 2.3 Point Model definition & predictions
      :name: 23-point-model-definition--predictions

.. container:: cell code

   .. code:: python

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

.. container:: cell code

   .. code:: python

      # obtaining point predictions

      # set start date and end date
      start = dt.date(2017, 1, 1)
      end = dt.date(2017, 1, 31)

      # fit point model
      pointModel.fit(data, start=start, end=end)

      # get predictions for differend calibration windows
      point_prediction_182 = pointModel.predict(rolling_window=182, inverse_predictions=True)
      point_prediction_364 = pointModel.predict(rolling_window=364, inverse_predictions=True)
      point_prediction_728 = pointModel.predict(rolling_window=728, inverse_predictions=True)

      # print summary
      pointModel.summary()

   .. container:: output execute_result

      ::

                                 MAE         MSE       RMSE       MAPE        R2
         prediction_182rw  10.604293  255.073186  15.971011  38.311686  0.616126
         prediction_364rw  10.575550  242.093443  15.559352  56.936760  0.635660
         prediction_728rw  11.036905  263.647687  16.237231  68.703245  0.603222

.. container:: cell code

   .. code:: python

      point_predictions = pd.concat([
          point_prediction_182,
          point_prediction_364,
          point_prediction_728
      ], axis=1)

      point_predictions

   .. container:: output execute_result

      ::

                              prediction_182rw  prediction_364rw  prediction_728rw
         DateTime
         2017-01-01 00:00:00         17.873043         20.770648         20.528032
         2017-01-01 01:00:00         16.679405         19.760847         19.634005
         2017-01-01 02:00:00         17.734410         18.835915         18.598335
         2017-01-01 03:00:00         17.860881         16.665106         16.783094
         2017-01-01 04:00:00         13.688883         11.613117         10.663689
         ...                               ...               ...               ...
         2017-01-31 19:00:00         48.427391         51.057764         51.552124
         2017-01-31 20:00:00         40.407145         43.323584         43.491338
         2017-01-31 21:00:00         39.663584         42.214230         40.840423
         2017-01-31 22:00:00         32.503318         33.842903         33.933611
         2017-01-31 23:00:00         33.013635         35.328445         34.316088

         [744 rows x 3 columns]

.. container:: cell markdown

   .. rubric:: #3 QRA model
      :name: 3-qra-model

.. container:: cell code

   .. code:: python

      # for now, QRA models require numpy arrays
      X = point_predictions.to_numpy()
      y = point_predictions.join(data)["price_da"].to_numpy()

.. container:: cell code

   .. code:: python

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

.. container:: cell markdown

   .. rubric:: #3.1 QRA models - direct usage
      :name: 31-qra-models---direct-usage

.. container:: cell code

   .. code:: python

      # sample prediction
      # three different quantiles
      # output is returned as np.array
      y_pred_q25 = QRA(quantile=0.25, fit_intercept=True).fit(X, y).predict(X)
      y_pred_q50 = QRA(quantile=0.50, fit_intercept=True).fit(X, y).predict(X)
      y_pred_q75 = QRA(quantile=0.75, fit_intercept=True).fit(X, y).predict(X)

      y_pred_q50[:10]

   .. container:: output execute_result

      ::

         array([19.71594299, 18.51816111, 18.40119957, 16.80886388, 12.58936152,
                 2.92735718, 18.6152022 , 23.41801672, 23.46669472, 23.17472373])

.. container:: cell code

   .. code:: python

      pd.DataFrame(
          dict(q25=y_pred_q25, q50=y_pred_q50, q75=y_pred_q75, y_true=y),
          index=point_predictions.index,
      ).tail(10)

   .. container:: output execute_result

      ::

                                    q25        q50        q75  y_true
         DateTime
         2017-01-31 14:00:00  54.186739  66.394722  80.972197   89.75
         2017-01-31 15:00:00  55.208994  66.808502  83.680725   90.20
         2017-01-31 16:00:00  61.288296  75.258385  95.000000   95.00
         2017-01-31 17:00:00  58.128670  70.469383  88.409293  104.33
         2017-01-31 18:00:00  51.582541  61.424381  76.406883   90.00
         2017-01-31 19:00:00  43.948801  50.856552  64.150751   84.15
         2017-01-31 20:00:00  37.979050  42.909250  54.060156   60.92
         2017-01-31 21:00:00  37.354382  42.899396  52.528673   54.90
         2017-01-31 22:00:00  31.108137  33.710300  40.956407   40.69
         2017-01-31 23:00:00  32.103146  35.591743  43.332177   44.91

.. container:: cell code

   .. code:: python

      # some QRA models may require addtional parameters
      # e.g. LQRA
      # lambda_: LASSO regularization parameter
      LQRA(quantile=0.50, lambda_=1, fit_intercept=True).fit(X, y).predict(X)[:10]

   .. container:: output execute_result

      ::

         array([20.82443897, 19.74028314, 19.55087324, 18.22740215, 13.4766444 ,
                 4.2748866 , 18.34540101, 22.75212594, 23.07812124, 23.05093359])

.. container:: cell markdown

   .. rubric:: #3.2 QRA Tester
      :name: 32-qra-tester

.. container:: cell code

   .. code:: python

      from remodels.qra.tester import QR_Tester

.. container:: cell code

   .. code:: python

      # QR_Tester fits model on initial `calibration_window` period
      # predicts next `prediction_window` values (every quantile)
      # moves window and repeats

      qra_model = QRA(fit_intercept=True)

      results = QR_Tester(
          calibration_window=72,
          prediction_window=24,
          qr_model=qra_model,     # any QR model
          max_workers=4,          # multiprocessing max workers
      ).fit_predict(X, y)

.. container:: cell code

   .. code:: python

      print("X.shape:", X.shape)
      print("y.shape:", y.shape)
      print("Y_pred.shape:", results.Y_pred.shape)  # without initial `calibration_window` period

   .. container:: output stream stdout

      ::

         X.shape: (744, 3)
         y.shape: (744,)
         Y_pred.shape: (672, 99)

.. container:: cell code

   .. code:: python

      # all 99 percentiles
      pd.DataFrame(results.Y_pred).tail(3)

   .. container:: output execute_result

      ::

                     0          1   ...         97         98
         669  34.551064  36.642725  ...  57.053207  57.053207
         670  31.233392  31.960833  ...  34.409566  34.410440
         671  31.059022  32.656342  ...  38.622451  38.622451

         [3 rows x 99 columns]

.. container:: cell markdown

   .. rubric:: # 3.3 Probabilistic predictions metrics
      :name: -33-probabilistic-predictions-metrics

.. container:: cell code

   .. code:: python

      # from 3.2
      results

   .. container:: output execute_result

      ::

         <remodels.qra.tester.qr_tester._Results at 0x2265c7dba30>

.. container:: cell code

   .. code:: python

      # average empirical coverage
      # alpha: length of prediction interval
      # e.g. alpha=50 --> prediction interval from 25 to 75 percentile

      # desired: 50, obtained: 34 --> prediction intervals are too narrow
      results.aec(alpha=50)

   .. container:: output execute_result

      ::

         0.34970238095238093

.. container:: cell code

   .. code:: python

      # average empirical coverage per hour
      results.ec_h(alpha=50)

   .. container:: output execute_result

      ::

         array([0.5       , 0.39285714, 0.5       , 0.42857143, 0.39285714,
                0.17857143, 0.25      , 0.39285714, 0.39285714, 0.25      ,
                0.25      , 0.21428571, 0.28571429, 0.35714286, 0.39285714,
                0.39285714, 0.39285714, 0.39285714, 0.28571429, 0.25      ,
                0.35714286, 0.39285714, 0.39285714, 0.35714286])

.. container:: cell code

   .. code:: python

      # mean absolute deviation of empirical coverage per hour
      results.ec_mad(alpha=50)

   .. container:: output execute_result

      ::

         0.15029761904761904

.. container:: cell code

   .. code:: python

      # Kupiec test for condidional coverage

      # returns: number of hours that test is not rejected
      # the higher, the better
      results.kupiec_test(alpha=50, significance_level=0.05)

   .. container:: output execute_result

      ::

         16

.. container:: cell code

   .. code:: python

      # Christoffersen test

      # returns: number of hours that test is not rejected
      # the higher, the better
      results.christoffersen_test(alpha=50, significance_level=0.05)

   .. container:: output execute_result

      ::

         15

.. container:: cell code

   .. code:: python

      # aggregate pinball score.
      results.aps()

   .. container:: output execute_result

      ::

         5.600464806241713
```
