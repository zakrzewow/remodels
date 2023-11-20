"""PointModel."""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class PointModel:
    """PointModel."""

    def __init__(self, pipeline, variables_per_hour, y_column="price_da"):
        """Initialize the PointModel with a data processing pipeline, variables mapped to each hour, and the target column name.

        :param pipeline: Sequence of data transformation steps and a predictive model.
        :type pipeline: list
        :param variables_per_hour: Mapping from hour ranges to the variables to be used in those hours.
        :type variables_per_hour: dict
        :param y_column: The name of the target column.
        :type y_column: str
        """
        self.variables_per_hour = variables_per_hour
        self.transformation_pipeline = pipeline[:-1]
        self.all_used_columns = list(
            np.unique(
                [
                    item
                    for sublist in self.variables_per_hour.values()
                    for item in sublist
                ]
            )
        )
        self.model = pipeline[-1]
        self.unique_hours = None
        self.y_column = [y_column]
        self.predictions = None
        self.training_data = None

    def fit(self, df, start, end):
        """Fit the model with the training data.

        :param df: DataFrame containing the training data.
        :type df: pandas.DataFrame
        """
        self.training_data = df
        self.set_unique_hours(df.index)
        self.start = start
        self.end = end

    def set_unique_hours(self, dates):
        """Set the unique hours for the model based on the provided datetime data.

        :param dates: Datetime data to extract unique hours from.
        :type dates: pandas.Series or similar
        """
        if self.unique_hours is None:
            self.unique_hours = np.sort(dates.hour.unique())

    def get_hour_variables(self, hour):
        """Retrieve the variables associated with a specific hour based on defined hour ranges.

        :param hour: The hour for which variables are needed.
        :type hour: int
        :return: List of variables associated with the given hour.
        :rtype: list
        """
        for hour_range, variables in self.variables_per_hour.items():
            if hour_range[0] <= hour <= hour_range[1]:
                return variables

    def separate_columns_by_dtype(self, df):
        """Separate columns in a DataFrame by data type (float vs non-float).

        :param df: DataFrame to separate columns from.
        :type df: pandas.DataFrame
        :return: Lists of float columns and non-float columns.
        :rtype: tuple
        """
        float_columns = df.select_dtypes(include=[np.float64]).columns.tolist()
        non_float_columns = [col for col in df.columns if col not in float_columns]
        return float_columns, non_float_columns

    def fit_transform_data(self, Xy, is_train=True):
        """Fit the transformation pipeline to the data and transform it if is_train is True, otherwise, only transform the data.

        :param Xy: DataFrame containing features and target to be transformed.
        :type Xy: pandas.DataFrame
        :param is_train: Flag to indicate whether to fit the transformer or not.
        :type is_train: bool
        :return: Transformed features and optionally transformed target.
        :rtype: tuple or pandas.DataFrame
        """
        # Separate columns by data type
        float_columns, non_float_columns = self.separate_columns_by_dtype(
            Xy[self.all_used_columns]
        )
        non_float_data = Xy[non_float_columns]

        if is_train:
            # Fit and transform the data
            X, y = self.transformation_pipeline.fit_transform(
                Xy[float_columns], Xy[self.y_column]
            )
            X = pd.concat([X, non_float_data], axis=1)
            return X, y
        else:
            # Only transform the data
            X_transformed = self.transformation_pipeline.transform(Xy[float_columns])
            X = pd.concat([X_transformed, non_float_data], axis=1)
            return X

    def train_and_predict_hours(
        self, day, Xy_train, Xy_test, predictions_list, inverse_predictions
    ):
        """Train the model and make predictions for each hour in the unique_hours, and store the predictions in a list.

        :param day: The day for which predictions are made.
        :type day: pandas.Timestamp
        :param Xy_train: Training data.
        :type Xy_train: pandas.DataFrame
        :param Xy_test: Testing data.
        :type Xy_test: pandas.DataFrame
        :param predictions_list: List to store the predictions.
        :type predictions_list: list
        :param inverse_predictions: Flag to determine whether to apply inverse transformation to predictions.
        :type inverse_predictions: bool
        """
        for hour in self.unique_hours:
            hour_variables = self.get_hour_variables(hour)
            Xy_train_hour = Xy_train.loc[Xy_train.index.hour == hour]
            X_test_hour = Xy_test.loc[Xy_test.index.hour == hour][hour_variables]
            Xy_train_hour, y_train_hour = self.fit_transform_data(
                Xy_train_hour[hour_variables + self.y_column]
            )
            X_test_hour = self.fit_transform_data(
                X_test_hour[hour_variables], is_train=False
            )
            date_ = pd.Timestamp(f"{day} {hour}:00:00")

            if len(Xy_train_hour) > 0 and len(X_test_hour) > 0:
                self.model.fit(Xy_train_hour, y_train_hour)
                prediction = self.model.predict(X_test_hour)

                if (len(prediction) > 0) and inverse_predictions:
                    _, prediction_df = self.transformation_pipeline.inverse_transform(
                        yt=prediction[0]
                    )
                    predictions_list.append((date_, prediction_df[0]))
                elif len(prediction) > 0:
                    predictions_list.append((date_, prediction[0, 0]))

    def predict(self, rolling_window=728, inverse_predictions=True):
        """Predict values over a given range, from start to end, using a rolling window, and store/update predictions in the model.

        :param df: DataFrame containing the data to be used for prediction.
        :type df: pandas.DataFrame
        :param rolling_window: Number of days to look back for training data.
        :type rolling_window: int
        :param inverse_predictions: Flag to determine whether to apply inverse transformation to predictions.
        :type inverse_predictions: bool
        :return: DataFrame of predicted values.
        :rtype: pandas.DataFrame
        """
        df = self.training_data
        predictions_list = []
        for day in pd.date_range(self.start, self.end, freq="D"):
            Xy_train = df.loc[
                (df.index.date >= day.date() - dt.timedelta(days=rolling_window))
                & (df.index.date < day.date())
            ][self.all_used_columns + self.y_column].dropna()
            Xy_test = df.loc[df.index.date == day.date()][
                self.all_used_columns + self.y_column
            ].dropna()
            self.train_and_predict_hours(
                day, Xy_train, Xy_test, predictions_list, inverse_predictions
            )

        # Prepare DataFrame from predictions_list
        new_predictions_df = pd.DataFrame(
            predictions_list, columns=["DateTime", f"prediction_{rolling_window}rw"]
        )
        new_predictions_df.set_index("DateTime", inplace=True)

        # Update or create the self.predictions DataFrame
        if (
            self.predictions is not None
            and new_predictions_df.columns[0] not in self.predictions.columns
        ):
            # Merge with existing predictions on DateTime index
            self.predictions = self.predictions.join(new_predictions_df, how="outer")
        else:
            self.predictions = new_predictions_df

        return self.predictions[[f"prediction_{rolling_window}rw"]]

    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

    def summary(self):
        """Generate a summary comparing stored predictions with actual values from the training data.

        :return: DataFrame with summary metrics.
        :rtype: pandas.DataFrame
        """
        if self.predictions is None:
            raise ValueError("No predictions have been made yet.")
        if self.training_data is None:
            raise ValueError("Model has not been fitted with training data yet.")

        # Align the predictions with the actual values based on the datetime index
        aligned_df = self.training_data[self.y_column].join(
            self.predictions, how="inner"
        )

        # Extract the actual and predicted values
        actual_values = aligned_df[self.y_column[0]]
        metrics = {}
        for col in self.predictions.columns:
            if col.startswith("prediction_"):
                # Calculate metrics for each set of predictions
                metrics[col] = self.calculate_metrics(actual_values, aligned_df[col])

        # Convert metrics to DataFrame for easier analysis
        metrics_df = pd.DataFrame(metrics).T

        return metrics_df
