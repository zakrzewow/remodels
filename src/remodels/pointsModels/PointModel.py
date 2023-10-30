import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt

class PointModel:
    def __init__(self, pipeline, variables_per_hour, y_column="price_da"):
        self.variables_per_hour = variables_per_hour
        self.transformation_pipeline = pipeline[:-1]
        self.all_used_columns = list(np.unique([item for sublist in self.variables_per_hour.values() for item in sublist]))
        self.model = pipeline[-1]
        self.unique_hours = None
        self.y_column = ["price_da"]

    def set_unique_hours(self, dates):
        if self.unique_hours is None:
            self.unique_hours = dates.hour.unique()

    def get_hour_variables(self, hour):
        for hour_range, variables in self.variables_per_hour.items():
            if hour_range[0] <= hour <= hour_range[1]:
                return variables

    def separate_columns_by_dtype(self, df):
        float_columns = df.select_dtypes(include=[np.float64]).columns.tolist()
        non_float_columns = [col for col in df.columns if col not in float_columns]
        return float_columns, non_float_columns

    def fit_transform_data(self, Xy, is_train=True):
        float_columns, non_float_columns = self.separate_columns_by_dtype(Xy[self.all_used_columns])
        non_float_data = Xy[non_float_columns]

        if is_train:
            X, y = self.transformation_pipeline.fit_transform(Xy[float_columns], Xy[self.y_column])
            X = pd.concat([X, non_float_data], axis=1)
            return X, y
        else:
            X_transformed, _ = self.transformation_pipeline.transform(Xy[float_columns])
            X = pd.concat([X_transformed, non_float_data], axis=1)
            return X

    def train_and_predict_hours(self, day, X_train, y_train, X_test, predictions_series):
        for hour in self.unique_hours:
            hour_variables = self.get_hour_variables(hour)

            X_train_hour = X_train.loc[X_train.index.hour == hour][hour_variables]
            y_train_hour = y_train.loc[y_train.index.hour == hour]
            X_test_hour = X_test.loc[X_test.index.hour == hour][hour_variables]
            date_ = pd.Timestamp(f"{day} {hour}:00:00")

            if len(X_train_hour) > 0 and len(X_test_hour) > 0:
                self.model.fit(X_train_hour, y_train_hour)
                prediction = self.model.predict(X_test_hour)

                if len(prediction) > 0:
                    _, prediction_df = self.transformation_pipeline.inverse_transform(yt=pd.DataFrame(prediction[0], columns=self.y_column))
                    predictions_series[date_] = prediction_df.iloc[0, 0]

    def predict(self, df, start, end, rolling_window=728):
        self.set_unique_hours(df.index)
        
        predictions_series = pd.Series(dtype='float64')
        for day in tqdm(pd.date_range(start, end, freq="D")):
            Xy_train = df.loc[(df.index.date >= day.date() - dt.timedelta(days=rolling_window)) & (df.index.date < day.date())][self.all_used_columns + self.y_column].dropna()
            Xy_test = df.loc[df.index.date == day.date()][self.all_used_columns + self.y_column].dropna()

            X_train, y_train = self.fit_transform_data(Xy_train)
            X_test = self.fit_transform_data(Xy_test, is_train=False)

            self.train_and_predict_hours(day, X_train, y_train, X_test, predictions_series)
        return predictions_series