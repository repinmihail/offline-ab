import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.transforms import TimeSeriesImputerTransform
from datetime import datetime


class FillTheGaps:
    """Класс для заполнения пропусков в временных рядах"""

    def __call__(self, df: pd.DataFrame, freq: str = "D", **kwargs) -> pd.DataFrame:
        """
        Переопределенная call-функция

        Args:
            df (pd.DataFrame): датафрейм, приведенный к виду etna_df
                поля: timestamp, segment, target
            freq (str): частота данных (по умолчанию: D)
            **kwargs: дополнительные параметры, которые принимает TimeSeriesImputerTransform

        Returns:
            df (pd.DataFrame): датафрейм с заполненными пропусками
        """

        def testing_df_cols_for_etna(df: pd.DataFrame) -> TSDataset:
            assert "timestamp" in df.columns, "timestamp column must be in TSDataset"
            assert "segment" in df.columns, "segment column must be in TSDataset"
            assert "target" in df.columns, "target column must be in TSDataset"
            assert len(df.columns) == 3
            return True

        def check_weekday(date: str) -> int:
            given_date = datetime.strptime(date, "%Y-%m-%d")
            day_of_week = given_date.weekday() + 1
            return day_of_week

        if testing_df_cols_for_etna(df):
            etna_ts = TSDataset(df, freq=freq)
            transform = [TimeSeriesImputerTransform(in_column="target", **kwargs)]
            etna_ts.fit_transform(transform)
            result_df = TSDataset.to_flatten(etna_ts[:, :, "target"])
            if result_df[result_df["target"].isna()].shape[0] > 0:
                etna_ts_ = TSDataset(result_df, freq=freq)
                transform_ = [
                    TimeSeriesImputerTransform(
                        in_column="target",
                        strategy="running_mean",
                    )
                ]
                etna_ts_.fit_transform(transform_)
                result_df = TSDataset.to_flatten(etna_ts_[:, :, "target"])
                if result_df[result_df["target"].isna()].shape[0] > 0:
                    result_df["timestamp"] = result_df["timestamp"].astype(str)
                    result_df["weekday"] = result_df["timestamp"].apply(lambda x: check_weekday(x))
                    segments_with_na = result_df[result_df["target"].isna()]["segment"].tolist()
                    for segment in segments_with_na:
                        segment_df = result_df[result_df["segment"] == segment]
                        week_days_with_na = segment_df[segment_df["target"].isna()]["weekday"]
                        for day in week_days_with_na:
                            mean = segment_df[segment_df["weekday"] == day]["target"].mean()
                            dt = segment_df[(segment_df["weekday"] == day) & (segment_df["target"].isna())][
                                "timestamp"
                            ].values[0]
                            result_df.loc[
                                (result_df["timestamp"] == dt) & (result_df["segment"] == segment), "target"
                            ] = mean
        return result_df[["timestamp", "segment", "target"]]
