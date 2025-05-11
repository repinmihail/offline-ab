import pandas as pd
import numpy as np
import functools
import operator

from offline_ab.abcore import ABCore
from typing import Tuple
from dataclasses import dataclass


class CUPED(ABCore):
    """
    Класс для расчета CUPED-метрики
    """

    def __init__(self, data: pd.DataFrame, all_groups: dict, *config, **kwargs):
        """
        Инициализация базовых атрибутов класса
        """
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        self.data = data
        self.all_groups = all_groups

    def sort_merge_for_cuped(
        self,
        pre_pilot_df: pd.DataFrame,
        pilot_df: pd.DataFrame,
        all_groups: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Формирование и сортировка датафрейма для cuped'a

        Args:
            pre_pilot_df (pd.DataFrame): данные предпилотного периода
            pilot_df (pd.DataFrame): данные пилотного периода
            all_groups (dict): все юниты теста

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: данные для cuped'a
        """
        # Определяем день недели
        pre_pilot_df["weekday"] = pre_pilot_df[self.config.time_series_field].apply(
            lambda x: self.check_weekday(x)
        )
        pilot_df["weekday"] = pilot_df[self.config.time_series_field].apply(
            lambda x: self.check_weekday(x)
        )
        # Все юниты в тесте и контроле
        all_units = functools.reduce(operator.iconcat, all_groups.values(), [])
        # Предпилотный период
        dates_for_lin = sorted(
            list(set(pre_pilot_df[self.config.time_series_field].values))
        )[-self.config.days_for_test - 1 :]
        pre_pilot_df = pre_pilot_df[
            pre_pilot_df[self.config.time_series_field].isin(dates_for_lin)
            & pre_pilot_df[self.config.id_field].isin(all_units)
        ]
        pilot_df = pilot_df[pilot_df[self.config.id_field].isin(all_units)]
        pilot_df_sort = pilot_df.sort_values(
            [self.config.id_field, self.config.time_series_field]
        )
        pre_pilot_df_sort = pre_pilot_df.sort_values(
            [self.config.id_field, self.config.time_series_field]
        )
        pilot_df_sort["row_number"] = [i for i in range(0, len(pilot_df_sort))]
        pre_pilot_df_sort["row_number"] = [i for i in range(0, len(pre_pilot_df_sort))]
        pilot_df_sort["period"] = "pilot"
        pre_pilot_df_sort["period"] = "history"
        cols = [
            self.config.time_series_field,
            self.config.id_field,
            self.config.target_metric,
            "weekday",
            "row_number",
            "period",
        ]
        self.sortmerge_list = [self.config.id_field, "row_number"]
        return pilot_df_sort[cols], pre_pilot_df_sort[cols]

    def _calculate_theta(
        self,
        *,
        prepilot_period: pd.DataFrame,
        pilot_period: pd.DataFrame,
        is_ratio=False,
    ) -> float:
        """
        Вычисляем Theta

        Args:
            y_prepilot (np.array): значения метрики во время пилота
            y_pilot (np.array): значения ковариант (той же самой метрики) на препилоте
            is_ratio (bool, optional): Если True, то в рассчитывается дисперсия для ratio-метрики

        Returns:
            float: значение коэффициента тета
        """

        self.y_prepilot = (np.array(prepilot_period[self.config.target_metric]),)
        self.y_pilot = np.array(pilot_period[self.config.target_metric])
        covariance = np.cov(self.y_prepilot, self.y_pilot)[0, 1]

        if is_ratio:
            variance = self.delta_method(prepilot_period, self.config.id_field)[
                "var_metric"
            ]
        else:
            variance = np.var(self.y_prepilot)
        theta = covariance / variance
        return theta

    def calculate_cuped_metric(
        self,
        *,
        df_history: pd.DataFrame,
        df_experiment: pd.DataFrame,
        is_ratio: bool = False,
        theta: float = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Расчет CUPED метрики

        Args:
            df_history (pd.DataFrame): таблица с данными предпилотными данными
            df_experiment (pd.DataFrame): таблица с данными пилота
            is_ratio (bool): False, если рассчитывается ratio-метрика
            theta (float): значение тета, если не передано, то рассчитывается в _calculate_theta
            verbose (bool): True, если нужно выводить информацию о расчете

        Returns:
            pd.DataFrame: датафрейм с cuped метрикой
        """
        prepilot_period = df_history[df_history["period"] == "history"].sort_values(
            self.sortmerge_list
        )
        pilot_period = df_experiment[df_experiment["period"] == "pilot"].sort_values(
            self.sortmerge_list
        )
        self.var_before_cuped = np.var(pilot_period[self.config.target_metric])
        if not theta:
            self.theta = self._calculate_theta(
                prepilot_period=prepilot_period,
                pilot_period=pilot_period,
                is_ratio=is_ratio,
            )
        else:
            self.theta = theta
        res = pd.merge(
            prepilot_period,
            pilot_period,
            how="inner",
            on=self.sortmerge_list,
            suffixes=["_prepilot", "_pilot"],
        )
        if verbose:
            print(
                f"Theta is: {self.theta}",
            )
        res[f"{self.config.target_metric}_cuped"] = (
            res[f"{self.config.target_metric}_pilot"]
            - self.theta * res[f"{self.config.target_metric}_prepilot"]
        )
        self.var_after_cuped = np.var(res[f"{self.config.target_metric}_cuped"])
        self.var_diff = np.round(
            (self.var_before_cuped / self.var_after_cuped - 1) * 100, 3
        )
        self.var_analysis_dict = dict(
            var_before_cuped=self.var_before_cuped,
            var_after_cuped=self.var_after_cuped,
            var_diff=self.var_diff,
        )
        return res

    def runner(self, is_pilot_estimation: bool = False) -> pd.DataFrame:
        """
        Метод для сборки всех шагов расчета CUPED-метрики

        Args:
            is_pilot_estimation (str): True, если запускается расчет на пилоте

        Returns:
            cuped_df датафрейм с CUPED-метрикой
        """
        if is_pilot_estimation:
            self.data["test_period"] = self.data[self.config.time_series_field].apply(
                lambda x: self.test_split(
                    x, self.config.start_of_test, self.config.end_of_test
                )
            )
            pilot_df_sort, pre_pilot_df_sort = self.sort_merge_for_cuped(
                self.data[self.data["test_period"] == "pre_pilot"],
                self.data[self.data["test_period"] == "pilot"],
                self.all_groups,
            )
        else:
            pilot_df_sort, pre_pilot_df_sort = self.sort_merge_for_cuped(
                self.data[self.data["periods"] == "knn"],
                self.data[self.data["periods"] == "validation"],
                self.all_groups,
            )
        cuped_df = self.calculate_cuped_metric(
            df_history=pre_pilot_df_sort, df_experiment=pilot_df_sort
        )
        return cuped_df


class CUPAC:
    """
    Класс для расчета CUPAC-метрики
    """

    def __init__(self):
        pass


@dataclass
class AllDecreasers:
    """Датакласс, который содержит все доступные методы для снижения дисперсии."""

    cuped: CUPED
    cupac: CUPAC


class GetDecreaser(ABCore):
    """
    Класс достает все доступные методы для снижения дисперсии.
    """

    def __call__(self, *config, **kwargs):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        decreaser = self.config.decreaser
        if isinstance(decreaser, str):
            decreaser = [decreaser]
        ready_decreaser = {
            i: AllDecreasers.__annotations__[i]
            for i in decreaser
            if i in AllDecreasers.__annotations__
        }
        return ready_decreaser
