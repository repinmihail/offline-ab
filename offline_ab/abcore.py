# version 0.0.1

import pandas as pd
import numpy as np
import datetime
import operator
import warnings

from functools import reduce
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Union
from datetime import datetime  # noqa: F811
from etna.datasets.tsdataset import TSDataset
from offline_ab.gapfillings import FillTheGaps
from offline_ab.configreader import ConfigReader

warnings.filterwarnings("ignore")


class StaticHelper:
    def __init__(self):
        pass

    @staticmethod
    def check_weekday(date: str) -> int:
        """
        Возвращает день недели

        Args:
            date (str): дата в формате %Y-%m-%d

        Returns:
            int: день недели
        """
        given_date = datetime.strptime(date, "%Y-%m-%d")
        day_of_week = given_date.weekday() + 1
        return day_of_week

    @staticmethod
    def multitest_correction(
        *,
        list_of_pvals: List,
        alpha: float = 0.05,
        method: str = "holm",
    ) -> dict:
        """
        Корректировка p-value для множественной проверки гипотез.

        Args:
            list_of_pvals - массив рассчитанных p-value значений
            alpha - уровень ошибки первого рода
            method - метод поправки, default: 'holm', подробнее по ссылке
                https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        Returns:
            dict: словарь с корректированными p-value значениями
        """
        decision, adj_pvals, sidak_aplha, bonf_alpha = multipletests(
            pvals=list_of_pvals, alpha=alpha, method=method
        )
        return dict(
            decision=list(decision),
            adjusted_pvals=[np.round(i, 10) for i in adj_pvals],
            sidak_aplha=sidak_aplha,
            bonf_alpha=bonf_alpha,
        )

    @staticmethod
    def validation_split(
        x: str, start_of_validation: str, start_of_test: str, end_of_test: str
    ) -> str:
        """
        Разметка данных.

        Args:
            x (str): строка данных
            start_of_validation (str): начало валидации
            start_of_test (str): начало теста
            end_of_test (str): конец теста

        Returns:
            str: строка данных с разметкой
        """
        if str(x) < start_of_validation:
            return "knn"
        if start_of_validation <= str(x) < start_of_test:
            return "validation"
        if start_of_test <= str(x) < end_of_test:
            return "test"
        if end_of_test > str(x):
            return "post_pilot"

    @staticmethod
    def test_split(x: str, start_of_test: str, end_of_test: str) -> str:
        """
        Разметка данных для теста.

        Args:
            x (str): строка данных
            start_of_test (str): начало теста
            end_of_test (str): конец теста
        Returns:
            str: строка данных с разметкой
        """
        if start_of_test <= str(x) <= end_of_test:
            return "pilot"
        if str(x) < start_of_test:
            return "pre_pilot"
        if end_of_test > str(x):
            return "post_pilot"

    @staticmethod
    def find_unique_closest_units(units_dict: dict) -> dict:
        """
        Функция принимает словарь юнитов с расстояниями до ближайших соседей и возвращает уникальные
        ближайшие юниты с учетом минимального расстояния.

        Args:
            units_dict: dict, словарь с контрольными юнитами и их ближайшими соседями

        Return: dict, словарь с уникальными ближайшими юнитами
        """
        result = {}
        used_units = set()
        # Множество ключей (контрольных юнитов)
        base_units = set(units_dict.keys())
        # Проходим по каждому контрольному юниту
        for base_unit, neighbors in units_dict.items():
            # Сортируем соседей по расстоянию
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1])
            for neighbor_unit, distance in sorted_neighbors:
                # юнит не должен быть среди уже использованных и среди ключей
                if neighbor_unit not in used_units and neighbor_unit not in base_units:
                    result[base_unit] = (neighbor_unit, distance)
                    used_units.add(neighbor_unit)
                    break
        return result


class ABCore(StaticHelper):
    """
    Основной класс с аналитическими функциями.
    """

    def __init__(self, *path: str, **kwargs):
        if path:
            self.config = ConfigReader(*path)
            self.path = path[0]
        else:
            self.config = ConfigReader(**kwargs)

    def get_percentile_ci(self, bootstrap_stats: Union[List[float]]):
        """
        Строит перцентильный доверительный интервал

        Args:
            bootstrap_stats (List[float]): бутстрапированная статистика

        Returns:
            Tuple[float, float]: границы доверительного интервала
        """
        left, right = np.quantile(
            bootstrap_stats, [self.config.alpha / 2, 1 - self.config.alpha / 2]
        )
        return left, right

    def get_normal_ci(
        self, bootstrap_stats: Union[np.array, List], pe: float
    ) -> Tuple[float, float]:
        """
        Строит нормальный доверительный интервал.

        Args:
            bootstrap_stats (Union[np.array, List]): массив значений посчитанной метрики
            pe (float): точечная оценка (рассчитывается на исходных данных)

        Returns:
            Tuple[float, float]: границы доверительного интервала
        """
        z = stats.norm.ppf(1 - self.config.alpha / 2)
        se = np.std(bootstrap_stats)
        left, right = pe - z * se, pe + z * se
        return left, right

    def delta_method(
        self, data: pd.DataFrame, column_for_grouped: list, is_sample: bool = False
    ) -> dict:
        """
        Дельта-метод

        Args:
            data (pd.DataFrame): датафрейм с метрикой
            column_for_grouped (list): поля для группировки (идентификатор)
            is_sample (bool, optional): выборочная оценка. Defaults to False.

        Returns:
            dict: словарь с рассчитанными параметрами
        """

        delta_df = data.groupby(column_for_grouped).agg(
            {self.config.target_metric: ["sum", "count"]}
        )
        n_users = len(delta_df)
        delta_df.columns = ["_".join(col).strip() for col in delta_df.columns.values]
        array_x = delta_df[f"{self.config.target_metric}_sum"].values
        array_y = delta_df[f"{self.config.target_metric}_count"].values
        mean_x, mean_y = np.mean(array_x), np.mean(array_y)
        var_x, var_y = np.var(array_x), np.var(array_y)
        cov_xy = np.cov(array_x, array_y)[0, 1]
        var_metric = (
            var_x / mean_y**2
            - 2 * (mean_x / mean_y**3) * cov_xy
            + (mean_x**2 / mean_y**4) * var_y
        )
        if is_sample:
            var_metric = var_metric / n_users
        info_dict = {}
        info_dict["mean_x, mean_y"] = [mean_x, mean_y]
        info_dict["var_x, var_y"] = [var_x, var_y]
        info_dict["cov_xy"] = cov_xy
        info_dict["n_users"] = n_users
        info_dict["var_metric"] = var_metric
        info_dict["std_metric"] = np.sqrt(var_metric)
        return info_dict

    def get_max_missing_in_current_exp(
        self, seasonality: int = 7, max_days: int = 140, max_gaps: int = 5
    ) -> dict:
        """
        Метод для расчета максимально возможного количества пропущенных дней в текущем эксперименте.

        Args:
            seasonality (int): сезонность для дневной гранулярности
            max_days (int): максимальное количество дней в эксперименте
            max_gaps (int): максимальное количество пропусков

        Returns:
            max_missing (dict): словарь с максимально возможным количеством пропущенных дней для каждого периода
        """
        keys = [i for i in range(seasonality, max_days, seasonality)]
        vals = [i if i < max_gaps else max_gaps for i in range(len(keys))]
        constraint_dict = dict(zip(keys, vals))
        # посчитаем, сколько всего можно допустить пропусков для каждого периода
        max_missing = dict()
        for key, val in constraint_dict.items():
            if self.config.days_for_test == key:
                max_missing["knn"] = val
            if self.config.days_for_test == key:
                max_missing["validation"] = val
            if self.config.days_for_test == key:
                max_missing["test"] = val
        return max_missing

    def rename_df_to_etna(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Метод для переименования полей в DataFrame под формат TSDataset.

        Args:
            df (pd.DataFrame): исходный pandas DataFrame
            metric (str): поле метрики

        Returns:
            df_mod (pd.DataFrame): модифицированный DataFrame
        """
        rename_dict = dict(
            zip(
                [self.config.time_series_field, self.config.id_field, metric],
                ["timestamp", "segment", "target"],
            )
        )
        df_mod = df[[self.config.time_series_field, self.config.id_field, metric]]
        return df_mod.rename(columns=rename_dict)

    def rename_df_to_pandas(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Метод для переименования полей в pandas из формата TSDataset.

        Args:
            df (pd.DataFrame): исходный pandas DataFrame
            metric (str): поле метрики

        Returns:
            df_mod (pd.DataFrame): модифицированный DataFrame
        """
        rename_dict = dict(
            zip(
                ["timestamp", "segment", "target"],
                [self.config.time_series_field, self.config.id_field, metric],
            )
        )
        return df.rename(columns=rename_dict)

    def get_df_with_missing_values(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Формирует датафрейм с пропусками.

        Args:
            df (pd.DataFrame): исходный pandas DataFrame
            metric (str): поле метрики

        Returns:
            ts_test (TSDataset): модифицированный DataFrame
        """
        df = self.rename_df_to_etna(df, metric)
        ts_test = TSDataset(df, freq="D").describe()
        return ts_test[ts_test["num_missing"] > 0]

    def get_critical_units(
        self, df: pd.DataFrame, max_missing: dict, metric: str
    ) -> dict:
        """
        Возвращает юниты с большим количеством пропусков.

        Args:
            df (pd.DataFrame): исходный pandas DataFrame
            max_missing (dict): максимальное количество пропусков
            metric (str): поле метрики

        Returns:
            critical_units (dict): юниты с большим количеством пропусков
        """
        critical_units = dict()
        df["periods"] = df[self.config.time_series_field].apply(
            lambda x: self.validation_split(
                x,
                self.config.start_of_validation,
                self.config.start_of_test,
                self.config.end_of_test,
            )
        )
        for key, val in max_missing.items():
            temp_df = df[df["periods"] == key][
                [self.config.time_series_field, self.config.id_field, metric]
            ]
            temp_miss = self.get_df_with_missing_values(temp_df, metric)[
                ["num_missing"]
            ]
            temp_miss["is_critical_num"] = temp_miss["num_missing"].apply(
                lambda x: 1 if x > val else 0
            )
            critical_units[key] = list(
                temp_miss[temp_miss["is_critical_num"] == 1].index
            )
        return critical_units

    def get_clear_data(self, df: pd.DataFrame, critical_units: dict) -> pd.DataFrame:
        """
        Возвращает DataFrame без пропусков.

        Args:
            df (pd.DataFrame): исходный pandas DataFrame
            critical_units (dict): юниты с большим количеством пропусков

        Returns:
            clear_data (pd.DataFrame): DataFrame без пропусков
        """
        all_critical_units = set(reduce(operator.iconcat, critical_units.values(), []))
        clear_data = df[~df[self.config.id_field].isin(all_critical_units)]
        return clear_data

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, dict]:
        """
        Метод для подготовки данных для всех метрик теста.

        Args:
            df (pd.DataFrame): Данные для теста
            all_metrics (List[str]): Список всех метрик теста
            **kwargs: Дополнительные параметры для стратегии заполнения пропусков

        Returns:
            ready_data (pd.DataFrame): Подготовленные данные
            critical_units (dict): Юниты с большим количеством пропусков
        """
        fill_the_gaps = FillTheGaps()
        ready_data = pd.DataFrame()
        critical_units = dict()
        max_missing = self.get_max_missing_in_current_exp()
        for metric in self.config.all_metrics:
            single_metric_df = df[
                [self.config.time_series_field, self.config.id_field, metric]
            ]
            critical_units_ = self.get_critical_units(
                single_metric_df, max_missing, metric
            )
            critical_units[metric] = critical_units_
            if critical_units_:
                clear_data_ = self.get_clear_data(single_metric_df, critical_units_)
                clear_data = self.rename_df_to_etna(clear_data_, metric)
                filling_data_ = fill_the_gaps(clear_data, **kwargs)
            else:
                filling_data_ = fill_the_gaps(single_metric_df, **kwargs)
            filling_data = self.rename_df_to_pandas(filling_data_, metric)
            if ready_data.empty:
                ready_data = filling_data
            else:
                ready_data = pd.merge(
                    ready_data,
                    filling_data,
                    how="inner",
                    on=[self.config.id_field, self.config.time_series_field],
                )
        # проверяем, остались ли NaN в данных,
        # если да, то запоминаем юниты и удаляем их
        units_with_nan = []
        for metric in self.config.all_metrics:
            units_with_nan.extend(
                list(
                    set(
                        ready_data[
                            ready_data[metric].isna()
                        ][self.config.id_field].tolist()
                    )
                )
            )
        ready_data_ = ready_data[
            ~ready_data[self.config.id_field].isin(units_with_nan)
        ]
        # проверяем, остались ли тестовые юниты в данных,
        # если нет, то обновляем атрибуты
        empty_units = []
        for unit in self.config.test_units:
            if unit not in set(ready_data_[self.config.id_field]):
                empty_units.append(unit)
        if empty_units:
            print(
                f"Найдено {len(empty_units)} тестовых юнитов \
                    с пропущенными значениями: {empty_units}. Они будут исключены из тестовой группы.")
            self.config.test_units = [i for i in self.config.test_units if i not in empty_units]
        return ready_data_, critical_units
