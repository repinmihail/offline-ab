import pandas as pd
import numpy as np
import operator

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List
from dtaidistance import dtw  # type: ignore
from functools import reduce
from dataclasses import dataclass
from offline_ab.abcore import ABCore, StaticHelper


class KNNDTWSelection(ABCore):
    """
    Класс для подбора соседей методом DTW.
    """

    def __init__(self, data: pd.DataFrame, *config, **kwargs):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        data["periods"] = data[self.config.time_series_field].apply(
            lambda x: self.validation_split(
                x,
                self.config.start_of_validation,
                self.config.start_of_test,
                self.config.end_of_test,
            )
        )
        self.data = data[data["periods"] == "knn"]

    def get_scaled_vectors_data(self) -> pd.DataFrame:
        """
        Метод для масштабирования и векторизации данных. Используется StandardScaler

        Raises:
            KeyError: В случае, если наименование столбца с метрикой указано
            неверно, бросается исключение

        Returns:
            pd.DataFrame: исходный датафрейм + масштабированная метрика
        """
        try:
            scaled_metric = StandardScaler().fit_transform(self.data[[self.config.target_metric]])
        except KeyError:
            raise KeyError(f"Frame data does not contain the field with name {self.config.target_metric}")
        self.data[f"scaled_{self.config.target_metric}"] = scaled_metric
        self.data = self.data.sort_values(by=[self.config.id_field, self.config.time_series_field])
        self.data_vec = (
            self.data.groupby(self.config.id_field).agg({f"scaled_{self.config.target_metric}": list}).reset_index()
        )
        self.data_vec[f"{self.config.target_metric}_array"] = [
            np.array(i) for i in self.data_vec[f"scaled_{self.config.target_metric}"]
        ]
        keys = self.data_vec[self.config.id_field].tolist()
        vals = self.data_vec[f"{self.config.target_metric}_array"].tolist()
        self.ids_dict = dict(zip([i for i in range(0, len(keys))], keys))
        return dict(zip(keys, vals))

    def get_all_neighbors_dtw(self, vectors: dict) -> dict:
        """
        Находит ближайших соседей по расстоянию DTW для заданного временного ряда

        Args:
            vectors (dict): словарь, где ключи — идентификаторы наблюдений (str), значения — вектора временных рядов

        Returns:
            dict (dict): словарь, где ключи — идентификаторы ближайших соседей,
                значения — словарь: сосед и расстояние DTW
        """
        self.unit_distances_dtw = {unit: [] for unit in self.config.test_units}
        for target_unit in self.config.test_units:
            distances = {}
            for key, vector in vectors.items():
                if key not in self.config.test_units:
                    distances[key] = dtw.distance(vector, vectors[target_unit])
            sorted_distances = sorted(distances.items(), key=lambda item: item[1])
            self.unit_distances_dtw[target_unit] = dict(sorted_distances[: self.config.n_neighbors_dtw])
        return self.unit_distances_dtw

    def get_test_control_groups(self, neighbors_dict: dict, adj_control: List[str] = None) -> dict:
        """
        Формирует словарь со списками тестовых и контрольных юнитов в значениях словаря

        Args:
            neighbors_dict (dict): словарь с наменованием юнита в ключе и списком соседей в значении
            adj_control (List[str]): лист с юнитами контрольной группы, скорректированной вручную

        Returns:
            dict: итоговый словарь {test_units: [str], control_units: [str]}
        """
        if not adj_control:
            unique_closest_cities = self.find_unique_closest_units(neighbors_dict)
            self.control_units = [list(val)[0] for _, val in unique_closest_cities.items()]
        else:
            self.control_units = adj_control
        return dict(
            test_units=self.config.test_units,
            control_units=self.control_units,
        )

    def runner(self) -> dict:
        """
        Основная функция, которая запускает все функции для получения данных для обучения модели dtw
        """
        vectors_data = self.get_scaled_vectors_data()
        all_neighbours_dtw = self.get_all_neighbors_dtw(vectors_data)
        all_groups_dtw = self.get_test_control_groups(all_neighbours_dtw)
        self.all_units = reduce(operator.iconcat, all_groups_dtw.values(), [])
        return all_groups_dtw


class KNNEUCLSelection(ABCore, StaticHelper):
    """
    Класс для подбора соседей через Евклидово расстояние.
    """

    def __init__(self, data: pd.DataFrame, *config, **kwargs):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        data["periods"] = data[self.config.time_series_field].apply(
            lambda x: self.validation_split(
                x,
                self.config.start_of_validation,
                self.config.start_of_test,
                self.config.end_of_test,
            )
        )
        self.data = data[data["periods"] == "knn"]

    def get_scaled_vectors_data(self) -> pd.DataFrame:
        """
        Метод для масштабирования и векторизации данных. Используется StandardScaler

        Raises:
            KeyError: В случае, если наименование столбца с метрикой указано
            неверно, бросается исключение

        Returns:
            pd.DataFrame: исходный датафрейм + масштабированная метрика
        """
        try:
            scaled_metric = StandardScaler().fit_transform(self.data[[self.config.target_metric]])
        except KeyError:
            raise KeyError(f"Frame data does not contain the field with name {self.config.target_metric}")
        self.data[f"scaled_{self.config.target_metric}"] = scaled_metric
        self.data = self.data.sort_values(by=[self.config.id_field, self.config.time_series_field])
        self.data_vec = (
            self.data.groupby(self.config.id_field).agg({f"scaled_{self.config.target_metric}": list}).reset_index()
        )
        self.data_vec[f"{self.config.target_metric}_array"] = [
            np.array(i) for i in self.data_vec[f"scaled_{self.config.target_metric}"]
        ]
        keys = self.data_vec[self.config.id_field].tolist()
        vals = self.data_vec[f"{self.config.target_metric}_array"].tolist()
        self.ids_dict = dict(zip([i for i in range(0, len(keys))], keys))
        return dict(zip(keys, vals))

    def get_all_neighbors_eucl(self, vectors: dict, algorithm="auto") -> dict:
        """
        Возвращает k ближайших соседей для одного заданного юнита

        Args:
            vectors (dict): словарь с наименованием юнита в ключе и вектором метрики в значении
            algorithm (str, optional): алгорит подбора соседей. Defaults to 'auto'.

        Returns:
            dict (dict): словарь, где ключи — идентификаторы ближайших соседей,
                значения — словарь: сосед и евклидово расстояние
        """

        def get_knn(vectors):
            vector_arrays = [list(i) for i in vectors.values()]
            return NearestNeighbors(n_neighbors=self.config.n_neighbors_eucl + 1, algorithm=algorithm).fit(
                vector_arrays
            )

        def flatten_neighbour_list(distance, ids):
            dist_list, nb_list = distance.tolist(), ids.tolist()
            return dist_list[0], nb_list[0]

        self.unit_distances_eucl = {unit: [] for unit in self.config.test_units}
        # из обучающей выборки исключены тестовые юниты
        knn = get_knn({key: val for key, val in vectors.items() if key not in self.config.test_units})
        for target_unit in self.config.test_units:
            vector = vectors[target_unit].reshape(1, -1)
            dist, nb_indexes = knn.kneighbors(vector, self.config.n_neighbors_eucl, return_distance=True)
            return_dist, return_nb_indexes = flatten_neighbour_list(dist, nb_indexes)
            return_names = [self.ids_dict[i] for i in return_nb_indexes]
            distances_eucl_ = {
                key: val for key, val in dict(zip(return_names, return_dist)).items() if key != target_unit
            }
            self.unit_distances_eucl[target_unit] = distances_eucl_
        return self.unit_distances_eucl

    def get_test_control_groups(self, neighbors_dict: dict, adj_control: List[str] = None) -> dict:
        """
        Формирует словарь со списками тестовых и контрольных юнитов в значениях словаря

        Args:
            neighbors_dict (dict): словарь с наменованием юнита в ключе и списком соседей в значении
            adj_control (List[str]): лист с юнитами контрольной группы, скорректированной вручную

        Returns:
            dict: итоговый словарь {test_units: [str], control_units: [str]}
        """
        if not adj_control:
            unique_closest_cities = self.find_unique_closest_units(neighbors_dict)
            self.control_units = [list(val)[0] for _, val in unique_closest_cities.items()]
        else:
            self.control_units = adj_control
        return dict(
            test_units=self.config.test_units,
            control_units=self.control_units,
        )

    def runner(self):
        """
        Основная функция, которая запускает все функции для получения данных для обучения модели eucl
        """
        vectors_data = self.get_scaled_vectors_data()
        all_neighbours_eucl = self.get_all_neighbors_eucl(vectors_data)
        all_groups_eucl = self.get_test_control_groups(all_neighbours_eucl)
        self.all_units = reduce(operator.iconcat, all_groups_eucl.values(), [])
        return all_groups_eucl


class PSMSelection:
    def __init__(self):
        pass


@dataclass
class AllSelectors:
    """Class to hold all selectors"""

    knn_dtw: KNNDTWSelection
    knn_eucl: KNNEUCLSelection


class GetSelectors(ABCore):
    def __call__(self, *config, **kwargs):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        selectors = self.config.selectors
        ready_selectors = {i: AllSelectors.__annotations__[i] for i in selectors if i in AllSelectors.__annotations__}
        return ready_selectors
