import yaml

from datetime import timedelta, datetime


class ConfigReader:
    """
    Класс для чтения конфигурации эксперимента из файла.
    """

    def __init__(self, *config, **kwargs):
        if config:
            with open(*config, "r") as yamlfile:
                params_dict = yaml.load(yamlfile, Loader=yaml.FullLoader)
        if kwargs:
            params_dict = kwargs

        # metrics
        self.target_metric = params_dict["target_metric"]
        if "additional_metrics" in params_dict:
            self.additional_metrics = params_dict["additional_metrics"]
            self.all_metrics = [self.target_metric] + self.additional_metrics
        else:
            self.all_metrics = [self.target_metric]

        # data
        self.id_field = params_dict["id_field"]
        self.time_series_field = params_dict["time_series_field"]
        self.test_units = params_dict["test_units"]

        # days
        self.days_for_test = params_dict["days_for_test"]

        # stat default params
        if "n_iter_bootstrap" in params_dict:
            if isinstance(params_dict["n_iter_bootstrap"], int):
                self.n_iter_bootstrap = params_dict["n_iter_bootstrap"]
            else:
                raise TypeError(
                    """Параметр n_iter_bootstrap должен быть в типе данных int"""
                )
        else:
            self.n_iter_bootstrap = 100_000

        if "max_missing_values" in params_dict:
            if isinstance(params_dict["max_missing_values"], int):
                self.max_missing_values = params_dict["max_missing_values"]
            else:
                raise TypeError(
                    """Параметр max_missing_values должен быть в типе данных int"""
                )
        else:
            self.max_missing_values = 5

        if "n_neighbors_dtw" in params_dict:
            if isinstance(params_dict["n_neighbors_dtw"], int):
                self.n_neighbors_dtw = params_dict["n_neighbors_dtw"]
            else:
                raise TypeError(
                    """Параметр n_neighbors_dtw должен быть в типе данных int"""
                )
        else:
            self.n_neighbors_dtw = 5

        if "n_neighbors_eucl" in params_dict:
            if isinstance(params_dict["n_neighbors_eucl"], int):
                self.n_neighbors_eucl = params_dict["n_neighbors_eucl"]
            else:
                raise TypeError(
                    """Параметр n_neighbors_eucl должен быть в типе данных int"""
                )
        else:
            self.n_neighbors_eucl = 5

        if "alpha" in params_dict:
            if isinstance(params_dict["alpha"], float):
                self.alpha = params_dict["alpha"]
            else:
                raise TypeError("""Параметр alpha должен быть в типе данных float""")
        else:
            self.alpha = 0.05

        if "estimator" in params_dict:
            if isinstance(params_dict["estimator"], str):
                self.estimator = params_dict["estimator"]
            else:
                raise TypeError("""Параметр estimator должен быть в типе данных str""")
        else:
            self.estimator = "bootstrap"

        if "decreaser" in params_dict:
            if isinstance(params_dict["decreaser"], str):
                self.decreaser = params_dict["decreaser"]
            else:
                raise TypeError("""Параметр decreaser должен быть в типе данных str""")
        else:
            self.decreaser = "cuped"

        if "selectors" in params_dict:
            self.selectors = params_dict["selectors"]
        else:
            self.selectors = ["knn_dtw", "knn_eucl"]

        # dates
        self.start_of_test = params_dict["start_of_test"]
        if kwargs:
            self.start_of_test = params_dict["start_of_test"]
        self.end_of_test = (
            datetime.strptime(self.start_of_test, "%Y-%m-%d")
            + timedelta(days=self.days_for_test)
        ).strftime("%Y-%m-%d")
        self.start_of_validation = (
            datetime.strptime(self.start_of_test, "%Y-%m-%d")
            + timedelta(days=-(self.days_for_test + 1))
        ).strftime("%Y-%m-%d")
        self.start_of_knn = (
            datetime.strptime(self.start_of_validation, "%Y-%m-%d")
            + timedelta(days=-(self.days_for_test + 2))
        ).strftime("%Y-%m-%d")
