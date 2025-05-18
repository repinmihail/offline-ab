import pandas as pd
import numpy as np

from collections import defaultdict
from tqdm.notebook import tqdm
from typing import List
from offline_ab.abcore import ABCore
from offline_ab.transformers.selectors import GetSelectors
from offline_ab.transformers.vardecreasers import GetDecreaser
from offline_ab.transformers.estimators import GetEstimator


class CrossValEstimation(ABCore):
    """
    Класс для проведения кросс-валидации и оценки пилота.
    """

    def __init__(self, *config, **kwargs):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        self.true_start_of_test = self.config.start_of_test

    def get_timeseries_bounds(self, df: pd.DataFrame) -> List[str]:
        """
        Метод возвращает минимальную и максимальную дату в наборе данных.

        Args:
            df (pd.DataFrame): Предобработанный датафрейм со всеми метриками теста

        Returns:
            validation_min_start_date, validation_max_end_date (str): Минимальная и максимальная даты
        """
        self.validation_min_start_date = sorted(list(set(df[self.config.time_series_field].tolist())))[
            self.config.days_for_test * 2 + 2
        ]
        self.validation_max_start_date = sorted(list(set(df[self.config.time_series_field].tolist())))[
            -self.config.days_for_test * 2 - 1
        ]
        len_timeseries_in_df = len(set(df[self.config.time_series_field].tolist()))
        if self.config.days_for_test * 2 > len_timeseries_in_df:
            raise ValueError("Задан слишком большой период для теста. Не хватает данных для кросс-валидации")
        date_list = pd.date_range(
            start=self.validation_min_start_date,
            end=self.validation_max_start_date,
            freq="D",
        )
        validation_date_list = [str(i)[:10] for i in date_list]
        return validation_date_list

    def cross_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Метод для time-series кросс-валидации одной целевой метрики

        Args:
            df (pd.DataFrame): предобработанный датафрейм, где должна быть целевая метрика

        Raises:
            AssertionError: если селектор ошибся, то ошибка запоминается в счетчик

        Returns:
            cv_output (pd.DataFrame): датафрейм с результатами кросс-валидации с полями
                fpr и conf_int для каждого метода оценки
        """
        selectors = GetSelectors(**vars(self.config))
        estimator = GetEstimator(**vars(self.config))
        decreaser = GetDecreaser(**vars(self.config))
        self.saving_results = defaultdict(dict, {k: {} for k in selectors().keys()})
        self.skip_exps = defaultdict(int, {k: 0 for k in selectors().keys()})
        self.validation_date_list = self.get_timeseries_bounds(df)
        for i in tqdm(range(0, len(self.validation_date_list))):
            self.config.start_of_test = self.validation_date_list[i]
            exp = ABCore(**vars(self.config))
            # selectors
            for name, selector in selectors().items():
                selector = selector(df, **vars(exp.config))
                all_groups = selector.runner()
                # decreaser
                # пока decreaser только один, поэтому результат не запоминаем
                # если добавятся еще, то нужно будет изменить этот блок и в self.saving_results
                for _, decrease_func in decreaser().items():
                    decrease_inst = decrease_func(df, all_groups, **vars(self.config))
                    decreaser_df = decrease_inst.runner()
                # estimator
                # пока estimator только один, то результат не запоминаем
                for _, estimator_func in estimator().items():
                    try:
                        estimator_inst = estimator_func(
                            decreaser_df,
                            np.mean,
                            selector.control_units,
                            var_analysis_dict=decrease_inst.var_analysis_dict,
                            **vars(self.config),
                        )
                        return_dict = estimator_inst.runner()
                        self.saving_results[name][self.validation_date_list[i]] = return_dict
                    except AssertionError:
                        self.skip_exps[name] += 1
                        if self.skip_exps[name] % 5 == 0:
                            print("skip state =", self.skip_exps)
                    continue
                continue

        # result df
        res_dict = defaultdict(dict, {k: {} for k in (selectors().keys())})
        for key, val in self.saving_results.items():
            help_dict = dict(cv_fpr=[], cv_conf_int=[], count_iter=[])
            for _, v in val.items():
                help_dict["cv_fpr"].append(v["aa_test"] * 1)
                help_dict["cv_conf_int"].append(v["ci_aa"][1] - v["ci_aa"][0])
            help_dict["count_iter"].append(len(val))
            res_dict[key] = help_dict
        calc_res = {
            key: {sub_key: np.mean(sub_val) for sub_key, sub_val in val.items()} for key, val in res_dict.items()
        }
        self.cv_output = pd.DataFrame.from_dict(calc_res, orient="index")
        return self.cv_output

    def cross_validation_for_all_metrics(self, df: pd.DataFrame) -> dict:
        """
        Метод для запуска кросс-валидации по всем метрикам (all_metrics)

        Args:
            df (pd.DataFrame): предобработанный датафрейм, где должна быть целевая метрика

        Returns:
            best_selectors_by_metric (dict): словарь, где в ключе указана метрика,
                а в значении лучший для нее селектор
        """
        self.all_metrics_cv_result_df = pd.DataFrame()
        for metric in self.config.all_metrics:
            self.config.target_metric = metric
            print("Running cross-validation for metric: ", self.config.target_metric)
            metric_df = self.cross_validation(df)
            metric_df = metric_df.sort_values("cv_fpr", ascending=True)
            metric_df["metric"] = metric

            # явно прописываем единственные пока варианты
            metric_df["estimator"] = self.config.estimator
            metric_df["decreaser"] = self.config.decreaser

            metric_df["row_number"] = [i for i in range(metric_df.shape[0])]
            self.all_metrics_cv_result_df = pd.concat([self.all_metrics_cv_result_df, metric_df])

        self.best_selectors_cv = (
            self.all_metrics_cv_result_df[self.all_metrics_cv_result_df["row_number"] == 0]
            .reset_index(names="selector")
            .drop("row_number", axis=1)
        )
        selectors = GetSelectors(**vars(self.config))
        self.best_selectors_by_metric = {
            i: selectors()[
                self.best_selectors_cv[self.best_selectors_cv["metric"] == i].reset_index().loc[0, "selector"]
            ]
            for i in self.best_selectors_cv["metric"]
        }
        return self.best_selectors_by_metric

    def estimation_pilot(self, df: pd.DataFrame, custom_selectors: dict = None) -> pd.DataFrame:
        """
        Метод для оценки пилота с помощью лучших селекторов, отобранных на кросс-валидации для каждой метрики

        Args:
            df (pd.DataFrame): предобработанный датафрейм, где должна быть целевая метрика
            custom_selectors (dict, optional): словарь, где в ключе указана метрика,
                а в значении заданный для нее селектор

        Returns:
            estimation_result_df (pd.DataFrame): результат оценки пилота
        """
        if custom_selectors:
            best_selectors = custom_selectors
        else:
            best_selectors = self.cross_validation_for_all_metrics(df)
        self.config.start_of_test = self.true_start_of_test
        self.estimation_result_dict = defaultdict(dict, {k: {} for k in (best_selectors.keys())})
        for metric in self.config.all_metrics:
            print("Running estimation for metric: ", metric)
            self.config.target_metric = metric
            exp = ABCore(**vars(self.config))
            selector = best_selectors[metric](df, **vars(exp.config))
            all_groups = selector.runner()

            # decreaser
            best_decreaser = GetDecreaser(**vars(self.config))
            for _, decrease_func in best_decreaser().items():
                decrease_inst = decrease_func(df, all_groups, **vars(self.config))
                self.decreaser_df = decrease_inst.runner(is_pilot_estimation=True)

            # estimator
            best_estimator = GetEstimator(**vars(self.config))
            for _, estimator_func in best_estimator().items():
                estimator_inst = estimator_func(
                    self.decreaser_df,
                    np.mean,
                    selector.control_units,
                    var_analysis_dict=decrease_inst.var_analysis_dict,
                    **vars(self.config),
                )
                return_dict = estimator_inst.runner()
            self.estimation_result_dict[metric] = return_dict
        estimation_result_df = pd.DataFrame.from_dict(self.estimation_result_dict, orient="index").reset_index(
            names="metric"
        )
        try:
            self.estimation_result_df = estimation_result_df.merge(self.best_selectors_cv, how="left", on="metric")
        except AttributeError:
            print("Results without data from cross-validation")
            self.estimation_result_df = estimation_result_df
        if len(self.config.all_metrics) > 1:
            self.pvals_before_correction = self.estimation_result_df["p_value_ab_boot"].tolist()
            adjusted_results = self.multitest_correction(
                list_of_pvals=self.estimation_result_df["p_value_ab_boot"].tolist()
            )
            self.estimation_result_df["adjusted_p_value_ab"] = adjusted_results["adjusted_pvals"]
            self.estimation_result_df["adjusted_decision"] = adjusted_results["decision"]

        return self.estimation_result_df
