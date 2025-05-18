import pandas as pd
import numpy as np
import offline_ab.utils.visualization as viz

from typing import Callable, List
from offline_ab.abcore import ABCore
from dataclasses import dataclass


class Bootstrap(ABCore):
    """
    Бутстрап
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metric_func: Callable,
        control_units: List,
        *config,
        var_analysis_dict: dict = None,
        effect: int = 0,
        is_cuped: bool = True,
        show_plot: str = None,
        **kwargs,
    ):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        self.data = data
        self.metric_func = metric_func
        self.effect = effect
        self.is_cuped = is_cuped
        self.show_plot = show_plot
        self.var_analysis_dict = var_analysis_dict
        self.control_units = control_units

    def prepare_data_for_bootstrap(self) -> dict:
        """
        Готовит массивы данных и размер бутстрап выборки для бутстрапа (will be deprecated)

        Returns:
            dict: словарь с массивами и размером бутстрап выборки
        """
        if self.is_cuped:
            cuped_metric = f"{self.config.target_metric}_cuped"
            control_values = np.array(self.data[self.data[self.config.id_field].isin(self.control_units)][cuped_metric])
            test_values = np.array(
                self.data[self.data[self.config.id_field].isin(self.config.test_units)][cuped_metric]
            )
            assert len(control_values) == len(test_values), f"{len(control_values), len(test_values)}"
        else:
            control_values = np.array(
                self.data[self.data[self.config.id_field].isin(self.control_units)][self.config.target_metric]
            )
            test_values = np.array(
                self.data[self.data[self.config.id_field].isin(self.config.test_units)][self.config.target_metric]
            )
            assert len(control_values) == len(test_values)
        n = len(control_values)
        control_group_true = np.array(
            self.data[self.data[self.config.id_field].isin(self.control_units)][f"{self.config.target_metric}_pilot"]
        )
        test_group_true = np.array(
            self.data[self.data[self.config.id_field].isin(self.config.test_units)][
                f"{self.config.target_metric}_pilot"
            ]
        )
        return dict(
            control_values=control_values,
            test_values=test_values,
            n=n,
            control_group_true_mean=control_group_true,
            test_group_true_mean=test_group_true,
        )

    def runner(self, **kwargs) -> dict:
        """
        Бутстрап

        Returns:
            dict: словарь с рассчитанными параметрами
        """

        def _help_function(func: Callable, group: np.ndarray) -> Callable:
            return func(group)

        # Готовим данные
        vals = self.prepare_data_for_bootstrap(**kwargs)
        test_values = vals["test_values"]
        control_values = vals["control_values"]
        bootstrap_group_length = vals["n"]

        # Бутстрап
        difference_aa = np.zeros(self.config.n_iter_bootstrap)
        difference_ab = np.zeros(self.config.n_iter_bootstrap)
        for i in range(self.config.n_iter_bootstrap):
            random_values_control = np.random.choice(control_values, bootstrap_group_length, True)
            random_values_test = np.random.choice(test_values, bootstrap_group_length, True)
            random_values_test_with_eff = np.random.choice(test_values + self.effect, bootstrap_group_length, True)

            control_metric = _help_function(self.metric_func, random_values_control)
            test_metric = _help_function(self.metric_func, random_values_test)
            test_metric_with_eff = _help_function(self.metric_func, random_values_test_with_eff)

            difference_aa[i] = test_metric - control_metric
            difference_ab[i] = test_metric_with_eff - control_metric
        # Расчет точечных оценок
        point_estimation_aa = _help_function(self.metric_func, test_values) - _help_function(
            self.metric_func, control_values
        )
        point_estimation_ab = _help_function(self.metric_func, test_values + self.effect) - _help_function(
            self.metric_func, control_values
        )
        # Считаем p-value
        adj_diffs_aa = difference_aa - point_estimation_aa
        adj_diffs_ab = difference_ab - point_estimation_ab
        false_positive_aa = np.sum(np.abs(adj_diffs_aa) >= np.abs(point_estimation_aa))
        false_positive_ab = np.sum(np.abs(adj_diffs_ab) >= np.abs(point_estimation_ab))
        p_value_aa_boot = false_positive_aa / self.config.n_iter_bootstrap
        p_value_ab_boot = false_positive_ab / self.config.n_iter_bootstrap

        # Расчет доверительных интервалов
        ci_aa = self.get_percentile_ci(difference_aa)
        ci_ab = self.get_percentile_ci(difference_ab)
        has_effect_aa = not (ci_aa[0] < 0 < ci_aa[1])
        has_effect_ab = not (ci_ab[0] < 0 < ci_ab[1])

        # Расчет эффекта для A/A теста
        effect_aa = np.round(point_estimation_aa / np.mean(vals["test_group_true_mean"]) * 100, 4)
        effect_left_bound_aa = np.round(
            (ci_aa[0] - point_estimation_aa) / np.mean(vals["test_group_true_mean"]) * 100,
            3,
        )
        effect_right_bound_aa = np.round(
            (ci_aa[1] - point_estimation_aa) / np.mean(vals["test_group_true_mean"]) * 100,
            3,
        )
        effect_avg_bound_aa = np.mean([np.abs(effect_left_bound_aa), np.abs(effect_right_bound_aa)])

        # Расчет эффекта для A/B теста
        effect_ab = np.round(
            point_estimation_ab / np.mean(vals["test_group_true_mean"] + self.effect) * 100,
            4,
        )
        effect_left_bound_ab = np.round(
            (ci_ab[0] - point_estimation_ab) / np.mean(vals["test_group_true_mean"] + self.effect) * 100,
            3,
        )
        effect_right_bound_ab = np.round(
            (ci_ab[1] - point_estimation_ab) / np.mean(vals["test_group_true_mean"] + self.effect) * 100,
            3,
        )
        effect_avg_bound_ab = np.mean([np.abs(effect_left_bound_ab), np.abs(effect_right_bound_ab)])

        effect_dict = dict(
            aa=[effect_aa, effect_avg_bound_aa],
            ab=[effect_ab, effect_avg_bound_ab],
            decreas_var=self.var_analysis_dict,
        )
        if self.show_plot == "aa":
            viz.plot_ci(
                difference_aa,
                point_estimation_aa,
                ci_aa,
                p_value_aa_boot,
                effect_dict,
                aa_test=True,
            )
        if self.show_plot == "ab":
            viz.plot_ci(
                difference_ab,
                point_estimation_ab,
                ci_ab,
                p_value_ab_boot,
                effect_dict,
                aa_test=False,
            )
        return dict(
            aa_test=has_effect_aa,
            ab_test=has_effect_ab,
            pe_aa=point_estimation_aa,
            pe_ab=point_estimation_ab,
            effect_aa=effect_aa,
            ci_aa=ci_aa,
            effect_ab=effect_ab,
            ci_ab=ci_ab,
            p_value_aa_boot=p_value_aa_boot,
            p_value_ab_boot=p_value_ab_boot,
            var_diff=self.var_analysis_dict["var_diff"],
        )


@dataclass
class AllEstimators:
    """Датакласс, который содержит все доступные методы для оценки эффекта."""

    bootstrap: Bootstrap


class GetEstimator(ABCore):
    """
    Класс достает все доступные методы для оценки эффекта.
    """

    def __call__(self, *config, **kwargs):
        if config:
            super().__init__(*config)
        if kwargs:
            super().__init__(**kwargs)
        estimator = self.config.estimator
        if isinstance(estimator, str):
            estimator = [estimator]
        ready_estimator = {i: AllEstimators.__annotations__[i] for i in estimator if i in AllEstimators.__annotations__}
        return ready_estimator
