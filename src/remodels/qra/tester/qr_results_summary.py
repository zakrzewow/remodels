"""QR Results Summary."""

from typing import Callable
from typing import Dict
from typing import List

import pandas as pd

from .qr_tester import QR_TestResults


class QR_ResultsSummary:
    """A class to summarize QR Test Results."""

    table_styles = [
        {"selector": ".red", "props": "background-color: #fcb8b8;"},
        {"selector": ".green", "props": "background-color: #b8fcc3;"},
        {"selector": ".bold", "props": "font-weight: bold;"},
    ]

    def __init__(self, results_dict: Dict[str, Dict[str, QR_TestResults]]) -> None:
        """Initialize the QR Results Summary class.

        :param results_dict: dictionary of results by dataset and QR variant
        :type results_dict: Dict[str, Dict[str, QR_TestResults]]
        """
        self.results_dict = results_dict
        self.dataset_names = results_dict.keys()

        self.qr_variants = []
        for dataset_results in results_dict.values():
            for qr_variant_name in dataset_results.keys():
                if qr_variant_name not in self.qr_variants:
                    self.qr_variants.append(qr_variant_name)

    def aec(self, alpha_list: List[int]) -> pd.DataFrame:
        """Average empirical coverage.

        :param alpha_list: length of prediction interval list
        :type alpha_list: List[int]
        :return: aec summary
        :rtype: pd.DataFrame
        """
        df = self._alpha_dataset_variant_summary(alpha_list, QR_TestResults.aec)
        style_df = self._get_style_df(df)

        for alpha in alpha_list:
            for dataset_name in self.dataset_names:
                for qr_variant in self.qr_variants:
                    value = df.at[qr_variant, (alpha, dataset_name)] * 100
                    if abs(alpha - value) < 1:
                        style_df.at[qr_variant, (alpha, dataset_name)] += "green "
                    elif abs(alpha - value) > 2.5:
                        style_df.at[qr_variant, (alpha, dataset_name)] += "red "
                    if (
                        abs(value - alpha)
                        == (df.loc[:, (alpha, dataset_name)] * 100 - alpha).abs().min()
                    ):
                        style_df.at[qr_variant, (alpha, dataset_name)] += "bold "

        return (
            df.astype(float)
            .applymap(lambda x: f"{(x*100):.2f}")
            .style.set_table_styles(self.table_styles, overwrite=True)
            .set_td_classes(style_df)
            .set_caption("Average empirical coverage")
        )

    def kupiec_test(self, alpha_list: List[int]) -> pd.DataFrame:
        """Kupiec test.

        :param alpha_list: length of prediction interval list
        :type alpha_list: List[int]
        :return: Kupiec test summary
        :rtype: pd.DataFrame
        """
        df = self._alpha_dataset_variant_summary(alpha_list, QR_TestResults.kupiec_test)
        style_df = self._get_style_df(df)

        for alpha in alpha_list:
            for dataset_name in self.dataset_names:
                for qr_variant in self.qr_variants:
                    value = df.at[qr_variant, (alpha, dataset_name)]
                    if value >= 20:
                        style_df.at[qr_variant, (alpha, dataset_name)] += "green "
                    elif value < 12:
                        style_df.at[qr_variant, (alpha, dataset_name)] += "red "

        return (
            df.astype(int)
            .style.set_table_styles(self.table_styles, overwrite=True)
            .set_td_classes(style_df)
            .set_caption("Kupiec test")
        )

    def _alpha_dataset_variant_summary(self, alpha_list: List[int], func: Callable):
        df = pd.DataFrame(
            index=self.qr_variants,
            columns=pd.MultiIndex.from_product(
                [alpha_list, self.dataset_names], names=["alpha", "dataset"]
            ),
        )

        for alpha in alpha_list:
            for dataset_name in self.dataset_names:
                for qr_variant in self.qr_variants:
                    df.at[qr_variant, (alpha, dataset_name)] = func(
                        self.results_dict[dataset_name][qr_variant], alpha
                    )

        return df

    def aps(self) -> pd.DataFrame:
        """Aggregate pinball score.

        :return: aggregate pinball score summary
        :rtype: pd.DataFrame
        """
        return self._dataset_variant_summary(
            func=QR_TestResults.aps, caption="Aggregate pinball score"
        )

    def aps_extreme_quantiles(self, n_quantiles: int) -> pd.DataFrame:
        """Aggregate pinball score for n extreme quantiles.

        :param n_quantiles: number of leftmost and rightmost quantiles
        :type n_quantiles: int
        :return: aggregate pinball score computed for extreme quantiles summary
        :rtype: pd.DataFrame
        """
        return self._dataset_variant_summary(
            func=QR_TestResults.aps_extreme_quantiles,
            caption=f"Aggregate pinball score<br>{2*n_quantiles} extreme quantiles",
            n_quantiles=n_quantiles,
        )

    def _dataset_variant_summary(self, func: Callable, caption: str, **kwargs):
        df = pd.DataFrame(index=self.qr_variants, columns=self.dataset_names)

        for dataset_name in self.dataset_names:
            for qr_variant in self.qr_variants:
                df.at[qr_variant, dataset_name] = func(
                    self.results_dict[dataset_name][qr_variant], **kwargs
                )

        style_df = self._get_style_df(df)

        for dataset_name in self.dataset_names:
            for qr_variant in self.qr_variants:
                value = df.at[qr_variant, dataset_name]
                if value == df.loc[:, dataset_name].min():
                    style_df.at[qr_variant, dataset_name] += "green "

        return (
            df.astype(float)
            .style.format(precision=3)
            .set_table_styles(
                [
                    {"selector": ".green", "props": "background-color: #b8fcc3;"},
                ],
                overwrite=True,
            )
            .set_td_classes(style_df)
            .set_caption(caption)
        )

    def _get_style_df(self, df: pd.DataFrame) -> pd.DataFrame:
        style_df = df.copy()
        style_df.loc[:, :] = ""
        return style_df
