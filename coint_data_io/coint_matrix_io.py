import os
from typing import Tuple

import pandas as pd
import numpy as np

from enum import Enum

# Local libraries
from coint_analysis.coint_analysis_result import CointAnalysisResult, CointInfo


class CointMatrixIO:
    class CointType(Enum):
        GRANGER = 1
        JOHANSEN = 2

    def __init__(self):
        self.cointegration_data_dir = 'cointegration_data'
        self.correlation_file_name = 'correlation.csv'
        self.granger_file_name = 'granger.csv'
        self.johansen_file_name = 'johansen.csv'
        self.correlation_file_path = self.cointegration_data_dir + os.path.sep + self.correlation_file_name
        self.granger_file_path = self.cointegration_data_dir + os.path.sep + self.granger_file_name
        self.johansen_file_path = self.cointegration_data_dir + os.path.sep + self.johansen_file_name

    def write_correlation_matrix(self, coint_analysis: pd.DataFrame) -> None:
        """
        Write out the pairs correlation DataFrame. The structure of the DataFrame is a set of
        columns with the pair names (e.g., 'AAPL:MPWR') and an index for the date that
        starts the time period. This code builds a new DaataFrame that does not include
        the cointegration data.

        :param coint_analysis: a DataFrame with the correlation and cointegeration data.
        :return: Nothing.
        """
        correlation_a = np.zeros(coint_analysis.shape)
        num_rows = coint_analysis.shape[0]
        num_columns = coint_analysis.shape[1]
        for row_ix in range(num_rows):
            for col_ix in range(num_columns):
                correlation_a[row_ix, col_ix] = coint_analysis.iloc[row_ix, col_ix][0]
        correlation_df = pd.DataFrame(correlation_a)
        correlation_df.columns = coint_analysis.columns
        correlation_df.index = coint_analysis.index
        correlation_df.to_csv(self.correlation_file_path, index_label='Date')

    def build_cointegeration_matrx(self, coint_analysis: pd.DataFrame, coint_type: CointType) -> pd.DataFrame:
        """
        Build a DataFrame that contains either the Granger or the Johansen cointegration data.

        The coint_analysis DataFrame contains elements that include the correlation value and objects with the
        Granger and Johansen data. This function builds a new DataFrame with the cointegration data.

        The resulting DataFrame includes row and column numbers which serve as foreign keys that can be used
        to reconstruct the original DataFrame.

        :param coint_analysis:
        :param coint_type:
        :return:
        """
        row_list = list()
        num_rows = coint_analysis.shape[0]
        num_columns = coint_analysis.shape[1]
        for row_ix in range(num_rows):
            for col_ix in range(num_columns):
                if coint_analysis.iloc[row_ix, col_ix][1] is not None:
                    obj: CointAnalysisResult = coint_analysis.iloc[row_ix, col_ix][1]
                    if coint_type == self.CointType.JOHANSEN:
                        coint_obj: CointInfo = obj.johansen_coint
                    else:
                        coint_obj: CointInfo = obj.granger_coint
                    row_tuple = (row_ix, col_ix, coint_obj.confidence, coint_obj.pair_str, coint_obj.weight, coint_obj.has_intercept, coint_obj.intercept)
                    row_list.append(row_tuple)
        coint_info_df = pd.DataFrame(row_list)
        columns = ['row_ix', 'col_ix', 'confidence', 'pair_str', 'weight', 'has_intercept', 'intercept']
        coint_info_df.columns = columns
        return coint_info_df

    def write_cointegration_matrix(self, coint_analysis: pd.DataFrame) -> None:
        granger_coint_df = self.build_cointegeration_matrx(coint_analysis, self.CointType.GRANGER)
        johansen_coint_df = self.build_cointegeration_matrx(coint_analysis, self.CointType.JOHANSEN)
        granger_coint_df.to_csv(self.granger_file_path, index=False)
        johansen_coint_df.to_csv(self.johansen_file_path, index=False)

    def write_files(self, coint_analysis: pd.DataFrame) -> None:
        self.write_correlation_matrix(coint_analysis)
        self.write_cointegration_matrix(coint_analysis)

    def has_files(self) -> bool:
        files_exist = False
        if os.access(self.cointegration_data_dir, os.R_OK):
            files_exist = os.access(self.correlation_file_path, os.R_OK) and \
                          os.access(self.johansen_file_path, os.R_OK) and \
                          os.access(self.granger_file_path, os.R_OK)
        return files_exist

    def build_coint_info(self, coint_row: pd.DataFrame) -> CointInfo:
        # columns: row_ix, col_ix, confidence, pair_str, weight, has_intercept, intercept
        pair_str = coint_row['pair_str']
        confidence = coint_row['confidence']
        weight = coint_row['weight']
        has_intercept = coint_row['has_intercept']
        intercept = coint_row['intercept']
        info = CointInfo(pair_str=pair_str,
                         confidence=confidence,
                         weight=weight,
                         has_intercept=has_intercept,
                         intercept=intercept)
        return info

    def add_coint_info(self, coint_info_a: np.array, coint_type: CointType) -> None:
        if coint_type == self.CointType.GRANGER:
            coint_data = pd.read_csv(self.granger_file_path)
        else:
            coint_data = pd.read_csv(self.johansen_file_path)
        assert coint_data.shape[0] == coint_info_a.shape[0] * coint_info_a.shape[1]
        for _, row in coint_data.iterrows():
            # columns: row_ix, col_ix, confidence, pair_str, weight, has_intercept, intercept
            coint_info = self.build_coint_info(row)
            row_ix = row['row_ix']
            col_ix = row['col_ix']
            if coint_info_a[row_ix, col_ix] == 0:
                coint_info_a[row_ix, col_ix] = (0.0, CointAnalysisResult())
            elem: Tuple = coint_info_a[row_ix, col_ix]
            coint: CointAnalysisResult = elem[1]
            if coint_type == self.CointType.GRANGER:
                coint.granger_coint = coint_info
            else:
                coint.johansen_coint = coint_info


    def read_files(self) -> pd.DataFrame:
        """
        Cointegeration DataFrames:

        row, column, confidence, pair_str, weight, has_intercept, intercept

        :return:
        """
        correlation_df = pd.read_csv(self.correlation_file_path, index_col='Date')
        coint_info_a = np.zeros(correlation_df.shape, dtype='O')
        self.add_coint_info(coint_info_a, self.CointType.GRANGER)
        self.add_coint_info(coint_info_a, self.CointType.JOHANSEN)
        for row_ix in range(correlation_df.shape[0]):
            for col_ix in range(correlation_df.shape[1]):
                corr_val = correlation_df.iloc[row_ix, col_ix]
                elem: Tuple = coint_info_a[row_ix, col_ix]
                elem[0] = corr_val
        coint_info_df = pd.DataFrame(coint_info_a)
        coint_info_df.columns = correlation_df.columns
        coint_info_df.index = correlation_df.index
        return coint_info_df
