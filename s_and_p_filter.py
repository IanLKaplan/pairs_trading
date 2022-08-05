from datetime import datetime, timedelta
from tabulate import tabulate
from typing import List, Tuple, TypedDict, Dict
from enum import Enum
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import numpy as np
from pathlib import Path
import tempfile
import os

#
# Read a set of files listing S&P 500 sector stocks.  Filter out duplicates (where a stock is in more than one
# sector. Write out a new CSV file that contains the stock systems, the company name and the sector.
#
s_and_p_directory = '/home/iank/Documents/finance/pairs_trading/s_and_p_sector_components'


class Sector(Enum):
    CONSUMER_DISCRETIONARY = 1
    CONSUMER_STAPLES = 2
    ENERGIES = 3
    FINANCIALS = 4
    HEALTH_CARE = 5
    INDUSTRIALS = 6
    INFO_TECH = 7
    MATERIALS = 8
    REAL_ESTATE = 9
    COMMUNICATION_SVCS = 10
    UTILITIES = 11


enum2str: dict = {Sector.CONSUMER_DISCRETIONARY: 'consumer-discretionary',
                  Sector.CONSUMER_STAPLES: 'consumer-staples',
                  Sector.ENERGIES: 'energies',
                  Sector.FINANCIALS: 'financials',
                  Sector.HEALTH_CARE: 'health-care',
                  Sector.INDUSTRIALS: 'industrials',
                  Sector.INFO_TECH: 'information-technology',
                  Sector.MATERIALS: 'materials',
                  Sector.REAL_ESTATE: 'real-estate',
                  Sector.COMMUNICATION_SVCS: 'communication-services',
                  Sector.UTILITIES: 'utilities'}

str2enum: dict = {'consumer-discretionary': Sector.CONSUMER_DISCRETIONARY,
                  'consumer-staples': Sector.CONSUMER_STAPLES,
                  'energies': Sector.ENERGIES,
                  'financials': Sector.FINANCIALS,
                  'health-care': Sector.HEALTH_CARE,
                  'industrials': Sector.INDUSTRIALS,
                  'information-technology': Sector.INFO_TECH,
                  'materials': Sector.MATERIALS,
                  'real-estate': Sector.REAL_ESTATE,
                  'communication-services': Sector.COMMUNICATION_SVCS,
                  'utilities': Sector.UTILITIES}


def enum_to_str(enum_val: Sector):
    return enum2str[enum_val]


def str_to_enum(name: str):
    return str2enum[name]


class StockInfo(TypedDict):
    symbol: str
    company_name: str
    sector: Sector


def get_sector_name(file_name: str) -> str:
    """
    Return the sector name for the S&P 500 sector
    :param file_name: a file name with a format like sp-sectors---information-technology-08-03-2022.csv
    :return: the sector. In this example 'information-technology'
    """
    s: str = ''
    if file_name is not None and len(file_name) > 0:
        s: str = file_name[13:-15]
    return s


def process_file(path: str, file_name: str) -> pd.DataFrame:
    """
    Read a barchart sector file that lists the stocks in an S&P 500 sector.
    The columns are:

    Symbol,Name,Last,Change,%Chg,Open,High,Low,Volume,Time

    :param path: the path to the directory containing the file
    :param file_name: the file name
    :param symbols: a dictionary that will be populated with the data
    :return: Nothing
    """
    sec_name: str = get_sector_name(file_name)
    file_path = path + os.path.sep + file_name
    sector_raw = pd.read_csv(file_path)
    # The last row of the data downloaded from barchart contains the line
    # "Downloaded from Barchart.com as of 08-02-2022 10:54am CDT"  Remove this line
    # if it exists
    last_line = sector_raw.iloc[-1,:][0]
    if type(last_line) == str:
        if sector_raw.iloc[-1,:][0].startswith('Downloaded'):
            sector_raw = sector_raw.head(sector_raw.shape[0] - 1)
    else:
        pass
    sector_df = sector_raw[['Symbol','Name']].copy()
    sec_enum = str_to_enum(sec_name)
    l = [sec_enum] * sector_df.shape[0]
    sector_df['Sector'] = l
    return sector_df


def process_files(path: str) -> pd.DataFrame:
    s_and_p_df = pd.DataFrame()
    if os.access(path, os.R_OK):
        for file_name in os.listdir(path):
            sector_df = process_file(path, file_name)
            s_and_p_df = pd.concat([s_and_p_df, sector_df], axis=0)
        s_and_p_df.to_csv(path + os.path.sep + 'foo')
    else:
        print(f'Could not read {path}')
    return s_and_p_df


if __name__ == '__main__':
    process_files(s_and_p_directory)
