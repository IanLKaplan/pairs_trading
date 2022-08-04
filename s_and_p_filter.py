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
    ENERGY = 3
    FINANCIALS = 4
    HEALTH_CARE = 5
    INDUSTRIALS = 6
    INFO_TECH = 7
    MATERIALS = 8
    REAL_ESTATE = 9
    CONSUMER_SVCS = 10
    UTILITIES = 11


enum2str: dict = {Sector.CONSUMER_DISCRETIONARY: 'consumer-discretionary',
                  Sector.CONSUMER_STAPLES: 'consumer-staples',
                  Sector.ENERGY: 'energy',
                  Sector.FINANCIALS: 'financials',
                  Sector.HEALTH_CARE: 'health-care',
                  Sector.INDUSTRIALS: 'industrials',
                  Sector.INFO_TECH: 'information-technology',
                  Sector.MATERIALS: 'materials',
                  Sector.REAL_ESTATE: 'real-estate',
                  Sector.CONSUMER_SVCS: 'consumer-services',
                  Sector.UTILITIES: 'utilities'}

str2enum: dict = {'consumer-discretionary': Sector.CONSUMER_DISCRETIONARY,
                  'consumer-staples': Sector.CONSUMER_STAPLES,
                  'energy': Sector.ENERGY,
                  'financials': Sector.FINANCIALS,
                  'health-care': Sector.HEALTH_CARE,
                  'industrials': Sector.INDUSTRIALS,
                  'information-technology': Sector.INFO_TECH,
                  'materials': Sector.MATERIALS,
                  'real-estate': Sector.REAL_ESTATE,
                  'consumer-services': Sector.CONSUMER_SVCS,
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
    '''
    Return the sector name for the S&P 500 sector
    :param file_name: a file name with a format like sp-sectors---information-technology-08-03-2022.csv
    :return: the sector. In this example 'information-technology'
    '''
    pass


def process_file(file_name: str, symbols: Dict) -> None:
    sec_name: str = get_sector_name(file_name)


def process_files(path: str) -> Dict:
    symbols: Dict = dict()
    if os.access(path, os.R_OK):
        for file_name in os.listdir(path):
            process_file(file_name)
    else:
        print(f'Could not read {path}');
    return symbols


if __name__ == '__main__':
    process_files(s_and_p_directory)
