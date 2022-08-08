
import os
import pandas as pd

#
# Build a single file that contains the S&P 500 stocks and sectors. This file is the base input to the pairs
# trading notebook.
#
# To build the file, read a set of files listing the S&P 500 sector stocks.  These files were downloaded over a
# period of three days from barchart.com.  Write out a new CSV file that contains the stock symbol, the company name
# and the sector.
#
s_and_p_directory = 's_and_p_sector_components'
output_file_name = 'sp_stocks.csv'


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
    :return: Nothing
    """
    sec_name: str = get_sector_name(file_name)
    file_path = path + os.path.sep + file_name
    sector_raw = pd.read_csv(file_path)
    # The last row of the data downloaded from barchart contains the line
    # "Downloaded from Barchart.com as of 08-02-2022 10:54am CDT"  Remove this line
    # if it exists
    last_line = sector_raw.iloc[-1, :][0]
    if type(last_line) == str:
        if sector_raw.iloc[-1, :][0].startswith('Downloaded'):
            sector_raw = sector_raw.head(sector_raw.shape[0] - 1)
    sector_df = sector_raw[['Symbol', 'Name']].copy()
    sector_df['Sector'] = sec_name
    return sector_df


def process_files(path: str) -> pd.DataFrame:
    s_and_p_df = pd.DataFrame()
    if os.access(path, os.R_OK):
        for file_name in os.listdir(path):
            prefix: str = file_name[0:13]
            if prefix == 'sp-sectors---':
                sector_df = process_file(path, file_name)
                s_and_p_df = pd.concat([s_and_p_df, sector_df], axis=0)
        s_and_p_df.to_csv(path + os.path.sep + output_file_name)
    else:
        print(f'Could not read {path}')
    return s_and_p_df


if __name__ == '__main__':
    process_files(s_and_p_directory)
