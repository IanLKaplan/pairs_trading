
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
s_and_p_stock_file = 'sp_stocks.csv'


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

def filter_stocks(s_and_p_df: pd.DataFrame) -> None:
    """
    Filter out the higher classes of stock.  If there is a class A, class B and a class C stock, filter out
    class A and B, leaving class C.

    For example, the class A stocks are dropped because they are not commonly traded. Even if they are traded
    they overlap with the lower class (e.g., class B or C).

    Symbol           Name                  Sector
    FOXA  Fox Corp Cl A  communication-services
    GOOGL  Alphabet Cl A  communication-services
    NWSA   News Cp Cl A  communication-services
    :param s_and_p_df:
    :return:
    """
    class_designation = ' Cl '
    names = s_and_p_df['Name']
    company_name_dict = dict()
    for ix, name in enumerate(names):
        str_ix = name.find(class_designation)
        if str_ix > 0:
            company_name = name[:str_ix]
            class_type = name[str_ix+len(class_designation):].strip()
            t = (class_type, ix)
            if company_name not in company_name_dict:
                company_name_dict[company_name] = list()
            company_name_dict[company_name].append(t)
    delete_list = list()
    for company_name, class_tuple_l in company_name_dict.items():
        if len(class_tuple_l) > 1:
            max_tuple = class_tuple_l[0]
            for i in range(1, len(class_tuple_l)):
                cur_tuple = class_tuple_l[i]
                if max_tuple[0] < cur_tuple[0]:
                    max_tuple = cur_tuple
                    delete_list.append((company_name, max_tuple[1]))
                else:
                    delete_list.append((company_name, cur_tuple[1]))
    # Get the row index from the name, row index tuple
    drop_rows = list(list(zip(*delete_list))[1])
    # Remove any of the stock classes that are less than the maximum (e.g., class A is less than class B)
    s_and_p_df.drop(index=s_and_p_df.iloc[drop_rows].index.tolist(), inplace=True)


def process_files(path: str) -> pd.DataFrame:
    s_and_p_df = pd.DataFrame()
    if os.access(path, os.R_OK):
        for file_name in os.listdir(path):
            prefix: str = file_name[0:13]
            if prefix == 'sp-sectors---':
                sector_df = process_file(path, file_name)
                s_and_p_df = pd.concat([s_and_p_df, sector_df], axis=0)
        # serially renumber the DataFrame index
        s_and_p_df.index = range(0, s_and_p_df.shape[0])
        filter_stocks(s_and_p_df)
        s_and_p_df.index = range(0, s_and_p_df.shape[0])
        s_and_p_df.to_csv(path + os.path.sep + s_and_p_stock_file)
    else:
        print(f'Could not read {path}')
    return s_and_p_df


if __name__ == '__main__':
    process_files(s_and_p_directory)
