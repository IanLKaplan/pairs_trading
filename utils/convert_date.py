
import numpy as np
from datetime import datetime


def convert_date(some_date) -> datetime:
    if type(some_date) == str:
        some_date = datetime.fromisoformat(some_date)
    elif type(some_date) == np.datetime64:
        ts = (some_date - np.datetime64('1970-01-01T00:00')) / np.timedelta64(1, 's')
        some_date = datetime.utcfromtimestamp(ts)
    # round datatime to year, month, day
    some_date = datetime(*some_date.timetuple()[:3])
    return some_date
