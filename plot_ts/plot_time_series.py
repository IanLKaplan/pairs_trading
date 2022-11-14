
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def plot_ts(data_s: pd.Series, title: str, x_label: str, y_label: str) -> None:
    fix, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    data_s.plot()
    ax.axhline(y=0, color='black')
    plt.show()


def plot_two_ts(data_a: pd.DataFrame, data_b: pd.DataFrame, title: str, x_label: str, y_label: str) -> None:
    color_a = 'blue'
    color_b = 'green'
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(data_a, color=color_a, label=data_a.columns[0])
    ax.tick_params(axis='y', labelcolor=color_a)
    ax2 = ax.twinx()
    ax2.plot(data_b, color=color_b, label=data_b.columns[0])
    ax2.tick_params(axis='y', labelcolor=color_b)
    plt.legend(loc='lower right')
    plt.show()
