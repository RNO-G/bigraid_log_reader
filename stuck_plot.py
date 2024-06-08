from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf
from matplotlib.ticker import MultipleLocator

from log_reader import LogReader
from preprocess import preprocess


def plot_timeseries(df, t_start, t_end, sections, ylim, title):
    df = df.copy()

    idx_start = df.index[df.index.searchsorted(t_start)]
    df.index = df.index - idx_start

    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(df['weight_on_bit'], label='Weight on Bit [kg]')
    ax.plot(df['[PLC]CABLESPEED'] * 100, label='Cablespeed [cm/min]')
    ax.plot(df['[PLC]DRILLFEEDBACKVEL'], label='Drill Rotation Speed [rpm]')
    ax.plot(df['[PLC]DRILLACTIVECURRENT'] * 10, label='Drill Current [1/10 A]')
    ax.plot(df['[PLC]WIRESPOOLEDOUT'], label='Drill Position [m]')
    ax.legend(fontsize=14)
    ax.set_title(title)
    ax.set_xlabel("Time [s]", fontsize=14)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(0, t_end - t_start)
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    for i in range(len(sections) - 1):
        ax.axvspan(sections[i], sections[i+1], color='lightgrey', alpha=0.5 + i%2 * 0.5)


    fig.set_size_inches(19, 9)
    fig.tight_layout()

    return fig

if __name__ == "__main__":
    # 2024 stuck drill
    reader = LogReader(tagfile="C:\\Users\\WIPAC\\NERC\\BAS BigRAID - Documents\\Season Reports\\2024\\DataLog\\2024 06 01 0000 BigRAID (Tagname).DAT")
    df = reader.as_df()
    df = preprocess(df)
    df.index = (df.index - df.index[0]).total_seconds()

    plot_timeseries(df, 380, 840, [0, 25, 160, 169, 176, 375, 406, 460], (-50, 300), title="Stuck Drill 2024").savefig('stuck_2024_long.svg')
    plot_timeseries(df, 380, 570, [0, 25, 160, 169, 176, 375, 406, 460], (-50, 300), title="Stuck Drill 2024").savefig("stuck_2024.svg")

    # 2022 stuck drill
    reader = LogReader(tagfile=r"C:\Users\WIPAC\NERC\BAS BigRAID - Documents\Season Reports\Data from control box\logs\Data Log\2022 07 07 0000 (Tagname).DAT")
    df = reader.as_df()
    df = preprocess(df)
    df.index = (df.index - df.index[0]).total_seconds()

    plot_timeseries(df, 16960, 17250, [0, 25, 215, 255, 264, 300], (-50, 380), title="Stuck Drill 2022").savefig("stuck_2022.svg")

    reader = LogReader(tagfile=r"C:\\Users\\WIPAC\\NERC\\BAS BigRAID - Documents\\Season Reports\\2024\\DataLog\\2024 05 30 0000 BigRAID (Tagname).DAT")
    df = reader.as_df()
    df = preprocess(df)
    df.index = (df.index - df.index[0]).total_seconds()

    plot_timeseries(df, 12800, 13800, [], (-50, 380), title="Normal runs").savefig("normal_run.svg")