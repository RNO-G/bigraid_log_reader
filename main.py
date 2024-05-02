from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

from log_reader import LogReader

if __name__ == "__main__":
    logfile = "C:\\Users\\WIPAC\\NERC\\BAS BigRAID - Documents\\Season Reports\\Data from control box\\logs\\Data Log\\2022 06 24 0000 (Tagname).DAT"
    logfile = Path(logfile)

    reader = LogReader(tagfile=logfile)
    df = reader.as_df()

    # Filter the data slightly, since the drill can sometimes record spurious bad values
    df = df.rolling(3).median()
    df = df.dropna()

    # Do some cleanup
    df[df["[PLC]DRILLACTIVECURRENT"].abs() > 1e10] = 0
    df[df["[PLC]WIRESPOOLEDOUT"].abs() > 1e6] = 0
    df[df["[PLC]DRILLFEEDBACKVEL"].abs() > 1e7] = 0

    # Mark drill runs
    df["run"] = 0.0
    df.loc[df["[PLC]WIRESPOOLEDOUT"] > 0.1, "run"] = 1.0
    run_ids = df['run'].ne(df['run'].shift()).cumsum() - 1
    in_run = df["run"] != 0.0
    df.loc[in_run, "run"] = run_ids.loc[in_run] / 2

    # Add a column that indicates if active cutting of new ice was happening. Kind of a guess based on other parameters
    df["cutting"] = 0
    df.loc[(df["[PLC]DRILLACTIVECURRENT"] > 0.5) & (df["run"] > 0) & (df["[PLC]WIRESPOOLEDOUT"] > 1) & (df["[PLC]CABLESPEED"].abs() < 1), "cutting"] = 1

    # Calculate the (running) cut depth drilled per run. For each run, this starts at 0 when drilling begins at
    # the bottom of the hole, and increases until drilling stops
    df["cut_depth"] = df[df["cutting"] == 1]["[PLC]WIRESPOOLEDOUT"].groupby(df["run"]).transform("first")
    df["cut_depth"] =  df["[PLC]WIRESPOOLEDOUT"] - df["cut_depth"]
    df["cut_depth"] = df['cut_depth'].groupby(df["run"]).transform("cummax")
    df["cut_depth"] = df["cut_depth"].fillna(0)

    df[df["cutting"] == 1].plot.scatter(x="cut_depth", y="[PLC]DRILLACTIVECURRENT", c="run")
    plt.title("Motor Current vs Running Cut Depth")

    df[df["cutting"] == 1].plot.scatter(x="[PLC]WIRESPOOLEDOUT", y="[PLC]CABLETENSION", c="run")
    plt.title("Cable Tension vs Depth")

    df[df["cutting"] == 1].plot.scatter(x="run", y="cut_depth")
    plt.title("Running Cut Depth per run")

    df[df["cutting"] == 1].plot.scatter(x="cut_depth", y="[PLC]CABLETENSION", c="run")
    plt.title("Cable Tension vs Running Cut Depth")

    df[df["cutting"] == 1].plot.scatter(x="cut_depth", y="[PLC]CABLETENSION", c="[PLC]CABLESPEED")
    plt.title("Cable Tension vs Running Cut Depth")

    fig = plt.figure()
    ax =  plt.subplot()
    df[df["run"] > 0].index.to_series().groupby(df["run"]).agg(np.ptp).plot(kind="bar", ax=ax, label="Total")
    df[df["cutting"] == 1].index.to_series().groupby(df["run"]).agg(np.ptp).plot(kind="bar", ax=ax, color="orange", label="Active Cutting")
    ax.set_ylabel("Drilling Time [s]")
    ax.legend()
    ax.set_title("Drilling Time per Run")

    df.plot.scatter(y="[PLC]WIRESPOOLEDOUT", x="[PLC]AUTODOWNSTOPDEPTH", c="run")
    plt.title("Stop Depth vs Depth")

    df_plt = df[df["[PLC]WIRESPOOLEDOUT"] == df["[PLC]WIRESPOOLEDOUT"].cummax()]

    print(df.columns)
    # [PLC]WIRESPOOLEDOUT vs [PLC]CABLETENSION
    # angle vs depth
    # autostopdepth vs depth
    # diff of autostopdepth vs time
    # depth vs time
    # feedbackvel vs depth

    # plt.show()
    with matplotlib.backends.backend_pdf.PdfPages(logfile.with_suffix(".pdf").name) as pdf:
        for fig in range(1,  plt.gcf().number + 1):
            pdf.savefig(fig)
