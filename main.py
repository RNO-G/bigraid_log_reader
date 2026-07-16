# package for parsing cli arguments
import argparse

# packages for folder selection UI
import tkinter as tk
from tkinter import filedialog

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf

# Other packages
from log_reader import LogReader
from preprocess import preprocess


if __name__ == "__main__":

    # Create the cli parser
    parser = argparse.ArgumentParser(description="Process some data logs.")

    # Define the named arguments
    parser.add_argument(
        "--dataLogDir", 
        type = str, 
        help = "Path to the data log directory"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check if the dataLogDir was supplied
    if args.dataLogDir:
        tagNameDir = args.dataLogDir
    else:
        # Create a root window and immediately hide it
        root = tk.Tk()
        root.withdraw()

        # Open the folder selection dialog
        tagNameDir = filedialog.askdirectory(title="Select a DataLog Folder")
    
    # Find all the Tagname files
    tagNameFiles = glob.glob(str(Path(tagNameDir).joinpath("*(Tagname).DAT")))

    # Process each Tagname file
    for tagNameFile in tagNameFiles:

        # Current Tagname file
        print("Processing:\t" + tagNameFile)

        try:

            # Read the tag file
            reader = LogReader(tagfile=tagNameFile)
            # Read associated float file and conver data to a Pandas DataFrame
            df = reader.as_df()

            # Add additional calculated columns to the data
            df = preprocess(df)

            # Generate some basic plots
            ax = df[df["cutting"] == 1].plot.scatter(
                x="cut_depth", y="[PLC]DRILLACTIVECURRENT", c="run",
                title="Motor Current vs Cut Length",
                xlabel="Cut length [m]",
                ylabel="Drill active current [A]")
            ax.collections[0].colorbar.set_label('Run #')

            ax = df[df["cutting"] == 1].plot.scatter(
                x="[PLC]WIRESPOOLEDOUT", y="[PLC]CABLETENSION", c="run",
                title="Cable Tension vs Cuter Depth",
                xlabel="Cutter depth [m]",
                ylabel="Cable tension [kg]")
            ax.collections[0].colorbar.set_label('Run #')

            ax = df[df["cutting"] == 1].plot.scatter(
                x="run", y="cut_depth",
                title="Cut Lengths",
                xlabel="Run #",
                ylabel="Cut length [m]")

            ax = df[df["cutting"] == 1].plot.scatter(
                x="cut_depth", y="[PLC]CABLETENSION", c="run",
                title="Cable Tension vs Cut Length",
                xlabel="Cut length [m]",
                ylabel="Cable tension [kg]")
            ax.collections[0].colorbar.set_label('Run #')

            ax = df[df["cutting"] == 1].plot.scatter(
                x="cut_depth", y="[PLC]CABLETENSION", c="[PLC]CABLESPEED",
                title="Cable Tension vs Cut Length",
                xlabel="Cut length [m]",
                ylabel="Cable tension [kg]")
            ax.collections[0].colorbar.set_label('Cable speed [m/min]')

            ax = df[(df["cutting"] == 1)].plot.scatter(
                x="cut_depth", y="weight_on_bit", c="run",
                title="Estimated Weight on Bit vs Cut Length",
                xlabel="Cut lenth [m]",
                ylabel="Weight on bit - estimated [kg]")
            ax.collections[0].colorbar.set_label('Run #')

            def group_duration(x):
                diff = x.diff()
                return diff[diff < np.timedelta64(10, 's')].sum()

            total = df[df["run"] > 0].index.to_series().groupby(df["run"]).agg(np.ptp).dt.total_seconds().rename("total")
            cutting = df[df["cutting"] == 1].index.to_series().groupby(df["run"]).agg(np.ptp).dt.total_seconds().rename("cutting")
            moving = df[df["[PLC]CABLESPEED"] > 2].index.to_series().groupby(df["run"]).agg(group_duration).dt.total_seconds().rename("moving")
            ejecting = df[df["[PLC]DRILLFEEDBACKVEL"] < -5].index.to_series().groupby(df["run"]).agg(np.ptp).dt.total_seconds().rename("ejecting")

            if "[PLC]ICECONVEYOR.SETSPEED" in df.columns:
                snowblower_travel = df[df['[PLC]ICECONVEYOR.SETSPEED'].abs() > 1].index.to_series().groupby(df["run"]).agg(group_duration).dt.total_seconds().rename("snowblower_travel")
            else:
                snowblower_travel = pd.Series(np.zeros(cutting.shape)).rename("snowblower_travel")
            grps = pd.concat([total, moving, cutting, ejecting, snowblower_travel], axis=1).fillna(0.0)

            fig = plt.figure()
            ax = plt.subplot()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            ax.bar(grps.index, grps['total'], color=colors[0], label="Total")
            ax.bar(grps.index, grps['cutting'], color=colors[1], label="Cutting")
            ax.bar(grps.index, grps['snowblower_travel'], color=colors[2], label="Snowblower Travel", bottom=grps['cutting'])
            ax.bar(grps.index, grps['ejecting'], color=colors[3], label="Ejecting", bottom=grps['cutting'] + grps['snowblower_travel'])
            ax.bar(grps.index, grps['moving'], color=colors[4], label="Moving", bottom=grps['cutting'] + grps['snowblower_travel'] + grps["ejecting"])
            ax.legend()
            ax.set_title("Time per Run")
            ax.set_ylabel("Time [s]")
            ax.set_xlabel("Run")

            fig, axes = plt.subplots(1, 2)
            display_max = 0
            for i, k in enumerate(["[PLC]IMUPITCH", "[PLC]IMUROLL"]):
                d = df[(df[k].abs() < 2) & (df["[PLC]WIRESPOOLEDOUT"] > 2)]
                # subtract the mean when the drill is hanging freely above the hole
                zero_offset = df[(df["[PLC]WIRESPOOLEDOUT"] < 1) & (df["[PLC]CABLESPEED"].abs() < 0.1) & (df["[PLC]DRILLFEEDBACKVEL"].abs() < 0.1)][k].mean()

                vals = d[k] - zero_offset
                display_max = max(vals.abs().quantile(0.99), display_max)
                depth_max = df['[PLC]WIRESPOOLEDOUT'].max()
                H, xedges, yedges = np.histogram2d(d["[PLC]WIRESPOOLEDOUT"], d[k] - zero_offset, bins=[int(depth_max/2), 50], range=[[0, depth_max], [-2, 2]])
                H_norm_rows = H / H.max(axis=1, keepdims=True)
                H_norm_rows = np.nan_to_num(H_norm_rows)
                axes[i].pcolormesh(yedges, xedges, H_norm_rows)
                axes[i].set_title(k)
            for ax in axes:
                ax.set_xlim(-display_max, display_max)
            fig.suptitle("Angle per Depth, normalized using IMU Yaw")
            fig.supylabel("Wire spooled out [m]")
            fig.supxlabel("Angle [deg]")

            fig, axes = plt.subplots(1, 2)
            display_max = 0
            for i, k in enumerate(["hole_pitch", "hole_roll"]):
                d = df[(df[k].abs() < 2) & (df["[PLC]WIRESPOOLEDOUT"] > 2)]
                # subtract the mean when the drill is hanging freely above the hole
                zero_offset = df[(df["[PLC]WIRESPOOLEDOUT"] < 1) & (df["[PLC]CABLESPEED"].abs() < 0.1) & (df["[PLC]DRILLFEEDBACKVEL"].abs() < 0.1)][k].mean()

                vals = d[k] - zero_offset
                display_max = max(vals.abs().quantile(0.99), display_max)
                depth_max = df['[PLC]WIRESPOOLEDOUT'].max()
                H, xedges, yedges = np.histogram2d(d["[PLC]WIRESPOOLEDOUT"], d[k] - zero_offset, bins=[int(depth_max/2), 50], range=[[0, depth_max], [-2, 2]])
                H_norm_rows = H / H.max(axis=1, keepdims=True)
                H_norm_rows = np.nan_to_num(H_norm_rows)
                axes[i].pcolormesh(yedges, xedges, H_norm_rows)
                axes[i].set_title(k)
            for ax in axes:
                ax.set_xlim(-display_max, display_max)
            fig.suptitle("Angle per Depth, normalized using IMU Yaw")
            fig.supylabel("Wire spooled out [m]")
            fig.supxlabel("Angle [deg]")

            df.plot.scatter(y="[PLC]WIRESPOOLEDOUT", x="[PLC]AUTODOWNSTOPDEPTH", c="run")
            plt.title("Stop Depth vs Depth")

            df_plt = df[df["[PLC]WIRESPOOLEDOUT"] == df["[PLC]WIRESPOOLEDOUT"].cummax()]

            df[["run", "cutting", "cut_depth", "[PLC]WIRESPOOLEDOUT"]].plot()

            print(df.columns)

            # [PLC]WIRESPOOLEDOUT vs [PLC]CABLETENSION
            # angle vs depth
            # autostopdepth vs depth
            # diff of autostopdepth vs time
            # depth vs time
            # feedbackvel vs depth

            # plt.show()
            with matplotlib.backends.backend_pdf.PdfPages(Path(tagNameFile).with_suffix(".pdf")) as pdf:
                for fig in range(1,  plt.gcf().number + 1):
                    pdf.savefig(fig)

            print("PDF saved: "+ str(Path(tagNameFile).with_suffix(".pdf")) + "\r")

            # save df (dataframe) as csv
            df.to_csv(str(Path(tagNameFile).with_suffix(".csv")), index=True)

        except Exception as e:
            print(f'caught {type(e)}: {e}')
            print(Path(tagNameFile).name + " cannot be parsed correctly.")
        finally: 
            plt.close('all')

