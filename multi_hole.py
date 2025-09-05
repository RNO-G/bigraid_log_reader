import json
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.style as mplstyle
#mplstyle.use('tableau-colorblind10')
#mplstyle.use('seaborn-colorblind')
from matplotlib.dates import DateFormatter

mplstyle.use('fast')
plt.rcParams['lines.markersize'] = 1

from multi_log_reader import MultiLogReader
from preprocess import preprocess


def load_cached_df(r_folder, r_hole_times):
    try:
        with Path("_cached_df.pkl").open("rb") as f:
            folder, hole_times, df = pickle.load(f)
            if folder == r_folder and hole_times == r_hole_times:
                return df
            return None
    except Exception:
        return None


def write_cached_df(folder, hole_times, df):
    with Path("_cached_df.pkl").open("wb") as f:
        x = (folder, hole_times, df)
        pickle.dump(x, f)


@click.command
@click.option("--folder", help="Folder containing BigRAID PLC logs",
              required=True)
@click.option("--hole_times", "hole_times",
              help="A json file containing information about the start and stop time "
                   "for each hole", required=True)
@click.option("--out", "out_file",
              default=None,
              help="Output path for the PDF. Defaults to a name based on the dates "
                   "in the current folder")
@click.option("--no-cache", "no_cache", is_flag=True,
              default=False,
              help="Don't use the cached dataframe. This will force the generation of "
                   "the dataframe from the log files, even if the parameters match a "
                   "previous invocation.")
def plot(folder, hole_times, out_file, no_cache):
    if out_file is None:
        out_file = Path(f"BigRAID.pdf")
    out_file = Path(out_file)

    with Path(hole_times).open("r") as f:
        hole_time_data = json.load(f)

    # Load a cached dataframe if it exists and matches the other parameters
    if not no_cache:
        cached_df = load_cached_df(folder, hole_times)
        if cached_df is not None:
            return _plot(cached_df, out_file)

    dataframes = []
    for hole_data in hole_time_data:
        try:
            number, site, hole, t_start, t_end, geoloc = (
                hole_data['number'],
                hole_data['site'], hole_data['hole'],
                hole_data['start'], hole_data.get('end'),
                hole_data.get('geoloc', "")
            )
        except KeyError as e:
            print(f"Error. Malformed hole times json. Missing: {e} in {hole_data}")
            continue

        print(f"Reading data for hole {number}-{site}-{hole}")
        cached_df = MultiLogReader.find_files(folder, t_start, t_end).as_df()
        cached_df = preprocess(cached_df)
        cached_df['number'] = number
        cached_df['site'] = site
        cached_df['hole'] = hole
        cached_df['geoloc'] = geoloc

        cached_df['hole_duration'] = np.arange(0, cached_df.shape[0])

        dataframes.append(cached_df)

    total_df = pd.concat(dataframes)

    # Write the df to the cache so we can load it faster next time
    write_cached_df(folder, hole_times, total_df)

    return _plot(total_df, out_file)


def _plot(df, out_path):
    hole_groups = df.groupby(['number','site', 'hole'])

    fig = plt.figure()
    ax = fig.subplots()

    # Generate colors from the colormap
    n = 12
    cmap = plt.get_cmap('viridis', n)
    i =0 
    for (number, site, hole), idx in hole_groups.groups.items():
        print(number, site, hole, i, cmap(i))
        data = df.loc[idx, ['[PLC]WIRESPOOLEDOUT', 'hole_duration']].cummax()
        # Choose 100 evenly spaced points and subsample the data
        subset_idx = np.round(np.linspace(0, data.shape[0]-1, 100)).astype(int)
        subsample = data.iloc[subset_idx]

        ax.scatter(
            pd.to_datetime(subsample['hole_duration'], unit='s'),
            subsample['[PLC]WIRESPOOLEDOUT'],
            s=3,
            label=f"Site {site} Hole {hole}",
            c = [cmap(i)]
        )
        i = i+1

    formatter = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Time from start [hh:mm]")
    ax.set_ylabel("Depth [m]")

    ax.legend()

    #plt.show()
    with matplotlib.backends.backend_pdf.PdfPages(out_path) as pdf:
        for fig in range(1,  plt.gcf().number + 1):
            pdf.savefig(fig)


if __name__ == "__main__":
    plot()
