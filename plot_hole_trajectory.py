from hole_trajectory import calc_binned_angles, calculate_trajectory_3d, plot_trajectory_3d
from multi_log_reader import MultiLogReader
from preprocess import preprocess

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import json

def get_hole_start_end(json_path, site_id, hole_id):

    start = None
    end = None

    # find start and end time for the hole that we want
    with open(json_path, 'r') as json_file:
        hole_metadata = json.load(json_file)

        for hole in hole_metadata:
            if int(hole["site"]) == site_id and int(hole["hole"]) == hole_id:
                start = hole["start"]
                end = hole["end"]
                break

    return start, end

def calculate_off_vertical_angle(xvals, yvals, zvals):
    delta_x = xvals[1:] - xvals[:-1]
    delta_y = yvals[1:] - yvals[:-1]
    delta_z = zvals[1:] - zvals[:-1]
    delta_r = np.linalg.norm([delta_x, delta_y], ord = 2, axis = 0)
    print(delta_r)
    off_vertical_deg = np.rad2deg(np.arctan2(delta_r, delta_z))
    return zvals[:-1], off_vertical_deg

def plot_hole_trajectory(json_path, site_id, hole_id, data_folder, export_data):
    start, end = get_hole_start_end(json_path, site_id, hole_id)

    print(f"start: {start}, end: {end}")
    df = MultiLogReader.find_files(data_folder, start, end).as_df() 
    df = preprocess(df)

    df_up = df[(df['[PLC]WIRESPOOLEDOUT'].diff() < 0) | (df["[PLC]CABLESPEED"].abs() < 0.1) & (df['cutting']==0)].dropna()
    dz = 1.0
    mean_angles_x, std_dev_angles_x, angle_bin_sizes = calc_binned_angles(df_up, "hole_pitch", dz=dz)
    mean_angles_y, std_dev_angles_y, _= calc_binned_angles(df_up, "hole_roll", dz=dz)
    
    z_coords, x_mean, x_std, y_mean, y_std = calculate_trajectory_3d(
        mean_angles_x, std_dev_angles_x,
        mean_angles_y, std_dev_angles_y,
        dz=dz
    )

    # Calculate and plot the off-vertical angle as a function of depth
    z_coords_angle, off_vertical_deg = calculate_off_vertical_angle(x_mean, y_mean, z_coords)
    fig = plt.figure(figsize = (6, 2.1), layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0])
    ax.plot(z_coords_angle, off_vertical_deg)
    fig.savefig(f"site_{site_id}_hole_{hole_id}_off_vertical.pdf")
    plt.close()

    if export_data:
        np.savetxt(f"site_{site_id}_hole_{hole_id}_off_vertical.txt", 
                   np.transpose([z_coords_angle, off_vertical_deg]))

    # Make the usual 3d hole-trajectory plot
    fig, ax = plot_trajectory_3d(z_coords, x_mean, x_std, y_mean, y_std)
    for ia,a in enumerate(ax):
        print(ia,a)
        a.tick_params(axis='both', labelsize=16)
        a.set_ylabel(a.get_ylabel(), fontsize=16)
        a.set_ylim(-0.5,0.2)
        a.set_xlabel(a.get_xlabel(), fontsize=16)
        if (ia==0):
            a.set_zlabel(a.get_zlabel(), fontsize=16)
        a.set_title(f"Site {site_id}, hole {hole_id}", fontsize=16)

    plt.savefig(f"site_{site_id}_hole_{hole_id}.pdf")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", dest = "data_folder")
    parser.add_argument("--drill-json", dest = "json_path", default = "2024-Drill-Log.json")
    parser.add_argument("--site", dest = "site_id", type = int)
    parser.add_argument("--hole", dest = "hole_id", type = int)
    parser.add_argument("--export-data", dest = "export_data", action = "store_true", default = False)
    args = vars(parser.parse_args())

    plot_hole_trajectory(**args)
