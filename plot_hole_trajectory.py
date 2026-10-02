from hole_trajectory import calc_binned_angles, calculate_trajectory_3d, plot_trajectory_3d
from multi_log_reader import MultiLogReader
from preprocess import preprocess

from matplotlib import pyplot as plt
import json

def get_hole_start_end(json_path, site_id, hole_id):

    start = None
    end = None

    # find start and end time for the hole that we want
    with open(json_path, 'r') as json_file:
        hole_metadata = json.load(json_file)

        for hole in hole_metadata:
            if hole["site"] == site_id and hole["hole"] == hole_id:
                start = hole["start"]
                end = hole["end"]
                break

    return start, end

def plot_hole_trajectory(json_path, site_id, hole_id, data_folder):
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
    data_folder = "/lustre/fs25/mdt1/radio/pwindi/position_calibration/drill/drill-data/2024/DataLog"
    json_path = "2024-Drill-Log.json"
    site_id = 14
    hole_id = 1

    plot_hole_trajectory(json_path, site_id, hole_id, data_folder)
