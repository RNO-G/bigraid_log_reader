import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calc_binned_angles(df, key, dz=1.0):
    """
    Calculate mean and stddev of 'key' based values binned by depth.

    :param df: Dataframe containing the drill log data
    :param key: The column to calculate mean and stddev for.
    :param dz: Bin size in meters.
    :return: A tuple of (mean, stddev), where mean and stddev are numpy arrays binned by
             the depth of the drill with bin size dz
    """
    d = df[(df[key].abs() < 2) & (df["[PLC]WIRESPOOLEDOUT"] > 2)]
    # subtract the mean when the drill is hanging freely above the hole
    zero_offset = df[(df["[PLC]WIRESPOOLEDOUT"] < 1) & (df["[PLC]CABLESPEED"].abs() < 0.1) & (df["[PLC]DRILLFEEDBACKVEL"].abs() < 0.1) & (df[key].abs() < 5.0)][key].mean()

    vals = d[key] - zero_offset
    depth_max = df['[PLC]WIRESPOOLEDOUT'].max()

    # Bin data by depth in 1 m steps
    bins = np.arange(0, depth_max, dz)
    binned = vals.groupby(pd.cut(d["[PLC]WIRESPOOLEDOUT"], bins), observed=False)

    # Calculate mean and standard deviation for each bin
    mean = binned.mean().fillna(0).values
    stddev = binned.std().fillna(0).values

    return mean, stddev, binned.size()

def calculate_trajectory_3d(
        mean_angles_x_deg, std_dev_angles_x_deg,
        mean_angles_y_deg, std_dev_angles_y_deg,
        dz=1.0
):
    """
    Calculates a 3D trajectory and its propagated uncertainty from angle pairs.

    The trajectory is assumed to progress along the z-axis at constant dz steps.

    Args:
        mean_angles_x_deg (np.ndarray): Mean angles in the X-Z plane (in degrees).
        std_dev_angles_x_deg (np.ndarray): Std dev of angles in the X-Z plane (in degrees).
        mean_angles_y_deg (np.ndarray): Mean angles in the Y-Z plane (in degrees).
        std_dev_angles_y_deg (np.ndarray): Std dev of angles in the Y-Z plane (in degrees).
        dz (float): The step size along the z-axis (e.g., 1 meter).

    Returns:
        tuple: A tuple containing (z_coords, x_mean, x_std, y_mean, y_std).
    """
    num_steps = len(mean_angles_x_deg)
    z_coords = np.arange(num_steps) * dz

    # --- Convert all angles to radians for calculations ---
    mean_angles_x_rad = np.radians(mean_angles_x_deg)
    std_dev_angles_x_rad = np.radians(std_dev_angles_x_deg)
    mean_angles_y_rad = np.radians(mean_angles_y_deg)
    std_dev_angles_y_rad = np.radians(std_dev_angles_y_deg)

    # We assume the trajectory starts at (0, 0, 0) with zero uncertainty.
    x_mean = np.zeros(num_steps)
    x_variance = np.zeros(num_steps)
    y_mean = np.zeros(num_steps)
    y_variance = np.zeros(num_steps)

    # Integrate and propagate a single step in a single dimension
    def step(x_prev, var_prev, mean_angle, std_angle):
        # Calculate the mean trajectory step
        x_mean = x_prev + dz * np.tan(mean_angle)

        # Calculate variance for the position from the angle variance
        mean_theta_x_n = mean_angle
        var_theta_x_n = std_angle ** 2
        cos_val_x = np.cos(mean_theta_x_n)
        if np.isclose(cos_val_x, 0):
            var_tan_theta_x_n = np.finfo(float).max
        else:
            var_tan_theta_x_n = (1 / cos_val_x**4) * var_theta_x_n

        # Propagate uncertainty
        x_var = var_prev + (dz**2) * var_tan_theta_x_n

        return x_mean, x_var

    # integrate and propagate errors in x and y
    for n in range(1, num_steps):
        x_mean[n], x_variance[n] = step(x_mean[n-1], x_variance[n-1], mean_angles_x_rad[n], std_dev_angles_x_rad[n])
        y_mean[n], y_variance[n] = step(y_mean[n-1], y_variance[n-1], mean_angles_y_rad[n], std_dev_angles_y_rad[n])

    # Calculate standard deviatins
    x_std_dev = np.sqrt(x_variance)
    y_std_dev = np.sqrt(y_variance)

    return z_coords, x_mean, x_std_dev, y_mean, y_std_dev


def calc_inclination(x_mean, x_std, y_mean, y_std):
    x_mean_rad = np.deg2rad(x_mean)
    x_std_rad = np.deg2rad(x_std)
    y_mean_rad = np.deg2rad(y_mean)
    y_std_rad = np.deg2rad(y_std)

    inclination_mean = np.arccos(np.cos(x_mean_rad) * np.cos(y_mean_rad))

    inclination_std = (1/np.tan(inclination_mean)) * np.sqrt(
        np.power(np.tan(x_mean_rad), 2) * np.power(x_std_rad, 2) +
        np.power(np.tan(y_mean_rad), 2) * np.power(y_std_rad, 2)
    )

    return np.rad2deg(inclination_mean), np.rad2deg(inclination_std)


def plot_trajectory_3d(z, x_mean, x_std, y_mean, y_std):
    """ Plot the 3D trajectory and its 2D projections. """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('Hole Trajectory with Uncertainty', fontsize=20)

    x_std3 = x_std * 3
    y_std3 = y_std * 3

    # Main 3D Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(x_mean, y_mean, z, lw=2, label='Mean Trajectory', color='royalblue')

    # Create a grid for the angles of the ellipse and the z-axis
    ellipse_angle = np.linspace(0, 2 * np.pi, 80)
    z_grid, ellipse_angle_grid = np.meshgrid(z, ellipse_angle)

    # Calculate the x and y coordinates of the surface
    X_surface = x_mean[:, np.newaxis] + x_std3[:, np.newaxis] * np.cos(ellipse_angle_grid).T
    Y_surface = y_mean[:, np.newaxis] + y_std3[:, np.newaxis] * np.sin(ellipse_angle_grid).T
    Z_surface = z_grid.T  # Transpose to match the shape of X and Y

    # Plot the surface
    ax1.plot_surface(X_surface, Y_surface, Z_surface,
                     color='cornflowerblue', alpha=0.1, linewidth=0, antialiased=True,
                     label='3σ')

    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.set_zlabel('Z Position (Depth) [m]')
    ax1.set_title('3D Hole Trajectory')
    ax1.legend()
    # Set plot limits. Make the x/y range at least 1 m
    max_range = np.array([x_mean.max()-x_mean.min(), y_mean.max()-y_mean.min(), 1.0]).max() / 2.0
    mid_x = (x_mean.max()+x_mean.min()) / 2.0
    mid_y = (y_mean.max()+y_mean.min()) / 2.0
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(z.min(), z.max())
    ax1.invert_zaxis()


    # 2D Projections with Uncertainty Bands
    # X-Z Plane Projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(z, x_mean, lw=2, label='Mean X', color='firebrick')
    ax2.fill_between(z, x_mean - x_std3, x_mean + x_std3, color='lightcoral', alpha=0.6, label='3σ')
    ax2.set_title('X-Z Plane Projection')
    ax2.set_xlabel('Z Position (Depth) [m]')
    ax2.set_ylabel('X Position [m]')
    ax2.set_ylim(-0.6,0.2)
    ax2.legend()
    ax2.grid(True)

    # Y-Z Plane Projection
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(z, y_mean, lw=2, label='Mean Y', color='forestgreen')
    ax3.fill_between(z, y_mean - y_std3, y_mean + y_std3, color='lightgreen', alpha=0.6, label='3σ')
    ax3.set_title('Y-Z Plane Projection')
    ax3.set_xlabel('Z Position (Depth) [m]')
    ax3.set_ylabel('Y Position [m]')
    ax3.set_ylim(-0.6,0.2)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return (fig, (ax1,ax2,ax3))

def plot_stability_comparison(z_full, x_mean_full, x_std_full, y_mean_full, y_std_full,
                              z_subset, x_mean_subset, y_mean_subset, subtitle):
    """
    Plots the full trajectory against a trajectory calculated from a subset of data.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f'Trajectory Stability Comparison', fontsize=18)
    fig.text(0.5, 0.925, subtitle, horizontalalignment="center")

    ax2 = fig.add_subplot(2, 1, 1)
    ax2.plot(z_full, x_mean_full, lw=2, label='Main Mean X', color='firebrick')
    ax2.fill_between(z_full, x_mean_full - x_std_full, x_mean_full + x_std_full, color='lightcoral', alpha=0.6, label='±1 Std Dev')
    ax2.plot(z_subset, x_mean_subset, lw=2, label='Subset Mean X', color='black', linestyle='--')
    ax2.set_title('X-Z Plane Projection')
    ax2.set_xlabel('Z Position (m)')
    ax2.set_ylabel('X Position (m)')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(z_full, y_mean_full, lw=2, label='Main Mean Y', color='forestgreen')
    ax3.fill_between(z_full, y_mean_full - y_std_full, y_mean_full + y_std_full, color='lightgreen', alpha=0.6, label='±1 Std Dev')
    ax3.plot(z_subset, y_mean_subset, lw=2, label='Subset Mean Y', color='black', linestyle='--')
    ax3.set_title('Y-Z Plane Projection')
    ax3.set_xlabel('Z Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return(ax2,ax3)


def plot_inclination(z_coords, inclination_mean, inclination_err):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 18))
    fig.suptitle(f'Inclination vs Depth', fontsize=18)

    ax = plt.subplot()
    ax.errorbar(inclination_mean, z_coords, xerr=inclination_err, fmt='o', capsize=2, ms=5, label="Inclination")
    ax.set_xlabel("Inclination [deg]")
    ax.set_ylabel('Depth [m]')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return(fig, ax)
