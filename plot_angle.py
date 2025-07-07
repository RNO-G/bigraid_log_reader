import numpy as np

def euler_to_rotation_matrix(yaw, pitch, roll):
    """
    Converts ZYX Euler angles (yaw, pitch, roll) to a rotation matrix.
    Angles are expected in radians.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combine rotations in ZYX order
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def get_xyz_from_euler(yaw, pitch, roll, initial_vector=np.array([1, 0, 0])):
    """
    Calculates the new x, y, z coordinates of a vector after applying
    yaw, pitch, and roll rotations.
    initial_vector: The vector to rotate (e.g., [1, 0, 0] for a forward-pointing vector).
    """
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    rotated_vector = np.dot(R, initial_vector)
    x, y, z = rotated_vector
    return x, y, z

# Example usage (angles in radians)
#yaw_angle = np.radians(45)  # 45 degrees yaw
#pitch_angle = np.radians(30) # 30 degrees pitch
#roll_angle = np.radians(10)  # 10 degrees roll

# Get the direction vector (e.g., representing the forward direction)
#x, y, z = get_xyz_from_euler(yaw_angle, pitch_angle, roll_angle, initial_vector=np.array([1, 0, 0]))
#print(f"Direction vector: x={x:.4f}, y={y:.4f}, z={z:.4f}")

# Get the position of a point initially at [0, 0, 5] relative to the origin
# after applying the rotation
#initial_point = np.array([0, 0, 5])
#x_pos, y_pos, z_pos = get_xyz_from_euler(yaw_angle, pitch_angle, roll_angle, initial_vector=initial_point)
#print(f"Rotated point position: x={x_pos:.4f}, y={y_pos:.4f}, z={z_pos:.4f}")
