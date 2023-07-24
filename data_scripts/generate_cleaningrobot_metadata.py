import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import re
from scipy.spatial.transform import Slerp

data_path = "/home/ningj/data/SIMPLE_RECON/cleaning_robot/2023-7-24-11-57"

# Load LDS pose data
with open(data_path + '/dataPose.txt', 'r') as f:
    lines = f.readlines()

# Initialize camera pose array
camera_pose = []

# Define rotation matrix to convert from LDS to camera coordinates
rotation_matrix = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])

# Define translation from LDS origin to camera origin
translation = np.array([0, -0.06, 0.169])

# Convert LDS pose to camera pose
for i, line in enumerate(lines):
    # Split line into components
    components = line.strip().split(',')

    # Check that line has the correct number of components
    if len(components) < 5:
        continue

    # Extract x, y, and theta values and convert to floats
    x, y, theta = float(components[2]), float(components[3]), float(components[4])

    # Create rotation matrix for theta
    theta_rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])

    # Convert LDS pose to camera pose
    camera_pose_xyz = np.dot(rotation_matrix, np.dot(theta_rotation_matrix, np.array([x, y, 0])) + translation)

    # Convert rotation matrix to quaternion
    camera_rotation = R.from_matrix(np.dot(rotation_matrix, theta_rotation_matrix))
    camera_quaternion = camera_rotation.as_quat()

    # Append image id and camera pose to camera_pose list
    camera_pose.append([components[1]] + list(camera_quaternion) + list(camera_pose_xyz))

camera_pose_data = np.array(camera_pose)

# Extract timestamps and poses
timestamps = camera_pose_data[:, 0].astype(int)
poses = camera_pose_data[:, 1:].astype(float)

# Function to find the nearest timestamp
def find_nearest_timestamp(target_timestamp, timestamps):
    return timestamps[np.abs(timestamps - target_timestamp).argmin()]

# Directory containing image files
image_directory = data_path + '/aiimgs'

# Get a sorted list of image files based on timestamp
image_files = os.listdir(image_directory)
image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

# List to store interpolated poses
interpolated_poses = []
cam_timestamps = []

# Process each image
for image_name in image_files:
    if image_name.endswith('.jpg'):
        # Extract timestamp from image name
        image_timestamp = int(re.search(r'\d+', image_name.split('-')[1]).group())
        # Find the nearest timestamp in the camera pose data
        nearest_timestamp = find_nearest_timestamp(image_timestamp, timestamps)

        # Find the index of the nearest timestamp
        nearest_index = np.where(timestamps == nearest_timestamp)[0][0]
        if nearest_timestamp > image_timestamp:
            nearest_index = nearest_index - 1
            nearest_timestamp = timestamps[nearest_index]

        # Extract the corresponding pose
        nearest_pose = poses[nearest_index]

        # Interpolate the pose if there is an exact match
        if nearest_timestamp == image_timestamp:
            interpolated_poses.append(nearest_pose)
            cam_timestamps.append(int(image_timestamp))

        else:
            # Perform interpolation
            next_index = nearest_index + 1
                
            if next_index < len(timestamps):
                next_pose = poses[next_index]
                next_timestamp = timestamps[next_index]
                t_diff = next_timestamp - nearest_timestamp
                weight = (image_timestamp - nearest_timestamp) / t_diff
                nearest_quat = nearest_pose[:4]
                next_quat = next_pose[:4]
                rotations_array = R.from_quat([nearest_quat, next_quat])
                slerp_interpolator = Slerp([0, 1], rotations_array)
                interpolated_rotation = slerp_interpolator(weight).as_quat()
                interpolated_pose = [0,0,0,0,0,0,0]
                interpolated_pose[:4] = interpolated_rotation
                interpolated_pose[4:] = (1 - weight) * nearest_pose[4:] + weight * next_pose[4:]
                interpolated_poses.append(interpolated_pose)
                cam_timestamps.append(int(image_timestamp))

                
# Save the interpolated poses into camera_pose_indexed.txt
output_file = 'camera_pose_indexed.txt'
output_data = np.column_stack((cam_timestamps, interpolated_poses))
np.savetxt(output_file, output_data, delimiter=',', fmt='%.6f')