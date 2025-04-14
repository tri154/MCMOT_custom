# Import required modules
import cv2
import numpy as np
import os
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (5, 8)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

#  3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]* CHECKERBOARD[1], 3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                      0:CHECKERBOARD[1]].transpose((1, 2, 0)).reshape(-1, 2)
objectp3d = objectp3d * 40.76
objectp3d[:, :, 0] += 652
objectp3d[:, :, 1] += 81
# print(objectp3d)
# print("_"*30)



image = cv2.imread('camera2/0.jpg')
grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

f = open("camera2/2d_point.txt")
line = f.readline()
corners2 = list()
while line:
    line = line.strip()
    if line == '':
        line = f.readline()
        continue
    x, y = line.split(' ')
    x = int(x)
    y = int(y)
    corners2.append(np.array([x, y], dtype=np.float32))
    line = f.readline()

# corners2 = np.array(corners2)[:, np.newaxis, :]
# print(corners2.shape)
# prev_img_shape = None
# threedpoints.append(objectp3d)
# twodpoints.append(corners2)

# new point
CHECKERBOARD1 = (3, 9)
objectp3d1 = np.zeros((1, CHECKERBOARD1[0]* CHECKERBOARD1[1], 3), np.float32)
objectp3d1[0, :, :2] = np.mgrid[0:CHECKERBOARD1[0],
                      0:CHECKERBOARD1[1]].transpose((1, 2, 0)).reshape(-1, 2)
objectp3d1 = objectp3d1 * 40.76

image = cv2.imread('camera2/0_temp.jpg')
grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

f = open("camera2/2d_point_00.txt")
line = f.readline()
while line:
    line = line.strip()
    if line == '':
        line = f.readline()
        continue
    x, y = line.split(' ')
    x = int(x)
    y = int(y)
    corners2.append(np.array([x, y], dtype=np.float32))
    line = f.readline()

corners2 = np.array(corners2)[:, np.newaxis, :]
print(corners2.shape)
objectp3d = np.concatenate((objectp3d, objectp3d1), axis=1)
print(objectp3d.shape)
threedpoints.append(objectp3d)
twodpoints.append(corners2)


h, w = image.shape[:2]

ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

r_vecs = np.array(r_vecs).reshape(1, 3)
matrix = np.array(matrix)
distortion = np.array(distortion)
t_vecs = np.array(t_vecs).reshape(-1)

# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

import cv2
import numpy as np



def image_to_world(u, v, H):
    # Convert image point to homogeneous coordinates
    image_point = np.array([u, v, 1]).reshape(3, 1)

    # Compute world point in homogeneous coordinates
    world_point_h = np.linalg.inv(H) @ image_point

    # Normalize
    world_point_h /= world_point_h[2, 0]

    # Extract (X, Y)
    X, Y = world_point_h[:2, 0]
    return X, Y




R, _ = cv2.Rodrigues(r_vecs[0])

# Extract R1, R2 (first two columns of R)
R1 = R[:, 0]
R2 = R[:, 1]
t = t_vecs.reshape(3, 1)

# Construct homography matrix H
H = matrix @ np.hstack([R1.reshape(3, 1), R2.reshape(3, 1), t])

# Example usage
u, v = 569, 753   # Example image point
X, Y = image_to_world(u, v, H)
print(f"World Coordinates: X = {X}, Y = {Y}, Z = 0")
