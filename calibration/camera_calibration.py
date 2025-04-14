import cv2
import numpy as np
import os

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
class Calibration:
    def __init__(self, camera_id, w, h, file_name='homography.npy', objectp3d=None, corners2=None):
        self.H = None
        self.file_name = file_name[:-4] + "_" + camera_id
        if os.path.exists(self.file_name):
            self.H = self.load_homography(self.file_name)
        elif objectp3d is None or corners2 is None:
            print("No 3D points or corresponding 2D points provided")
            return None
        self.camera_id  = camera_id
        self.threedpoints = []
        self.threedpoints.append(objectp3d)
        self.twodpoints = []
        self.twodpoints.append(corners2)
        self.im_shape = (w, h)

    def calibrate(self):
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(self.threedpoints, self.twodpoints, self.im_shape,
                                                                      None, None)
        r_vecs = np.array(r_vecs).reshape(1, 3)
        matrix = np.array(matrix)
        distortion = np.array(distortion)
        t_vecs = np.array(t_vecs).reshape(-1)

        R, _ = cv2.Rodrigues(r_vecs[0])
        R1 = R[:, 0]
        R2 = R[:, 1]
        t = t_vecs.reshape(3, 1)
        # Construct homography matrix H
        H = matrix @ np.hstack([R1.reshape(3, 1), R2.reshape(3, 1), t])


        self.ret = ret
        self.matrix = matrix
        self.distortion = distortion
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs
        self.H = H
        self.save_homography(H, self.file_name)


    def image_to_world_z(self, u, v):
        # Convert image point to homogeneous coordinates
        image_point = np.array([u, v, 1]).reshape(3, 1)

        # Compute world point in homogeneous coordinates
        world_point_h = np.linalg.inv(self.H) @ image_point

        # Normalize
        world_point_h /= world_point_h[2, 0]

        # Extract (X, Y)
        X, Y = world_point_h[:2, 0]
        return X, Y

    def save_homography(self, H, filename="homography.npy"):
        np.save(filename, H)
        print(f"Homography matrix saved to {filename}")

    def load_homography(self, filename="homography.npy"):
        H = np.load(filename)
        print(f"Homography matrix loaded from {filename}")
        return H


def calibrate_camera_2():
    image = cv2.imread("camera2/1_temp.jpg")
    CHECKERBOARD = (5, 8)
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].transpose((1, 2, 0)).reshape(-1, 2)
    objectp3d = objectp3d * 40.76
    objectp3d[:, :, 0] += 652
    objectp3d[:, :, 1] += 81

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

    CHECKERBOARD1 = (3, 9)
    objectp3d1 = np.zeros((1, CHECKERBOARD1[0] * CHECKERBOARD1[1], 3), np.float32)
    objectp3d1[0, :, :2] = np.mgrid[0:CHECKERBOARD1[0],
                           0:CHECKERBOARD1[1]].transpose((1, 2, 0)).reshape(-1, 2)
    objectp3d1 = objectp3d1 * 40.76

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

    h, w = image.shape[:2]
    calib = Calibration("camera2", w, h, objectp3d=objectp3d, corners2=corners2)
    if calib.H is None:
        calib.calibrate()

    u, v = 569, 753
    X, Y = calib.image_to_world_z(u, v)
    print(f"World Coordinates: X = {X}, Y = {Y}, Z = 0")

    return calib

def calibrate_camera_1():
    image = cv2.imread("camera1/1_temp.jpg")
    CHECKERBOARD = (5, 8)
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].transpose((1, 2, 0)).reshape(-1, 2)
    objectp3d = objectp3d * 40.76
    objectp3d[:, :, 0] += 652
    objectp3d[:, :, 1] += 81

    f = open("camera1/2d_point.txt")
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

    CHECKERBOARD1 = (3, 9)
    objectp3d1 = np.zeros((1, CHECKERBOARD1[0] * CHECKERBOARD1[1], 3), np.float32)
    objectp3d1[0, :, :2] = np.mgrid[0:CHECKERBOARD1[0],
                           0:CHECKERBOARD1[1]].transpose((1, 2, 0)).reshape(-1, 2)
    objectp3d1 = objectp3d1 * 40.76

    f = open("camera1/2d_point_00.txt")
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

    h, w = image.shape[:2]
    calib = Calibration("camera1", w, h, objectp3d=objectp3d, corners2=corners2)
    if calib.H is None:
        calib.calibrate()

    u, v = 1396, 233
    X, Y = calib.image_to_world_z(u, v)
    print(f"World Coordinates: X = {X}, Y = {Y}, Z = 0")

    return calib


if __name__ == '__main__':
    calibrate_camera_2()
