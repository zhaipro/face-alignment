# coding: utf-8
import json
import math
import time

import cv2
import numpy as np


# 获取旋转向量和平移向量
def get_pose_estimation(img_size, points):
    # 3D model points.
    model_points = np.array([
                                [0.0, 0.0, 0.0],             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])
    model_points = np.array([
                                [0.0, 0.0, 0.0],             # Nose tip
                                (0.0, 330.0, 0.0),           # Chin
                                (-225.0, 170.0, -30.0),      # Left eye left corner
                                (225.0, 170.0, -30.0),       # Right eye right corne
                                (-150.0, -150.0, -30.0),    # Left Mouth corner
                                (150.0, -150.0, -30.0)      # Right mouth corner
                            ])
    model_points = np.array([
                                 [0.0,  0.0, 0.0],
                                 [0, 150, 0.0],
                                 [-100, -85, 0.0],
                                 [100, -85, 0.0],
                                 [-50, 60, 0.0],
                                 [50, 60, 0.0],
                            ])
    # model_points /= 2
    # Camera internals
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array([
                                [focal_length, 0, center[0]],
                                [0, img_size[0], center[1]],
                                [0, 0, 1]],
                                dtype='float64')

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    # print(help(cv2.solvePnP))
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return rotation_vector, translation_vector

# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    # 单位转换：将弧度转换为度
    Y = (pitch / math.pi) * 180
    X = (yaw / math.pi) * 180
    Z = (roll / math.pi) * 180

    return X, Y, Z


if __name__ == '__main__':
    data = np.load('../datasets/faces.npz')
    faces = data['faces']
    angles = data['angles']
    shapes = data['shapes']
    print(faces.shape)
    print(angles.shape)
    print(shapes.shape)

    for shape, face, angle in zip(shapes, faces, angles):
        rotation_vector, translation_vector = get_pose_estimation(shape, face)
        yaw, pitch, roll = get_euler_angle(rotation_vector)
        euler_angle_str = f'X:{yaw:.0f}, Y:{pitch:.0f}, Z:{roll:.0f}'
        print(euler_angle_str, angle)
