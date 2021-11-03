import cv2 as cv
import tifffile
import numpy as np
import glob
from skimage import morphology, filters
from matplotlib import colors, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functools import partial
from scipy import optimize, ndimage
from scipy.ndimage import filters as ndfilter
from PIL import Image
import queue
import copy
import math
import time
from pyneval.model.swc_node import SwcNode, SwcTree, Make_Virtual
from pyneval.tools.re_sample import down_sample_swc_tree
from pyneval.io.save_swc import swc_save
from swclib.SWC import *
import multiprocessing as mp
import random
from scipy.misc import derivative

alpha = 0.8
lambda_ = 0.3
#lambda_小 central大


beta = 0.3

def get_centerline_circle(centerline_sample_A, centerline_direction, r):
    circle_num = centerline_sample_A.shape[0]

    theta = np.arange(0.001, 2 * np.pi, 1 / (r + 0.1))

    circle_x = np.zeros([circle_num, theta.shape[0]])
    circle_y = np.zeros([circle_num, theta.shape[0]])
    circle_z = np.zeros([circle_num, theta.shape[0]])

    circle_center = np.zeros([circle_num, 3])
    circle_angle = np.zeros([circle_num, 2])

    # 圆的横截面 R
    # circle_x_plane_index = np.zeros([circle_num, r * theta.shape[0]])
    # circle_y_plane_index = np.zeros([circle_num, r * theta.shape[0]])
    # circle_z_plane_index = np.zeros([circle_num, r * theta.shape[0]])

    u_vector = np.zeros([circle_num, 3])
    v_vector = np.zeros([circle_num, 3])

    for i in range(circle_num):
        # 获取圆的中心位置
        circle_mid_x = centerline_sample_A[i][0]
        circle_mid_y = centerline_sample_A[i][1]
        circle_mid_z = centerline_sample_A[i][2]

        circle_center[i][0] = circle_mid_x
        circle_center[i][1] = circle_mid_y
        circle_center[i][2] = circle_mid_z

        # 获取centerline, norm_vector
        norm_vector_x = centerline_direction[i][0]
        norm_vector_y = centerline_direction[i][1]
        norm_vector_z = centerline_direction[i][2]

        # 获取平面上的向量u
        u_x = norm_vector_y
        u_y = - norm_vector_x
        u_z = 1e-7

        # 获取与n,u都正交的向量v
        v_x = norm_vector_x * norm_vector_z
        v_y = norm_vector_y * norm_vector_z
        v_z = - norm_vector_x ** 2 - norm_vector_y ** 2 + 1e-7

        # 获取u,v的单位向量
        u_n = math.sqrt(u_x ** 2 + u_y ** 2 + u_z ** 2)
        u_x_tilde = u_x / u_n
        u_y_tilde = u_y / u_n
        u_z_tilde = u_z / u_n

        v_n = math.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
        v_x_tilde = v_x / v_n
        v_y_tilde = v_y / v_n
        v_z_tilde = v_z / v_n

        u_vector[i][0] = u_x_tilde
        u_vector[i][1] = u_y_tilde
        u_vector[i][2] = u_z_tilde

        v_vector[i][0] = v_x_tilde
        v_vector[i][1] = v_y_tilde
        v_vector[i][2] = v_z_tilde

        # 圆的参数方程
        theta = np.arange(0.001, 2 * np.pi, 1 / (r + 0.1))

        x = circle_mid_x + r * u_x_tilde * np.cos(theta) + r * v_x_tilde * np.sin(theta)
        y = circle_mid_y + r * u_y_tilde * np.cos(theta) + r * v_y_tilde * np.sin(theta)
        z = circle_mid_z + r * u_z_tilde * np.cos(theta) + r * v_z_tilde * np.sin(theta)

        circle_x[i:i + 1, :] = copy.deepcopy(x.reshape([1, theta.shape[0]]))
        circle_y[i:i + 1, :] = copy.deepcopy(y.reshape([1, theta.shape[0]]))
        circle_z[i:i + 1, :] = copy.deepcopy(z.reshape([1, theta.shape[0]]))

        x_angle = 360 - math.atan(norm_vector_z / (norm_vector_x + 1e-5)) * 57.3
        y_angle = 360 - math.atan(norm_vector_y / (norm_vector_z + 1e-5)) * 57.3

        circle_angle[i][0] = x_angle
        circle_angle[i][1] = y_angle

        # 圆的横截面 R
        # for r_temp in range(1,R+1):
        #     x_temp = circle_mid_x + r_temp * u_x_tilde * np.cos(theta) + r_temp * v_x_tilde * np.sin(theta)
        #     y_temp = circle_mid_y + r_temp * u_y_tilde * np.cos(theta) + r_temp * v_y_tilde * np.sin(theta)
        #     z_temp = circle_mid_z + r_temp * u_z_tilde * np.cos(theta) + r_temp * v_z_tilde * np.sin(theta)
        #     circle_x_plane_index[i:i + 1, (r_temp-1)*theta.shape[0]:(r_temp)*theta.shape[0]] = copy.deepcopy(x_temp.reshape([1, theta.shape[0]]))
        #     circle_y_plane_index[i:i + 1, (r_temp-1)*theta.shape[0]:(r_temp)*theta.shape[0]] = copy.deepcopy(y_temp.reshape([1, theta.shape[0]]))
        #     circle_z_plane_index[i:i + 1, (r_temp-1)*theta.shape[0]:(r_temp)*theta.shape[0]] = copy.deepcopy(z_temp.reshape([1, theta.shape[0]]))

    # 四舍五入取整
    # circle_x = np.around(circle_x)
    # circle_y = np.around(circle_y)
    # circle_z = np.around(circle_z)
    # circle_center = np.around(circle_center)

    return circle_x, circle_y, circle_z, circle_center, circle_angle, u_vector, v_vector

def o_distance(point_A, point_B):
    distance = math.sqrt((point_A[0][0] - point_B[0][0]) ** 2 + (point_A[0][1] - point_B[0][1]) ** 2 + (
            point_A[0][2] - point_B[0][2]) ** 2)
    return distance

def find_centerline_flag_0(centerline_flag):
    point_num = centerline_flag.shape[0]
    for i in range(point_num):
        if centerline_flag[i] == 0:
            return False, i
    return True, -1

def swc_save_lc(save_dir, swc_data):
    with open(save_dir, 'w') as fp:
        for i in range(swc_data.shape[0]):
            fp.write('%d %d %g %g %g %g %d\n' % (
                swc_data[i][0], swc_data[i][1], swc_data[i][2], swc_data[i][3], swc_data[i][4], swc_data[i][5],
                swc_data[i][6]))
        fp.close()

def convert_centerline_to_swc(centerline):
    centerline_tree = np.zeros([centerline.shape[0], 7])
    for i in range(centerline.shape[0]):
        centerline_tree[i][0] = i + 1
        centerline_tree[i][1] = 1
        centerline_tree[i][5] = 1
    # print(centerline_tree.shape)

    point_num = centerline.shape[0]
    centerline_flag = np.zeros([point_num])
    # 待判断点队列
    temp_queue_index = queue.Queue()
    temp_queue_parent = queue.Queue()
    # 初始点加入待判断队列
    is_end, begin_pos = find_centerline_flag_0(centerline_flag)
    temp_queue_index.put(begin_pos)
    temp_queue_parent.put(-1)
    current_id = 0

    while is_end == False:
        son_point_num = 0
        temp_index = temp_queue_index.get()
        parent_id = temp_queue_parent.get()

        # print(temp_index,current_id)
        # centerline_tree[current_id][2] = centerline[temp_index][2]
        # centerline_tree[current_id][3] = centerline[temp_index][1]
        # centerline_tree[current_id][4] = centerline[temp_index][0]
        # centerline_tree[current_id][6] = parent_id

        # if current_id == 717:
        #     break

        point_A = copy.deepcopy(centerline[temp_index:temp_index + 1, :])

        # 将当前的判断点flag设置为1
        centerline_flag[temp_index] = 1

        for i in range(point_num):
            if centerline_flag[i] == 0:
                point_B = copy.deepcopy(centerline[i:i + 1, :])
                temp_distance = o_distance(point_A, point_B)

                # 如果节点距离在 更号3 以内，说明为子节点
                if temp_distance < math.sqrt(4):
                    current_id = current_id + 1
                    print(temp_index, current_id)
                    # centerline_tree[current_id][2:5] = centerline[i][0:3]
                    temp_queue_index.put(i)
                    temp_queue_parent.put(temp_index + 1)

                    son_point_num = son_point_num + 1
        time.sleep(0.0000001)
        is_Empty = temp_queue_index.empty()
        if is_Empty == True:
            is_end, temp_index = find_centerline_flag_0(centerline_flag)
            temp_queue_index.put(temp_index)
            temp_queue_parent.put(-1)

    # swc_save_dir = 'data/test.swc'
    # swc_save_lc(swc_save_dir, centerline_tree)
    return centerline_tree

def get_centerline_direction_and_neighbour(resample_tree_data):
    centerline_direction = np.zeros([resample_tree_data.shape[0], 3])
    centerline_neighbour = np.zeros([resample_tree_data.shape[0], 2])
    for i in range(resample_tree_data.shape[0]):
        node_id = resample_tree_data[i][0]
        node_id_p = resample_tree_data[i][6]

        if node_id_p == -1:
            node_A = int(node_id) - 1
            node_B = int(node_id + 1) - 1

            neigh_A = -1
            neigh_B = int(node_id + 1) - 1
        else:

            node_son = int(node_id + 1) - 1
            if node_id != resample_tree_data.shape[0]:
                if resample_tree_data[node_son][6] == node_id:
                    node_A = int(node_id) - 1
                    node_B = int(node_id_p) - 1

                    neigh_A = int(node_id_p) - 1
                    neigh_B = int(node_id + 1) - 1
            else:
                node_A = int(node_id) - 1
                node_B = int(node_id_p) - 1

                neigh_A = int(node_id_p) - 1
                neigh_B = -1

        centerline_direction[i][0] = resample_tree_data[node_A][4] - resample_tree_data[node_B][4]
        centerline_direction[i][1] = resample_tree_data[node_A][3] - resample_tree_data[node_B][3]
        centerline_direction[i][2] = resample_tree_data[node_A][2] - resample_tree_data[node_B][2]

        # print(centerline_direction[i])
        # pause

        # centerline_direction[i][0] = -2
        # centerline_direction[i][1] = -4
        # centerline_direction[i][2] = 0
        #========================================================================

        centerline_neighbour[i][0] = neigh_A
        centerline_neighbour[i][1] = neigh_B

    return centerline_direction, centerline_neighbour

def image_with_circle(img_, circle_x, circle_y, circle_z, color=0, init=False):
    if init == True:
        SHAPE = img_.shape
        color_data = np.zeros([SHAPE[0], 3, SHAPE[1], SHAPE[2]], dtype=np.uint8)
        color_data[:, 0, :, :] = copy.deepcopy(img_)
        color_data[:, 1, :, :] = copy.deepcopy(img_)
        color_data[:, 2, :, :] = copy.deepcopy(img_)
    else:
        color_data = img_

    SHAPE = color_data.shape
    circle_x = np.floor(circle_x)
    circle_y = np.floor(circle_y)
    circle_z = np.floor(circle_z)

    for i in range(circle_x.shape[0]):
        for j in range(circle_x.shape[1]):
            x = int(circle_x[i][j])
            y = int(circle_y[i][j])
            z = int(circle_z[i][j])

            if x >= SHAPE[2]:
                x = SHAPE[2] - 1
            if x < 0:
                x = 0
            if y >= SHAPE[3]:
                y = SHAPE[3] - 1
            if y < 0:
                y = 0
            if z >= SHAPE[0]:
                z = SHAPE[0] - 1
            if z < 0:
                z = 0

            if color == 0:
                color_data[z][0][y][x] = 255
                color_data[z][1][y][x] = 0
                color_data[z][2][y][x] = 0
            if color == 1:
                color_data[z][0][y][x] = 0
                color_data[z][1][y][x] = 255
                color_data[z][2][y][x] = 0
            if color == 2:
                color_data[z][0][y][x] = 0
                color_data[z][1][y][x] = 0
                color_data[z][2][y][x] = 255

    return color_data

# =====================================================================================================================
def R_kernal(x_pos, y_pos, radius):
    return (x_pos ** 2 + y_pos ** 2) / radius ** 2

def K_kernal(x_pos, y_pos, radius):
    result = (2 - R_kernal(x_pos, y_pos, radius)) * np.exp(-R_kernal(x_pos, y_pos, radius) / 2)
    return result

def K_kernal_jac(x_pos, y_pos, radius):
    result = (4* R_kernal(x_pos, y_pos, radius) - R_kernal(x_pos, y_pos, radius)**2) * np.exp(-R_kernal(x_pos, y_pos, radius) / 2) / radius
    return result

def snake_energy_3D_ext_Offset(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    external_offset_energy = np.zeros([pts.shape[0]])

    for circle_num in range(pts.shape[0]):
        pt_x = pts[circle_num][0]
        pt_y = pts[circle_num][1]
        pt_r = pts[circle_num][2]
        u = u_vector[circle_num]
        v = v_vector[circle_num]
        pos = pos_all[circle_num]

        pts_z = pos[0] + pt_x * u[0] + pt_y * v[0]
        pts_x = pos[1] + pt_x * u[1] + pt_y * v[1]
        pts_y = pos[2] + pt_x * u[2] + pt_y * v[2]

        # 新的参数方程
        if pt_r < 0.5:
            pt_r = 0.5
        theta = np.arange(1e-5, 2 * np.pi, 1 / pt_r + 1e-5)

        cl_z = pts_z + pt_r * u[0] * np.cos(theta) + pt_r * v[0] * np.sin(theta)
        cl_x = pts_x + pt_r * u[1] * np.cos(theta) + pt_r * v[1] * np.sin(theta)
        cl_y = pts_y + pt_r * u[2] * np.cos(theta) + pt_r * v[2] * np.sin(theta)

        # offset term
        dist_vals_grad = np.zeros([3, theta.shape[0]])

        dist_vals_grad[0] = ndimage.interpolation.map_coordinates(img_grad[0], [cl_z, cl_x, cl_y], order=1)  # z
        dist_vals_grad[1] = ndimage.interpolation.map_coordinates(img_grad[1], [cl_z, cl_x, cl_y], order=1)  # x
        dist_vals_grad[2] = ndimage.interpolation.map_coordinates(img_grad[2], [cl_z, cl_x, cl_y], order=1)  # y

        o = np.zeros([3, theta.shape[0]])
        o[0] = u[0] * np.cos(theta) + v[0] * np.sin(theta)
        o[1] = u[1] * np.cos(theta) + v[1] * np.sin(theta)
        o[2] = u[2] * np.cos(theta) + v[2] * np.sin(theta)

        external_energy_offset = - np.mean(dist_vals_grad * o) * 3
        external_offset_energy[circle_num] = external_energy_offset

    external_offset_energy_mean = -np.sum(external_offset_energy)/ pts.shape[0]

    # r_reg = 1 * (np.mean(1 / (pts[:, 2] ** 2)) + np.mean(pts[:, 2] ** 2))

    return external_offset_energy_mean

def snake_energy_3D_ext_Central(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    external_central_energy = np.zeros([pts.shape[0]])

    for circle_num in range(pts.shape[0]):
    # for circle_num in range(0,1):
        pt_x = pts[circle_num][0]
        pt_y = pts[circle_num][1]
        pt_r = pts[circle_num][2]


        u = u_vector[circle_num]
        v = v_vector[circle_num]
        pos = pos_all[circle_num]

        # print(u)
        # print(v)

        # u = [-1,0,0]
        # v = [0,0,-1]

        # print(u)
        # print(v)

        # pause

        # pt_x = -10
        # pt_y = 0


        pts_z = pos[0] + pt_x * u[0] + pt_y * v[0]
        pts_x = pos[1] + pt_x * u[1] + pt_y * v[1]
        pts_y = pos[2] + pt_x * u[2] + pt_y * v[2]

        # print(pts_z,pts_x,pts_y)
        # pause

        # if pt_r < 0.5:
        #     pt_r = 0.5


        # X = np.arange(-3.0 * pt_r, 3.0 * pt_r, pt_r/6+1e-5)
        # Y = np.arange(-3.0 * pt_r, 3.0 * pt_r, pt_r/6+1e-5)
        X = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        Y = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        X_mesh, Y_mesh = np.meshgrid(X, Y)


        Z_new = pts_z + X_mesh * u[0] + Y_mesh * v[0]
        X_new = pts_x + X_mesh * u[1] + Y_mesh * v[1]
        Y_new = pts_y + X_mesh * u[2] + Y_mesh * v[2]

        dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Z_new, X_new, Y_new], order=1)  # Y, X, Z
        K_value = K_kernal(X_mesh, Y_mesh, pt_r)

        # for i in range(X_mesh.shape[0]):
        #     for j in range(X_mesh.shape[1]):
        #         # 3R 范围内
        #         if math.sqrt((X_mesh[i][j]) ** 2 + (Y_mesh[i][j]) ** 2) > 3 * pt_r:
        #             K_value[i][j] = 0

        external_central_energy[circle_num] = - np.mean(K_value * dist_vals_img_org)
        # print(Z_new[1][0])
        # print(X_new)
        # print(Y_new[1][0])

        # fig = plt.figure()
        # ax = fig.add_subplot(1,3,1, projection='3d')
        # ax.scatter(X_mesh, Y_mesh, K_value, s=1,c='g')
        # ax = fig.add_subplot(1,3,2, projection='3d')
        # ax.scatter(X_mesh, Y_mesh, dist_vals_img_org, s=1,c='g')
        # ax = fig.add_subplot(1, 2, 1, projection='3d')
        # ax.scatter(Z_new, X_new, Y_new, s=1, c='g')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax = fig.add_subplot(1, 2, 2)
        # ax.scatter(pts_z,pts_x,pts_y, s=3, c='r')
        # plt.imshow(dist_vals_img_org, cmap=plt.cm.gray)
        # ax.scatter(Z_new, Y_new, s=1, c='g')

        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()



    # print(np.mean(external_central_energy))
    # print(np.sum(external_central_energy))
    # pause
    external_central_energy_mean = np.sum(external_central_energy)/ pts.shape[0]

    return external_central_energy_mean


def snake_energy_3D_int(flattened_pts,img_org, img_grad, pos_all, neighbour, u_vector, v_vector):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    internal_energy = np.zeros([pts.shape[0]])

    for circle_num in range(pts.shape[0]):

        pt_x = pts[circle_num][0]
        pt_y = pts[circle_num][1]
        pt_r = pts[circle_num][2]
        u = u_vector[circle_num]
        v = v_vector[circle_num]
        pos = pos_all[circle_num]

        pts_y = pos[0] + pt_x * u[0] + pt_y * v[0]
        pts_x = pos[1] + pt_x * u[1] + pt_y * v[1]
        pts_z = pos[2] + pt_x * u[2] + pt_y * v[2]

        # 内部能量项 internal energy

        node_A = int(neighbour[circle_num][0])
        node_B = int(neighbour[circle_num][1])

        if node_A == -1:
            pos_B = pos_all[node_B]
            u_B = u_vector[node_B]
            v_B = v_vector[node_B]
            node_B_y = pos_B[0] + pts[node_B][0] * u_B[0] + pts[node_B][1] * v_B[0]
            node_B_x = pos_B[1] + pts[node_B][0] * u_B[1] + pts[node_B][1] * v_B[1]
            node_B_z = pos_B[2] + pts[node_B][0] * u_B[2] + pts[node_B][1] * v_B[2]
            node_B_r = pts[node_B][2]
            internal_energy_term = 0.5 * (
                    beta * ((pts_x - node_B_x) ** 2 + (pts_y - node_B_y) ** 2 + (pts_z - node_B_z) ** 2) + (
                    1 - beta) * ((pt_r - node_B_r) ** 2))
        elif node_B == -1:
            pos_A = pos_all[node_A]
            u_A = u_vector[node_A]
            v_A = v_vector[node_A]
            node_A_y = pos_A[0] + pts[node_A][0] * u_A[0] + pts[node_A][1] * v_A[0]
            node_A_x = pos_A[1] + pts[node_A][0] * u_A[1] + pts[node_A][1] * v_A[1]
            node_A_z = pos_A[2] + pts[node_A][0] * u_A[2] + pts[node_A][1] * v_A[2]
            node_A_r = pts[node_A][2]
            internal_energy_term = 0.5 * (
                    beta * ((pts_x - node_A_x) ** 2 + (pts_y - node_A_y) ** 2 + (pts_z - node_A_z) ** 2) + (
                    1 - beta) * ((pt_r - node_A_r) ** 2))
        else:
            pos_A = pos_all[node_A]
            u_A = u_vector[node_A]
            v_A = v_vector[node_A]
            pos_B = pos_all[node_B]
            u_B = u_vector[node_B]
            v_B = v_vector[node_B]
            node_A_y = pos_A[0] + pts[node_A][0] * u_A[0] + pts[node_A][1] * v_A[0]
            node_A_x = pos_A[1] + pts[node_A][0] * u_A[1] + pts[node_A][1] * v_A[1]
            node_A_z = pos_A[2] + pts[node_A][0] * u_A[2] + pts[node_A][1] * v_A[2]
            node_A_r = pts[node_A][2]
            node_B_y = pos_B[0] + pts[node_B][0] * u_B[0] + pts[node_B][1] * v_B[0]
            node_B_x = pos_B[1] + pts[node_B][0] * u_B[1] + pts[node_B][1] * v_B[1]
            node_B_z = pos_B[2] + pts[node_B][0] * u_B[2] + pts[node_B][1] * v_B[2]
            node_B_r = pts[node_B][2]

            internal_energy_term = 0.25 * (beta * (
                    (pts_x - node_A_x) ** 2 + (pts_y - node_A_y) ** 2 + (pts_z - node_A_z) ** 2 + (
                    pts_x - node_B_x) ** 2 + (pts_y - node_B_y) ** 2 + (pts_z - node_B_z) ** 2) + (1 - beta) * (
                                                   (pt_r - node_A_r) ** 2 + (pt_r - node_B_r) ** 2))
        internal_energy[circle_num] = internal_energy_term
    internal_energy_sum = np.sum(internal_energy) / pts.shape[0]

    return internal_energy_sum


def snake_energy_3D_new(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector):

    external_energy_offset = snake_energy_3D_ext_Offset(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector)
    # external_energy_offset = 0
    # external_energy_central = snake_energy_3D_ext_Central(flattened_pts, img_org, img_grad, pos_all, neighbour,u_vector, v_vector)
    external_energy_central = 0

    # internal_energy = snake_energy_3D_int(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector)
    internal_energy = 0

    energy_sum = alpha * (lambda_ * external_energy_offset + (1 - lambda_) * external_energy_central) + (1 - alpha) * internal_energy
    print('offset: %2f, central: %2f, internal: %2f' % (alpha * lambda_ * external_energy_offset, alpha * (1 - lambda_) *external_energy_central, (1 - alpha) * internal_energy))
    return energy_sum


def snake_energy_3D_jac(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector):
    der = np.zeros_like(flattened_pts)
    # der_temp = 0.1
    der_temp = 1e-5

    for i in range(flattened_pts.shape[0]):
        x_new_big = copy.deepcopy(flattened_pts)
        x_new_small = copy.deepcopy(flattened_pts)
        x_new_big[i] += der_temp
        x_new_small[i] -= der_temp

        x_new_big_value = snake_energy_3D_new(x_new_big.ravel(), img_org=img_org, img_grad=img_grad,
                                              pos_all=pos_all,
                                              neighbour=neighbour,
                                              u_vector=u_vector, v_vector=v_vector)
        x_new_small_value = snake_energy_3D_new(x_new_small.ravel(), img_org=img_org, img_grad=img_grad,
                                                pos_all=pos_all, neighbour=neighbour,
                                                u_vector=u_vector, v_vector=v_vector)


        der[i] = (x_new_big_value - x_new_small_value) / (2 * der_temp)

    return der


def snake_energy_3D_jac_int_ext(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector):
    der = np.zeros_like(flattened_pts)

    # der_temp = 0.1
    der_temp = 1e-5

    for i in range(flattened_pts.shape[0]):
        x_new_big = copy.deepcopy(flattened_pts)
        x_new_small = copy.deepcopy(flattened_pts)
        x_new_big[i] += der_temp
        x_new_small[i] -= der_temp

        # =================================OFFSET=====================================
        x_new_big_value = snake_energy_3D_ext_Offset(x_new_big.ravel(), img_org=img_org, img_grad=img_grad,
                                                     pos_all=pos_all, neighbour=neighbour,
                                                     u_vector=u_vector, v_vector=v_vector)
        x_new_small_value = snake_energy_3D_ext_Offset(x_new_small.ravel(), img_org=img_org, img_grad=img_grad,
                                                       pos_all=pos_all, neighbour=neighbour,
                                                       u_vector=u_vector, v_vector=v_vector)
        der[i] += alpha * lambda_ * (x_new_big_value - x_new_small_value) / (2 * der_temp)
        # =================================CENTRAL=====================================
        # print("central")
        x_new_big_value = snake_energy_3D_ext_Central(x_new_big.ravel(), img_org=img_org,
                                                      img_grad=img_grad,
                                                      pos_all=pos_all, neighbour=neighbour,
                                                      u_vector=u_vector, v_vector=v_vector)
        x_new_small_value = snake_energy_3D_ext_Central(x_new_small.ravel(),
                                                        img_org=img_org,
                                                        img_grad=img_grad,
                                                        pos_all=pos_all, neighbour=neighbour,
                                                        u_vector=u_vector, v_vector=v_vector)
        # der[i] += alpha * (1 - lambda_) * (x_new_big_value - x_new_small_value) / (2 * der_temp)

        # =================================INTERNAL=====================================
        x_new_big_value = snake_energy_3D_int(x_new_big.ravel(), img_org=img_org, img_grad=img_grad, pos_all=pos_all,
                                              neighbour=neighbour,
                                              u_vector=u_vector, v_vector=v_vector)
        x_new_small_value = snake_energy_3D_int(x_new_small.ravel(), img_org=img_org, img_grad=img_grad,
                                                pos_all=pos_all, neighbour=neighbour,
                                                u_vector=u_vector, v_vector=v_vector)
        # der[i] += (1 - alpha) * (x_new_big_value - x_new_small_value) / (2 * der_temp)
    return der


def snake_energy_3D_jac_test(flattened_pts, img_org, img_grad, img_grad_jac, pos_all, neighbour, u_vector, v_vector):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    der = np.zeros_like(pts)
    der_test = np.zeros_like(pts)

    np.set_printoptions(suppress=True)
    # print("=====================ext=======================")
    der_temp = 1e-3

    # for i in range(pts.shape[0]):
    #     for j in range(pts.shape[1]):
    #         x_new_big = copy.deepcopy(pts)
    #         x_new_small = copy.deepcopy(pts)
    #         x_new_big[i][j] += der_temp
    #         x_new_small[i][j] -= der_temp
    #
    #         # print("ext offset")
    #         x_new_big_value_Offset = snake_energy_3D_ext_Offset(x_new_big.ravel(), img_org=img_org, img_grad=img_grad,
    #                                                       pos_all=pos_all, neighbour=neighbour,
    #                                                       u_vector=u_vector, v_vector=v_vector)
    #         x_new_small_value_Offset = snake_energy_3D_ext_Offset(x_new_small.ravel(), img_org=img_org, img_grad=img_grad,
    #                                                         pos_all=pos_all, neighbour=neighbour,
    #                                                         u_vector=u_vector, v_vector=v_vector)
    #         der_test[i][j] += (x_new_big_value_Offset - x_new_small_value_Offset) / (2 * der_temp)
    #         # print("ext central")
    #         x_new_big_value_Central = snake_energy_3D_ext_Central(x_new_big.ravel(), img_org=img_org,
    #                                                               img_grad=img_grad,
    #                                                               pos_all=pos_all, neighbour=neighbour,
    #                                                               u_vector=u_vector, v_vector=v_vector)
    #         x_new_small_value_Central = snake_energy_3D_ext_Central(x_new_small.ravel(), img_org=img_org,
    #                                                                 img_grad=img_grad,
    #                                                                 pos_all=pos_all, neighbour=neighbour,
    #                                                                 u_vector=u_vector, v_vector=v_vector)
    #         der_test[i][j] += (x_new_big_value_Central - x_new_small_value_Central) / (2 * der_temp)

    #         x_new_big_value_int = snake_energy_3D_int(x_new_big.ravel(), img_org=img_org, img_grad=img_grad,pos_all=pos_all, neighbour=neighbour,u_vector=u_vector, v_vector=v_vector)
    #         x_new_small_value_int = snake_energy_3D_int(x_new_small.ravel(), img_org=img_org,img_grad=img_grad,pos_all=pos_all, neighbour=neighbour,u_vector=u_vector, v_vector=v_vector)
    #         der_test[i][j] += (x_new_big_value_int - x_new_small_value_int) / (2 * der_temp)

    # print("===========")
    # print("-----der_test------")
    # print(der_test)
    # print("-------------------")

    for circle_num in range(pts.shape[0]):

        pt_x = pts[circle_num][0]
        pt_y = pts[circle_num][1]
        pt_r = pts[circle_num][2]
        u = u_vector[circle_num]
        v = v_vector[circle_num]
        pos = pos_all[circle_num]

        pts_z = pos[0] + pt_x * u[0] + pt_y * v[0]
        pts_x = pos[1] + pt_x * u[1] + pt_y * v[1]
        pts_y = pos[2] + pt_x * u[2] + pt_y * v[2]

        # 新的参数方程
        if pt_r < 0.5:
            pt_r = 0.5

        theta = np.arange(1e-5, 2 * np.pi, 1 / pt_r)
        cl_z = pts_z + pt_r * u[0] * np.cos(theta) + pt_r * v[0] * np.sin(theta)
        cl_x = pts_x + pt_r * u[1] * np.cos(theta) + pt_r * v[1] * np.sin(theta)
        cl_y = pts_y + pt_r * u[2] * np.cos(theta) + pt_r * v[2] * np.sin(theta)

        # print("pos:")
        # print(cl_z)
        # print(cl_x)
        # print(cl_y)
        # pause

        # img_grad z y x
        dist_vals_grad = np.zeros([3, theta.shape[0]])
        dist_vals_grad[0] = ndimage.interpolation.map_coordinates(img_grad[0], [cl_z, cl_x, cl_y], order=1)
        dist_vals_grad[1] = ndimage.interpolation.map_coordinates(img_grad[1], [cl_z, cl_x, cl_y], order=1)
        dist_vals_grad[2] = ndimage.interpolation.map_coordinates(img_grad[2], [cl_z, cl_x, cl_y], order=1)

        o = np.zeros([3, theta.shape[0]])
        o[0] = u[0] * np.cos(theta) + v[0] * np.sin(theta)
        o[1] = u[1] * np.cos(theta) + v[1] * np.sin(theta)
        o[2] = u[2] * np.cos(theta) + v[2] * np.sin(theta)

        # print("img_o:")
        # print(o[0])
        # print(o[1])
        # print(o[2])

        # img_grad_jac x y z 1 0 2
        dist_vals_jac = np.zeros([3, 3, theta.shape[0]])
        dist_vals_jac[0][0] = ndimage.interpolation.map_coordinates(img_grad_jac[0][0], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[0][1] = ndimage.interpolation.map_coordinates(img_grad_jac[0][1], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[0][2] = ndimage.interpolation.map_coordinates(img_grad_jac[0][2], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[1][0] = ndimage.interpolation.map_coordinates(img_grad_jac[1][0], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[1][1] = ndimage.interpolation.map_coordinates(img_grad_jac[1][1], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[1][2] = ndimage.interpolation.map_coordinates(img_grad_jac[1][2], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[2][0] = ndimage.interpolation.map_coordinates(img_grad_jac[2][0], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[2][1] = ndimage.interpolation.map_coordinates(img_grad_jac[2][1], [cl_z, cl_x, cl_y], order=1)
        dist_vals_jac[2][2] = ndimage.interpolation.map_coordinates(img_grad_jac[2][2], [cl_z, cl_x, cl_y], order=1)

        o = np.zeros([3, 1, theta.shape[0]])
        o[0][0] = u[0] * np.cos(theta) + v[0] * np.sin(theta)
        o[1][0] = u[1] * np.cos(theta) + v[1] * np.sin(theta)
        o[2][0] = u[2] * np.cos(theta) + v[2] * np.sin(theta)

        offset_jac = np.zeros([3, theta.shape[0]])
        for i in range(theta.shape[0]):
            offset_jac[0, i] = dist_vals_jac[:, :, i] @ o[:, 0, i] @ u
            offset_jac[1, i] = dist_vals_jac[:, :, i] @ o[:, 0, i] @ v
            offset_jac[2, i] = o[:, 0, i].T @ dist_vals_jac[:, :, i] @ o[:, 0, i]

        der[circle_num][0] += alpha * lambda_ * np.mean(offset_jac[0]) / pts.shape[0]
        der[circle_num][1] += alpha * lambda_ * np.mean(offset_jac[1]) / pts.shape[0]
        der[circle_num][2] += alpha * lambda_ * np.mean(offset_jac[2]) / pts.shape[0]

        offset_jac_x = np.mean(offset_jac[0]) #/ pts.shape[0]
        offset_jac_y = np.mean(offset_jac[1]) #/ pts.shape[0]
        offset_jac_r = np.mean(offset_jac[2]) #/ pts.shape[0]

        # print('%d, %5f, %5f, %5f' % (circle_num, offset_jac_x, offset_jac_y, offset_jac_r))
        # pause

        # ===============================central jac ==========================================

        X = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        Y = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        # X = np.arange(-3.0 * pt_r, 3.0 * pt_r, pt_r/6+1e-5)
        # Y = np.arange(-3.0 * pt_r, 3.0 * pt_r, pt_r/6+1e-5)
        X_mesh, Y_mesh = np.meshgrid(X, Y)

        # X = np.arange(- 3.0 * pts_r, 3.0 * pts_r, pts_r/6)
        # Y = np.arange(- 3.0 * pts_r, 3.0 * pts_r, pts_r/6)#pts_r/6
        # X_mesh, Y_mesh = np.meshgrid(X, Y)
        # # print(X_mesh)
        # # pause
        # X_mesh = copy.deepcopy(X_mesh[0:36, 0:36])
        # Y_mesh = copy.deepcopy(Y_mesh[0:36, 0:36])

        # r_temp = 0.1
        # X1 = np.arange(- 3 * (pts_r + r_temp), 3 * (pts_r + r_temp), (pts_r + r_temp)/6)
        # Y1 = np.arange(- 3 * (pts_r + r_temp), 3 * (pts_r + r_temp), (pts_r + r_temp)/6)
        # X_new_mesh1, Y_new_mesh1 = np.meshgrid(X1, Y1)
        # X_new_mesh1 = copy.deepcopy(X_new_mesh1[0:36, 0:36])
        # Y_new_mesh1 = copy.deepcopy(Y_new_mesh1[0:36, 0:36])
        #
        # X2 = np.arange(- 3 * (pts_r - r_temp), 3 * (pts_r - r_temp), (pts_r - r_temp)/6)
        # Y2 = np.arange(- 3 * (pts_r - r_temp), 3 * (pts_r - r_temp), (pts_r - r_temp)/6)
        # X_new_mesh2, Y_new_mesh2 = np.meshgrid(X2, Y2)
        # X_new_mesh2 = copy.deepcopy(X_new_mesh2[0:36, 0:36])
        # Y_new_mesh2 = copy.deepcopy(Y_new_mesh2[0:36, 0:36])

        Z_new = pts_z + X_mesh * u[0] + Y_mesh * v[0]
        X_new = pts_x + X_mesh * u[1] + Y_mesh * v[1]
        Y_new = pts_y + X_mesh * u[2] + Y_mesh * v[2]

        # Z_new1 = pts_z + X_new_mesh1 * u[0] + Y_new_mesh1 * v[0]
        # X_new1 = pts_x + X_new_mesh1 * u[1] + Y_new_mesh1 * v[1]
        # Y_new1 = pts_y + X_new_mesh1 * u[2] + Y_new_mesh1 * v[2]
        #
        # Z_new2 = pts_z + X_new_mesh2 * u[0] + Y_new_mesh2 * v[0]
        # X_new2 = pts_x + X_new_mesh2 * u[1] + Y_new_mesh2 * v[1]
        # Y_new2 = pts_y + X_new_mesh2 * u[2] + Y_new_mesh2 * v[2]

        img_org = img_org.astype(np.float64)

        dist_vals_grad = np.zeros([3, X_mesh.shape[0], X_mesh.shape[1]])

        dist_vals_grad[0] = ndimage.interpolation.map_coordinates(img_grad[0], [Z_new, X_new, Y_new])
        dist_vals_grad[1] = ndimage.interpolation.map_coordinates(img_grad[1], [Z_new, X_new, Y_new])
        dist_vals_grad[2] = ndimage.interpolation.map_coordinates(img_grad[2], [Z_new, X_new, Y_new])
        dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Z_new, X_new, Y_new])  # Y, X, Z
        # dist_vals_img_org_new1 = ndimage.interpolation.map_coordinates(img_org, [Z_new1, X_new1, Y_new1],order=1)  # Y, X, Z
        # dist_vals_img_org_new2 = ndimage.interpolation.map_coordinates(img_org, [Z_new2, X_new2, Y_new2],order=1)  # Y, X, Z


        K_value = K_kernal(X_mesh, Y_mesh, pt_r) #* K_value_mask
        # K_value1 = K_kernal(X_new_mesh1, Y_new_mesh1, pts_r + r_temp)
        # K_value2 = K_kernal(X_new_mesh2, Y_new_mesh2, pts_r - r_temp)
        #
        # K_value11 = K_kernal(X_new_mesh1, Y_new_mesh1, pts_r)
        # K_value22 = K_kernal(X_new_mesh2, Y_new_mesh2, pts_r)


        K_value_jac = K_kernal_jac(X_mesh, Y_mesh, pt_r) #* K_value_mask
        # K_value_new = K_kernal(X_mesh, Y_mesh, (pts_r+0.01)) * K_value_mask
        # K_value_jac_test = (K_value_new - K_value)/0.01

        # dist_vals_img_org_jac = (np.mean(K_value11 * dist_vals_img_org_new1) - np.mean(K_value22 * dist_vals_img_org_new2)) / (r_temp * 2)

        external_energy_central_jac = np.zeros([3, dist_vals_img_org.shape[0], dist_vals_img_org.shape[1]])
        external_energy_central_jac[0] = dist_vals_grad[0] * K_value
        external_energy_central_jac[1] = dist_vals_grad[1] * K_value
        external_energy_central_jac[2] = dist_vals_grad[2] * K_value

        central_jac = np.zeros([3, dist_vals_img_org.shape[0], dist_vals_img_org.shape[1]])
        for i in range(dist_vals_img_org.shape[0]):
            for j in range(dist_vals_img_org.shape[1]):
                central_jac[0, i, j] = external_energy_central_jac[:, i, j] @ u
                central_jac[1, i, j] = external_energy_central_jac[:, i, j] @ v
        central_jac[2] = K_value_jac * dist_vals_img_org

        #dist_vals_img_org_jac * K_value  # + 3* dist_vals_img_org_jac * K_value
        # central_jac[2] = K_value * dist_vals_img_org_jac

        # a = dist_vals_img_org * K_value_jac
        # b = dist_vals_img_org * K_value_jac + dist_vals_img_org_jac * K_value
        # central_jac_new = np.mean(K_value_jac * dist_vals_img_org) + np.mean(K_value * (dist_vals_img_org - dist_vals_img_org_new))*10
        # print(np.mean(K_value_jac * dist_vals_img_org))
        # print(np.mean(K_value * (dist_vals_img_org - dist_vals_img_org_new))*10)
        #
        der[circle_num][0] += - alpha * (1 - lambda_) * np.mean(central_jac[0]) / pts.shape[0]
        der[circle_num][1] += - alpha * (1 - lambda_) * np.mean(central_jac[1]) / pts.shape[0]
        der[circle_num][2] += - alpha * (1 - lambda_) * np.mean(central_jac[2]) / pts.shape[0]

        central_jac_x = -  np.mean(central_jac[0]) / pts.shape[0]
        central_jac_y = -  np.mean(central_jac[1]) / pts.shape[0]
        central_jac_r = -  np.mean(central_jac[2]) / pts.shape[0]

        # central_jac_r = -  dist_vals_img_org_jac / pts.shape[0]
        # central_jac_r = - (np.mean(central_jac[2]) + dist_vals_img_org_jac) / pts.shape[0]

        # central_jac_r_a = - np.mean(a) / pts.shape[0]
        # central_jac_r_b = - np.mean(b) / pts.shape[0]
        # print(central_jac_r_a,central_jac_r_b)
        # central_jac_r = - np.mean(K_value_jac * dist_vals_img_org + K_value*(dist_vals_img_org_new-dist_vals_img_org)/r_temp) / pts.shape[0]

        # print(np.mean(K_value_jac * dist_vals_img_org))
        # print(np.mean(K_value*(dist_vals_img_org_new-dist_vals_img_org)/r_temp))

        # print('%d, %5f, %5f, %5f' % (circle_num, central_jac_x, central_jac_y, central_jac_r))
    # pause
    # pause
    # print("=====================int=======================")
    leaf_node = []

    for i in range(pts.shape[0]):
        pts_r = pts[i][2]
        node_A = int(neighbour[i][0])
        node_B = int(neighbour[i][1])

        # print(node_A,i,node_B)
        if node_A == -1:
            pos_B = pos_all[node_B]
            u_B = u_vector[node_B]
            v_B = v_vector[node_B]

            u_i = u_vector[i]
            v_i = v_vector[i]

            X_i = pos_all[i] + pts[i][0] * u_i + pts[i][1] * v_i
            X_i_b = pos_B + pts[node_B][0] * u_B + pts[node_B][1] * v_B

            der[i][0] += (1 - alpha) * beta * 1.5 * (X_i - X_i_b) @ u_vector[i] / pts.shape[0]
            der[i][1] += (1 - alpha) * beta * 1.5 * (X_i - X_i_b) @ v_vector[i] / pts.shape[0]
            der[i][2] += (1 - alpha) * (1 - beta) * 1.5 * (pts_r - pts[node_B][2]) / pts.shape[0]
            leaf_node.append(i)

        elif node_B == -1:
            pos_A = pos_all[node_A]
            u_A = u_vector[node_A]
            v_A = v_vector[node_A]

            u_i = u_vector[i]
            v_i = v_vector[i]

            X_i = pos_all[i] + pts[i][0] * u_i + pts[i][1] * v_i
            X_i_a = pos_A + pts[node_A][0] * u_A + pts[node_A][1] * v_A

            der[i][0] += (1 - alpha) * beta * 1.5 * (X_i - X_i_a) @ u_vector[i] / pts.shape[0]
            der[i][1] += (1 - alpha) * beta * 1.5 * (X_i - X_i_a) @ v_vector[i] / pts.shape[0]
            der[i][2] += (1 - alpha) * (1 - beta) * 1.5 * (pts_r - pts[node_A][2]) / pts.shape[0]
            leaf_node.append(i)

        else:
            node_A_A = int(neighbour[node_A][0])
            node_A_B = int(neighbour[node_A][1])
            node_B_A = int(neighbour[node_B][0])
            node_B_B = int(neighbour[node_B][1])
            # print(node_A_A,node_A, node_A_B)

            pos_A = pos_all[node_A]
            u_A = u_vector[node_A]
            v_A = v_vector[node_A]
            node_A_r = pts[node_A][2]
            pos_B = pos_all[node_B]
            u_B = u_vector[node_B]
            v_B = v_vector[node_B]
            node_B_r = pts[node_B][2]

            u_i = u_vector[i]
            v_i = v_vector[i]

            X_i = pos_all[i] + pts[i][0] * u_i + pts[i][1] * v_i
            X_i_a = pos_A + pts[node_A][0] * u_A + pts[node_A][1] * v_A
            X_i_b = pos_B + pts[node_B][0] * u_B + pts[node_B][1] * v_B

            if node_A_A == -1:
                der[i][0] += (1 - alpha) * (beta) * 0.5 * ((5 * X_i - 3 * X_i_a - 2 * X_i_b) @ u_i) / pts.shape[0]
                der[i][1] += (1 - alpha) * (beta) * 0.5 * ((5 * X_i - 3 * X_i_a - 2 * X_i_b) @ v_i) / pts.shape[0]
                der[i][2] += (1 - alpha) * (1 - beta) * 0.5 * (5 * pts[i][2] - 3 * node_A_r - 2 * node_B_r) / pts.shape[
                    0]
            elif node_B_B == -1:
                der[i][0] += (1 - alpha) * (beta) * 0.5 * ((5 * X_i - 2 * X_i_a - 3 * X_i_b) @ u_i) / pts.shape[0]
                der[i][1] += (1 - alpha) * (beta) * 0.5 * ((5 * X_i - 2 * X_i_a - 3 * X_i_b) @ v_i) / pts.shape[0]
                der[i][2] += (1 - alpha) * (1 - beta) * 0.5 * (5 * pts[i][2] - 2 * node_A_r - 3 * node_B_r) / pts.shape[
                    0]
            else:
                der[i][0] += (1 - alpha) * (beta) * 0.5 * ((2 * X_i - X_i_a - X_i_b) @ u_i) / pts.shape[0] * 2
                der[i][1] += (1 - alpha) * (beta) * 0.5 * ((2 * X_i - X_i_a - X_i_b) @ v_i) / pts.shape[0] * 2
                der[i][2] += (1 - alpha) * (1 - beta) * 0.5 * (2 * pts[i][2] - node_A_r - node_B_r) / pts.shape[
                    0] * 2  ### 与上下文有关

            # x =

            # print('%d, %5f, %5f, %5f' % (i, x, y, der[i][2]-der_ext[i][2]))
    # pause

    return der.ravel()



def fit_snake_3D(pts, img_org, img_grad, img_hess, pos, resample_tree_data, centerline_neighbour, u_vector, v_vector, nits=100):
    # pts[2][2] = 2


    # pts_xy = pts[:,0:2]
    # pts_r = pts[:,2]

    begin_time = time.time()
    # a = snake_energy_3D_jac(pts.ravel(), img_org=img_org, img_grad=img_grad, pos_all=pos, neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    end_time = time.time()
    # print(end_time - begin_time)


    begin_time = time.time()
    # b = snake_energy_3D_jac_int_ext(pts.ravel(), img_org=img_org, img_grad=img_grad, pos_all=pos, neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    end_time = time.time()
    # print(end_time - begin_time)

    begin_time = time.time()
    # c = snake_energy_3D_jac_test(pts.ravel(), img_org=img_org, img_grad=img_grad, img_grad_jac=img_hess, pos_all=pos,
    #                              neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    end_time = time.time()
    # print(end_time - begin_time)

    np.set_printoptions(suppress=True)
    # a_ = np.reshape(a, (int(len(a) / 3), 3))
    # b_ = np.reshape(b, (int(len(b) / 3), 3))
    # c_ = np.reshape(c, (int(len(c) / 3), 3))
    #
    # print("a")
    # print(a_)
    #
    # print("b")
    # print(b_)
    #
    # print("c")
    # print(c_)

    # pause

    # optimize
    cost_function = partial(snake_energy_3D_new, img_org=img_org, img_grad=img_grad, pos_all=pos,
                            neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    # cost_function_jac = partial(snake_energy_3D_jac, img_org=img_org, img_grad=img_grad, pos_all=pos,
    #                             neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    cost_function_jac = partial(snake_energy_3D_jac_int_ext, img_org=img_org, img_grad=img_grad, pos_all=pos, neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    # cost_function_jac = partial(snake_energy_3D_jac_test, img_org=img_org, img_grad=img_grad, img_grad_jac=img_hess, pos_all=pos, neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector)
    # options = { 'disp': True}
    # options['maxiter'] = nits  # FIXME: check convergence
    method = 'SLSQP'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' 'SLSQP' 'L-BFGS-B'

    begin_time = time.time()
    # res = optimize.minimize(cost_function, pts.ravel(), method='Nelder-Mead')
    res = optimize.minimize(cost_function, pts.ravel(), method='SLSQP', jac=cost_function_jac)
    # res = optimize.minimize(cost_function, pts.ravel(), method='BFGS', jac=cost_function_jac)
    # res = optimize.minimize(cost_function, pts.ravel(), method='BFGS', jac=cost_function_jac, options={'disp': True})
    end_time = time.time()

    # res = optimize.fmin_bfgs(cost_function, pts.ravel())
    # end_time = time.time()
    print(res)
    print('耗时 %f' % (end_time - begin_time))
    optimal_pts = np.reshape(res.x, (int(len(res.x) / 3), 3))

    return optimal_pts
    # return 0


def cal_3d_image_grad(img_org, dim):
    # dim = 0 1 2  z x y
    img_org = img_org.astype(np.float64)
    img_grad = np.zeros_like(img_org)

    z_shape, x_shape, y_shape = img_org.shape

    if dim == 0:
        img_grad[0, :, :] = img_org[1, :, :] - img_org[0, :, :]
        for i in range(1, z_shape - 1):
            img_grad[i, :, :] = (img_org[i + 1, :, :] - img_org[i - 1, :, :]) / 2
        img_grad[z_shape - 1, :, :] = img_org[z_shape - 1, :, :] - img_org[z_shape - 2, :, :]
    if dim == 1:
        img_grad[:, 0, :] = img_org[:, 1, :] - img_org[:, 0, :]
        for i in range(1, x_shape - 1):
            img_grad[:, i, :] = (img_org[:, i + 1, :] - img_org[:, i - 1, :]) / 2
        img_grad[:, x_shape - 1, :] = img_org[:, x_shape - 1, :] - img_org[:, x_shape - 2, :]
    if dim == 2:
        img_grad[:, :, 0] = img_org[:, :, 1] - img_org[:, :, 0]
        for i in range(y_shape - 1):
            img_grad[:, :, i] = (img_org[:, :, i + 1] - img_org[:, :, i - 1]) / 2
        img_grad[:, :, y_shape - 1] = img_org[:, :, y_shape - 1] - img_org[:, :, y_shape - 2]
    return img_grad


if __name__ == "__main__":
    first_run = False

    resample_tree = SwcTree_convert()
    # 原始图像z,x,y, centerline x,y,z
    # 载入图像
    # img_name = 'Image4'
    # img_name = 'sample'
    # img_name = '16'
    img_name = 'single_branch_noise_0'
    org_dir = 'data/single_branch/' + img_name + '.tif'
    skel_dir = 'data/single_branch/' + img_name + '.skl.tif'
    # resample_tree_data = resample_tree.load_matric('data/16/' + img_name + '.resample.swc')
    resample_tree_data = resample_tree.load_matric('data/single_branch/' + img_name + '.swc')
    image_color_dir = 'data/single_branch/' + img_name + '.color.tif'
    new_to_swc_dir = 'data/single_branch/' + img_name + '.new.swc'
    org_to_swc_dir = 'data/single_branch/' + img_name + '.org.swc'

    # img_name = '1'
    # org_dir = 'data/sample/' + img_name + '.tif'
    # skel_dir = 'data/sample/' + img_name + '.skl.tif'
    # resample_tree_data = resample_tree.load_matric('data/sample/' + img_name + '.swc')

    img_org = tifffile.imread(org_dir)  # /255
    img_skel = tifffile.imread(skel_dir)

    # 16bit
    # img_org[img_org > 500] = 500
    # img_org[img_org < 100] = 100
    # img_org = (img_org - 100) / (500 - 100) * 255
    # img_org = img_org.astype(np.uint8)

    # 求图像梯度
    img_grad = np.zeros([3, img_org.shape[0], img_org.shape[1], img_org.shape[2]])

    img_grad[0] = cal_3d_image_grad(img_org, 0)  # z方向导数
    img_grad[1] = cal_3d_image_grad(img_org, 1)  # x方向导数
    img_grad[2] = cal_3d_image_grad(img_org, 2)  # y方向导数


    # 求图像的二阶导
    img_hess = np.zeros([3, 3, img_org.shape[0], img_org.shape[1], img_org.shape[2]])
    img_hess[0][0] = cal_3d_image_grad(img_grad[0], 0)  # zz方向导数
    img_hess[1][0] = cal_3d_image_grad(img_grad[1], 0)  # xz方向导数
    img_hess[2][0] = cal_3d_image_grad(img_grad[2], 0)  # yz方向导数
    img_hess[0][1] = cal_3d_image_grad(img_grad[0], 1)  # zx方向导数
    img_hess[1][1] = cal_3d_image_grad(img_grad[1], 1)  # xx方向导数
    img_hess[2][1] = cal_3d_image_grad(img_grad[2], 1)  # yx方向导数
    img_hess[0][2] = cal_3d_image_grad(img_grad[0], 2)  # zy方向导数
    img_hess[1][2] = cal_3d_image_grad(img_grad[1], 2)  # xy方向导数
    img_hess[2][2] = cal_3d_image_grad(img_grad[2], 2)  # yy方向导数

    img_org = img_org.astype(np.float64)
    img_grad = img_grad.astype(np.float64)
    img_hess = img_hess.astype(np.float64)


    # 获取原始轮廓参数方程
    print("获取centerline...")
    centerline_temp = np.where(img_skel == 1)

    centerline = np.zeros([len(centerline_temp[0]), 3], dtype=np.int)
    print("centerline的长度：", len(centerline_temp[0]))
    centerline[:, 0] = np.array(centerline_temp[0])
    centerline[:, 1] = np.array(centerline_temp[1])
    centerline[:, 2] = np.array(centerline_temp[2])



    # print(resample_tree_data.shape)
    # pause

    centerline_sample_A = np.zeros([resample_tree_data.shape[0], 3])
    centerline_sample_A[:, 0] = copy.deepcopy(resample_tree_data[:, 4])
    centerline_sample_A[:, 1] = copy.deepcopy(resample_tree_data[:, 3])
    centerline_sample_A[:, 2] = copy.deepcopy(resample_tree_data[:, 2])
    # print(centerline_sample_A)

    centerline_direction, centerline_neighbour = get_centerline_direction_and_neighbour(resample_tree_data)
    # print(centerline_direction)

    #
    print("计算初始法平面")
    r = 4.0
    circle_z, circle_y, circle_x, circle_center, circle_angle, u_vector, v_vector = get_centerline_circle(
        centerline_sample_A, centerline_direction, r)
    # print(centerline_sample_A.shape)

    # print("test:")

    # print(centerline_sample_A.shape)
    # print(centerline_sample_branch.shape)
    # print(centerline_sample_leaf.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(centerline[:, 0], centerline[:, 2], centerline[:, 1])
    # ax.scatter(centerline_sample_A[:, 2], centerline_sample_A[:, 0], centerline_sample_A[:, 1], marker='o', c= 'green')

    # ax.scatter(centerline_sample_branch[:, 0], centerline_sample_branch[:, 2], centerline_sample_branch[:, 1], marker='.', c= 'green')
    # ax.scatter(centerline_sample_leaf[:, 0], centerline_sample_leaf[:, 2], centerline_sample_leaf[:, 1], marker='.', c= 'yellow')

    # 初始化优化值
    pos_r_opt = np.zeros([circle_x.shape[0], 3])
    r_temp_global = np.zeros([circle_x.shape[0]])
    for i in range(circle_x.shape[0]):
        pos_r_opt[i][0] = 1e-4 #random.uniform(0,0.1)
        pos_r_opt[i][1] = 1e-4 #random.uniform(0,0.1)
        if i%2==0:
            pos_r_opt[i][2] = 5 # random.uniform(3,5)
        else:
            pos_r_opt[i][2] = 5  # random.uniform(3,5)
        # pos_r_opt[i][0] = random.uniform(0, 0.1)
        # pos_r_opt[i][1] = random.uniform(0, 0.1)
        # pos_r_opt[i][2] = random.uniform(3, 5)
    print(pos_r_opt[0][2])

    # 将初始化的swc保存
    centerline_org_to_swc = np.zeros([resample_tree_data.shape[0], 7])
    for i in range(resample_tree_data.shape[0]):
        centerline_org_to_swc[i][0] = resample_tree_data[i][0]
        centerline_org_to_swc[i][1] = resample_tree_data[i][1]
        centerline_org_to_swc[i][2] = resample_tree_data[i][2]
        centerline_org_to_swc[i][3] = resample_tree_data[i][3]
        centerline_org_to_swc[i][4] = resample_tree_data[i][4]
        centerline_org_to_swc[i][5] = pos_r_opt[i][2]
        centerline_org_to_swc[i][6] = resample_tree_data[i][6]
    swc_save_lc(org_to_swc_dir, centerline_org_to_swc)


    # 优化
    snake_opt = fit_snake_3D(pos_r_opt, img_org, img_grad, img_hess, centerline_sample_A, resample_tree_data,
                             centerline_neighbour, u_vector, v_vector, nits=100)
    print(snake_opt)


    # 绘制优化前后的结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(centerline[:, 0], centerline[:, 2], centerline[:, 1], marker='.')
    ax.plot(centerline[:, 0], centerline[:, 2], centerline[:, 1])
    # for i in range(circle_x.shape[0]):
    #     ax.plot(circle_z[i, :], circle_x[i, :], circle_y[i, :], color='salmon')

    # image_color = image_with_circle(img_org, circle_x, circle_y, circle_z, color=0, init=True)
    centerline_new = np.zeros([circle_x.shape[0], 4])
    for i in range(circle_x.shape[0]):
        pos_org = np.zeros([1, 3])
        pos_new = np.zeros([1, 3])

        pos_org[0][0] = centerline_sample_A[i][0]
        pos_org[0][1] = centerline_sample_A[i][1]
        pos_org[0][2] = centerline_sample_A[i][2]

        pos_new[0][0] = centerline_sample_A[i][0] + snake_opt[i][0] * u_vector[i][0] + snake_opt[i][1] * v_vector[i][0]
        pos_new[0][1] = centerline_sample_A[i][1] + snake_opt[i][0] * u_vector[i][1] + snake_opt[i][1] * v_vector[i][1]
        pos_new[0][2] = centerline_sample_A[i][2] + snake_opt[i][0] * u_vector[i][2] + snake_opt[i][1] * v_vector[i][2]

        centerline_direction_new = np.zeros([1, 3])

        centerline_direction_new[0][0] = centerline_direction[i][0]
        centerline_direction_new[0][1] = centerline_direction[i][1]
        centerline_direction_new[0][2] = centerline_direction[i][2]

        centerline_new[i][0] = pos_new[0][0]
        centerline_new[i][1] = pos_new[0][1]
        centerline_new[i][2] = pos_new[0][2]
        centerline_new[i][3] = snake_opt[i][2]


        circle_z_org, circle_y_org, circle_x_org, _, _, _, _ = get_centerline_circle(pos_org, centerline_direction_new, pos_r_opt[i][2])

        circle_z_temp, circle_y_temp, circle_x_temp, _, _, _, _ = get_centerline_circle(pos_new, centerline_direction_new, snake_opt[i][2])

        ax.plot(circle_z_org[0, :], circle_x_org[0, :], circle_y_org[0, :], color='salmon')
        ax.plot(circle_z_temp[0, :], circle_x_temp[0, :], circle_y_temp[0, :], color='red')
        # image_color = image_with_circle(image_color, circle_x_temp, circle_y_temp, circle_z_temp, color=1, init=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.xlim(-5, 60)
    plt.ylim(-5, 60)


    plt.show()

    # tifffile.imsave(image_color_dir, image_color)

    centerline_new_to_swc = np.zeros([resample_tree_data.shape[0], 7])
    for i in range(resample_tree_data.shape[0]):
        centerline_new_to_swc[i][0] = i + 1
        centerline_new_to_swc[i][1] = 0
        centerline_new_to_swc[i][2] = centerline_new[i][2]
        centerline_new_to_swc[i][3] = centerline_new[i][1]
        centerline_new_to_swc[i][4] = centerline_new[i][0]
        centerline_new_to_swc[i][5] = centerline_new[i][3]
        centerline_new_to_swc[i][6] = resample_tree_data[i][6]



    swc_save_lc(new_to_swc_dir, centerline_new_to_swc)



