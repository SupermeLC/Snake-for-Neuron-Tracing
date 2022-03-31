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
import os
import sys

abs_dir = os.path.abspath("")
sys.path.append(abs_dir)

from neuron_tracing.lib.pyneval.pyneval_io.swc_io import swc_save, read_swc_tree
from neuron_tracing.lib.swclib.SWC import *
import multiprocessing as mp
import random
from scipy.misc import derivative



energy_sum_list = []
energy_sum_list_int_ext = []
energy_sum_list_test = []

r_list_test = []

def get_centerline_circle(centerline_sample_A, centerline_direction, r):
    circle_num = centerline_sample_A.shape[0]

    theta = np.arange(0.001, 2 * np.pi, 1 / (r + 0.1))

    circle_x = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
    circle_y = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)
    circle_z = np.zeros([circle_num, theta.shape[0]], dtype=np.float32)

    circle_center = np.zeros([circle_num, 3], dtype=np.float32)
    circle_angle = np.zeros([circle_num, 2], dtype=np.float32)

    # 圆的横截面 R
    # circle_x_plane_index = np.zeros([circle_num, r * theta.shape[0]])
    # circle_y_plane_index = np.zeros([circle_num, r * theta.shape[0]])
    # circle_z_plane_index = np.zeros([circle_num, r * theta.shape[0]])

    u_vector = np.zeros([circle_num, 3], dtype=np.float32)
    v_vector = np.zeros([circle_num, 3], dtype=np.float32)

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
                    # print(temp_index, current_id)
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

    return centerline_tree

def get_centerline_direction_and_neighbour(resample_tree_data):
    centerline_direction = np.zeros([resample_tree_data.shape[0], 3], dtype=np.float32)
    centerline_neighbour = np.zeros([resample_tree_data.shape[0], 2], dtype=np.float32)
    for i in range(resample_tree_data.shape[0]):
        node_id = resample_tree_data[i][0]
        node_id_p = resample_tree_data[i][6]

        if node_id_p == -1: # parent 是否为root
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

        centerline_neighbour[i][0] = neigh_A
        centerline_neighbour[i][1] = neigh_B

    return centerline_direction, centerline_neighbour

def cal_3d_image_grad(img_org, dim):
    # dim = 0 1 2  z x y
    img_org = img_org.astype(np.float32)
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
    external_offset_energy = np.zeros([pts.shape[0]], dtype=np.float32)

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

        external_energy_offset = np.mean(dist_vals_grad * o) * 3
        external_offset_energy[circle_num] = external_energy_offset

    external_offset_energy_mean = np.sum(external_offset_energy)/ pts.shape[0]

    return external_offset_energy_mean

def snake_energy_3D_ext_Central(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    external_central_energy = np.zeros([pts.shape[0]], dtype=np.float32)
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

        X = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        Y = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        X_mesh, Y_mesh = np.meshgrid(X, Y)

        Z_new = pts_z + X_mesh * u[0] + Y_mesh * v[0]
        X_new = pts_x + X_mesh * u[1] + Y_mesh * v[1]
        Y_new = pts_y + X_mesh * u[2] + Y_mesh * v[2]

        dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Z_new, X_new, Y_new], order=1)  # Y, X, Z
        K_value = K_kernal(X_mesh, Y_mesh, pt_r)

        external_central_energy[circle_num] = np.mean(K_value * dist_vals_img_org)

    external_central_energy_mean = - np.sum(external_central_energy) / pts.shape[0]
    return external_central_energy_mean

def snake_energy_3D_int(flattened_pts, pos_all, neighbour, u_vector, v_vector, beta=0.3):
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
    internal_energy_mean = np.sum(internal_energy) / pts.shape[0]
    return internal_energy_mean

def snake_energy_3D(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector, energy_list, print_info = False, alpha=0.5, beta=0.3,lambda_=0.5):
    print("======================")
    print_time = False

    begin_time = time.time()
    external_energy_offset = snake_energy_3D_ext_Offset(flattened_pts, img_org, img_grad, pos_all, neighbour, u_vector, v_vector)
    end_time = time.time()
    if print_time:
        print('ext offset time: %f' % (end_time - begin_time))

    begin_time = time.time()
    external_energy_central = snake_energy_3D_ext_Central(flattened_pts, img_org, img_grad, pos_all, neighbour,u_vector, v_vector)
    end_time = time.time()
    if print_time:
        print('ext central time: %f' % (end_time - begin_time))

    begin_time = time.time()
    internal_energy = snake_energy_3D_int(flattened_pts, pos_all, neighbour, u_vector, v_vector, beta)
    end_time = time.time()
    if print_time:
        print('int time: %f' % (end_time - begin_time))

    energy_sum = alpha * (lambda_ * external_energy_offset + (1 - lambda_) * external_energy_central) + (1 - alpha) * internal_energy
    if print_info == True:
        pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
        r_list_test.append(np.mean(pts[:,2]))
        energy_list.append(energy_sum)
        print('offset: %2f, central: %2f, internal: %2f' % (alpha * lambda_ * external_energy_offset, alpha * (1 - lambda_) *external_energy_central, (1 - alpha) * internal_energy))

    return energy_sum*2

def snake_energy_3D_jac(flattened_pts, img_org, img_grad, img_grad_jac, pos_all, neighbour, u_vector, v_vector, alpha=0.5, beta=0.3, lambda_=0.5):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    der = np.zeros_like(pts)
    print_time = False
    np.set_printoptions(suppress=True)


    # =============================== offset jac ==========================================
    begin_time = time.time()
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

        theta = np.arange(1e-5, 2 * np.pi, 1 / pt_r + 1e-5)
        cl_z = pts_z + pt_r * u[0] * np.cos(theta) + pt_r * v[0] * np.sin(theta)
        cl_x = pts_x + pt_r * u[1] * np.cos(theta) + pt_r * v[1] * np.sin(theta)
        cl_y = pts_y + pt_r * u[2] * np.cos(theta) + pt_r * v[2] * np.sin(theta)

        o = np.zeros([3, theta.shape[0]])
        o[0] = u[0] * np.cos(theta) + v[0] * np.sin(theta)
        o[1] = u[1] * np.cos(theta) + v[1] * np.sin(theta)
        o[2] = u[2] * np.cos(theta) + v[2] * np.sin(theta)

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

    end_time = time.time()
    if print_time:
        print('ext offset jac time: %f' % (end_time - begin_time))

    # ===============================central jac ==========================================
    begin_time = time.time()
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

        X = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        Y = np.arange(-3.0 * pt_r, 3.0 * pt_r, 3 / np.pi + 1e-5)
        X_mesh, Y_mesh = np.meshgrid(X, Y)

        Z_new = pts_z + X_mesh * u[0] + Y_mesh * v[0]
        X_new = pts_x + X_mesh * u[1] + Y_mesh * v[1]
        Y_new = pts_y + X_mesh * u[2] + Y_mesh * v[2]

        dist_vals_grad = np.zeros([3, X_mesh.shape[0], X_mesh.shape[1]], dtype=np.float32)
        dist_vals_grad[0] = ndimage.interpolation.map_coordinates(img_grad[0], [Z_new, X_new, Y_new], order=1) # Y, X, Z
        dist_vals_grad[1] = ndimage.interpolation.map_coordinates(img_grad[1], [Z_new, X_new, Y_new], order=1)
        dist_vals_grad[2] = ndimage.interpolation.map_coordinates(img_grad[2], [Z_new, X_new, Y_new], order=1)
        dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Z_new, X_new, Y_new], order=1)

        K_value = K_kernal(X_mesh, Y_mesh, pt_r) #* K_value_mask
        K_value_jac = K_kernal_jac(X_mesh, Y_mesh, pt_r) #* K_value_mask

        external_energy_central_jac = np.zeros([3, dist_vals_img_org.shape[0], dist_vals_img_org.shape[1]])
        external_energy_central_jac[0] = dist_vals_grad[0] * K_value
        external_energy_central_jac[1] = dist_vals_grad[1] * K_value
        external_energy_central_jac[2] = dist_vals_grad[2] * K_value

        central_jac = np.zeros([3, dist_vals_img_org.shape[0], dist_vals_img_org.shape[1]])
        for i in range(dist_vals_img_org.shape[0]):
            for j in range(dist_vals_img_org.shape[1]):
                central_jac[0, i, j] = external_energy_central_jac[:, i, j] @ u
                central_jac[1, i, j] = external_energy_central_jac[:, i, j] @ v
        central_jac[2] = dist_vals_img_org * K_value_jac

        der[circle_num][0] += - alpha * (1 - lambda_) * np.mean(central_jac[0]) / pts.shape[0]
        der[circle_num][1] += - alpha * (1 - lambda_) * np.mean(central_jac[1]) / pts.shape[0]
        der[circle_num][2] += - alpha * (1 - lambda_) * np.mean(central_jac[2]) / pts.shape[0]

    end_time = time.time()
    if print_time:
        print('ext central jac time: %f' % (end_time - begin_time))

    # ==============================internal jac ==========================================
    begin_time = time.time()
    leaf_node = []
    for i in range(pts.shape[0]):
        pts_r = pts[i][2]
        node_A = int(neighbour[i][0])
        node_B = int(neighbour[i][1])

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
            node_B_B = int(neighbour[node_B][1])

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
                der[i][2] += (1 - alpha) * (1 - beta) * 0.5 * (5 * pts[i][2] - 3 * node_A_r - 2 * node_B_r) / pts.shape[0]

            elif node_B_B == -1:
                der[i][0] += (1 - alpha) * (beta) * 0.5 * ((5 * X_i - 2 * X_i_a - 3 * X_i_b) @ u_i) / pts.shape[0]
                der[i][1] += (1 - alpha) * (beta) * 0.5 * ((5 * X_i - 2 * X_i_a - 3 * X_i_b) @ v_i) / pts.shape[0]
                der[i][2] += (1 - alpha) * (1 - beta) * 0.5 * (5 * pts[i][2] - 2 * node_A_r - 3 * node_B_r) / pts.shape[0]

            else:
                der[i][0] += (1 - alpha) * (beta) * 0.5 * ((2 * X_i - X_i_a - X_i_b) @ u_i) * 2 / pts.shape[0]
                der[i][1] += (1 - alpha) * (beta) * 0.5 * ((2 * X_i - X_i_a - X_i_b) @ v_i) * 2 / pts.shape[0]
                der[i][2] += (1 - alpha) * (1 - beta) * 0.5 * (2 * pts[i][2] - node_A_r - node_B_r) * 2 / pts.shape[0] ### 与上下文有关


    end_time = time.time()
    if print_time:
        print('int jac time: %f' % (end_time - begin_time))

    return der.ravel()



# 限制r大于max_r
def con(max_r, min_r, arg_num):
    r_cons = []

    for i in range(arg_num):
        arg_num_temp = i * 3 + 2
        dict1 = {'type': 'ineq', 'fun': lambda x, arg_num_temp = arg_num_temp: x[arg_num_temp] - min_r}
        r_cons.append(copy.deepcopy(dict1))
        dict1 = {'type': 'ineq', 'fun': lambda x, arg_num_temp=arg_num_temp: max_r - x[arg_num_temp]}
        r_cons.append(copy.deepcopy(dict1))

    return r_cons

def fit_snake_3D(pts, img_org, img_grad, img_hess, pos, centerline_neighbour, u_vector, v_vector, nits=100, alpha=0.5,beta=0.3,lambda_=0.5,show_result=False):
    # optimize set
    min_r = 0.6
    max_r = 5.0
    cons = con(max_r, min_r, pts.shape[0])


    begin_time_3 = time.time()
    cost_function = partial(snake_energy_3D, img_org=img_org, img_grad=img_grad, pos_all=pos,
                            neighbour=centerline_neighbour, u_vector=u_vector, v_vector=v_vector, energy_list=energy_sum_list_test, print_info=True, alpha=alpha, beta=beta, lambda_=lambda_)
    cost_function_jac = partial(snake_energy_3D_jac, img_org=img_org, img_grad=img_grad, img_grad_jac=img_hess, pos_all=pos, neighbour=centerline_neighbour,
                                     u_vector=u_vector, v_vector=v_vector, alpha=alpha, beta=beta, lambda_=lambda_)
    res_3 = optimize.minimize(cost_function, pts.ravel(), method='SLSQP', jac=cost_function_jac, constraints=cons, options={'disp': True, 'maxiter': nits, 'ftol': 1e-6})
    end_time_3 = time.time()
    print(res_3)


    time_3 = end_time_3 - begin_time_3
    print('理论梯度耗时 %f, 迭代步数： %d' % (time_3, len(energy_sum_list_test)))

    x_range_test = np.arange(0, time_3, time_3/len(energy_sum_list_test))

    if show_result:
        plt.figure()
        plt.plot(x_range_test, energy_sum_list_test, marker='.', color='b', label='v3')
        plt.xlabel(u'time', fontsize=14)
        plt.ylabel(u'loss', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()


    optimal_pts = np.reshape(res_3.x, (int(len(res_3.x) / 3), 3))
    return optimal_pts


def snake_3D(org_image_dir,swc_dir,swc_new_dir, alpha=0.5,beta=0.3,lambda_=0.5,nits=100,show_result=False):# lambda_小 central大
    # 载入图像
    resample_tree = SwcTree_convert()
    resample_tree_data = resample_tree.load_matric(swc_dir)  # .neutube
    img_org = tifffile.imread(org_image_dir)  # /255


    # 16bit
    img_org[img_org > 500] = 500
    img_org[img_org < 100] = 100
    img_org = (img_org - 100) / (500 - 100) * 255
    img_org = img_org.astype(np.uint8)

    # 求图像一阶导梯度
    img_grad = np.zeros([3, img_org.shape[0], img_org.shape[1], img_org.shape[2]], dtype=np.float32)
    img_grad[0] = cal_3d_image_grad(img_org, 0)  # z方向导数
    img_grad[1] = cal_3d_image_grad(img_org, 1)  # x方向导数
    img_grad[2] = cal_3d_image_grad(img_org, 2)  # y方向导数

    # 求图像的二阶导
    img_hess = np.zeros([3, 3, img_org.shape[0], img_org.shape[1], img_org.shape[2]], dtype=np.float32)
    img_hess[0][0] = cal_3d_image_grad(img_grad[0], 0)  # zz方向导数
    img_hess[1][0] = cal_3d_image_grad(img_grad[1], 0)  # xz方向导数
    img_hess[2][0] = cal_3d_image_grad(img_grad[2], 0)  # yz方向导数
    img_hess[0][1] = cal_3d_image_grad(img_grad[0], 1)  # zx方向导数
    img_hess[1][1] = cal_3d_image_grad(img_grad[1], 1)  # xx方向导数
    img_hess[2][1] = cal_3d_image_grad(img_grad[2], 1)  # yx方向导数
    img_hess[0][2] = cal_3d_image_grad(img_grad[0], 2)  # zy方向导数
    img_hess[1][2] = cal_3d_image_grad(img_grad[1], 2)  # xy方向导数
    img_hess[2][2] = cal_3d_image_grad(img_grad[2], 2)  # yy方向导数

    img_org = img_org.astype(np.float32)
    img_grad = img_grad.astype(np.float32)
    img_hess = img_hess.astype(np.float32)

    #计算枝干方向
    centerline_sample_A = np.zeros([resample_tree_data.shape[0], 3], dtype=np.float32)
    centerline_sample_A[:, 0] = copy.deepcopy(resample_tree_data[:, 4])
    centerline_sample_A[:, 1] = copy.deepcopy(resample_tree_data[:, 3])
    centerline_sample_A[:, 2] = copy.deepcopy(resample_tree_data[:, 2])
    centerline_direction, centerline_neighbour = get_centerline_direction_and_neighbour(resample_tree_data)

    # 计算初始法平面（可视化）
    circle_z, circle_y, circle_x, circle_center, circle_angle, u_vector, v_vector = get_centerline_circle(
        centerline_sample_A, centerline_direction, r=5)

    # 初始化参数
    pos_r_opt = np.zeros([circle_x.shape[0], 3], dtype=np.float32)
    pos_new_temp = np.zeros([circle_x.shape[0], 3])
    for i in range(circle_x.shape[0]):
        pos_r_opt[i][0] = random.uniform(-0.5, 0.5)
        pos_r_opt[i][1] = random.uniform(-0.5, 0.5)
        pos_new_temp[i][2] = centerline_sample_A[i][0] + pos_r_opt[i][0] * u_vector[i][0] + pos_r_opt[i][1] * \
                             v_vector[i][0]
        pos_new_temp[i][1] = centerline_sample_A[i][1] + pos_r_opt[i][0] * u_vector[i][1] + pos_r_opt[i][1] * \
                             v_vector[i][1]
        pos_new_temp[i][0] = centerline_sample_A[i][2] + pos_r_opt[i][0] * u_vector[i][2] + pos_r_opt[i][1] * \
                             v_vector[i][2]


        pos_r_opt[i][2] = resample_tree_data[i][5] + random.uniform(-0.5, 0.5)
        if pos_r_opt[i][2] < 0:
            pos_r_opt[i][2] =0.5


    # 保存初始化的swc
    # centerline_org_to_swc = np.zeros([resample_tree_data.shape[0], 7])
    # for i in range(resample_tree_data.shape[0]):
    #     centerline_org_to_swc[i][0] = resample_tree_data[i][0]
    #     centerline_org_to_swc[i][1] = resample_tree_data[i][1]
    #     centerline_org_to_swc[i][2] = pos_new_temp[i][0]
    #     centerline_org_to_swc[i][3] = pos_new_temp[i][1]
    #     centerline_org_to_swc[i][4] = pos_new_temp[i][2]
    #     centerline_org_to_swc[i][5] = pos_r_opt[i][2]
    #     centerline_org_to_swc[i][6] = resample_tree_data[i][6]
    # swc_save_lc(org_to_swc_dir, centerline_org_to_swc)


    # 半径与位置优化
    snake_opt = fit_snake_3D(pos_r_opt, img_org, img_grad, img_hess, centerline_sample_A,
                             centerline_neighbour, u_vector, v_vector, nits, alpha, beta, lambda_, show_result)
    # 输出优化结果
    # print(snake_opt)

    # 新的坐标点与半径
    centerline_new = np.zeros([circle_x.shape[0], 4])
    for i in range(circle_x.shape[0]):
        pos_org = np.zeros([1, 3])

        pos_org[0][0] = centerline_sample_A[i][0]
        pos_org[0][1] = centerline_sample_A[i][1]
        pos_org[0][2] = centerline_sample_A[i][2]

        centerline_new[i][0] = centerline_sample_A[i][0] + snake_opt[i][0] * u_vector[i][0] + snake_opt[i][1] * \
                               v_vector[i][0]
        centerline_new[i][1] = centerline_sample_A[i][1] + snake_opt[i][0] * u_vector[i][1] + snake_opt[i][1] * \
                               v_vector[i][1]
        centerline_new[i][2] = centerline_sample_A[i][2] + snake_opt[i][0] * u_vector[i][2] + snake_opt[i][1] * \
                               v_vector[i][2]
        centerline_new[i][3] = snake_opt[i][2]

    # 保存为swc
    centerline_new_to_swc = np.zeros([resample_tree_data.shape[0], 7])
    for i in range(resample_tree_data.shape[0]):
        centerline_new_to_swc[i][0] = i + 1
        centerline_new_to_swc[i][1] = 0
        centerline_new_to_swc[i][2] = centerline_new[i][2]
        centerline_new_to_swc[i][3] = centerline_new[i][1]
        centerline_new_to_swc[i][4] = centerline_new[i][0]
        centerline_new_to_swc[i][5] = centerline_new[i][3]
        centerline_new_to_swc[i][6] = resample_tree_data[i][6]
    swc_save_lc(swc_new_dir, centerline_new_to_swc)



if __name__ == "__main__":
    org_image_dir = 'data/single_branch/noise_image_70/noise_image_70.tif'
    swc_dir = 'data/single_branch/noise_image_70/noise_image_70.swc'
    swc_new_dir = 'data/single_branch/noise_image_70/noise_image_70.new.swc'

    # lambda_小 central大
    snake_3D(org_image_dir, swc_dir, swc_new_dir, alpha=0.5, beta=0.3, lambda_=0.6, nits=30, show_result=True)

# python neuron_tracing/model/snake_3D.py


