import cv2 as cv
import tifffile
import numpy as np
import glob
from skimage import morphology, filters
from matplotlib import pyplot as plt
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
import random

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """
    以参数方程的形式，获取n个离散点围成的圆形/椭圆形轮廓
    输入：中心centre=（x0, y0）, 半轴长radius=(a, b)， 离散点数N
    输出：由离散点坐标(x, y)组成的2xN矩阵
    """
    t = np.linspace(0, 2 * np.pi, N)
    # noise = np.random.randint(1, 5, (t.shape[0],))
    noise = 0
    x = centre[0] + radius[0] * np.cos(t) + noise
    y = centre[1] + radius[1] * np.sin(t) + noise

    # x[x < 0] = 0
    # y[y < 0] = 0
    # x[x > 39] = 39
    # y[y > 39] = 39

    return np.array([x, y])


def o_distance(point_A, point_B):
    distance = math.sqrt((point_A[0][0] - point_B[0][0]) ** 2 + (point_A[0][1] - point_B[0][1]) ** 2 + (
                point_A[0][2] - point_B[0][2]) ** 2)
    return distance

# 不重要
# 初始化：获取在当前点，对应方向的圆  （一圈点）
def get_centerline_circle_2D(centerline_x, centerline_y, r):
    theta = np.arange(0.001, 2 * np.pi, 1 / r)

    circle_x = np.zeros([theta.shape[0], 1])
    circle_y = np.zeros([theta.shape[0], 1])

    for i in range(theta.shape[0]):
        circle_x[i][0] = centerline_x + r * np.sin(theta[i])
        circle_y[i][0] = centerline_y + r * np.cos(theta[i])

    return circle_x, circle_y

# x,y,r 的关系
def R_kernal(x_pos, y_pos, radius):
    return (x_pos ** 2 + y_pos ** 2) / radius ** 2

# K kernel的表达式
def K_kernal(x_pos, y_pos, radius):
    result = (2 - R_kernal(x_pos, y_pos, radius)) * np.exp(-R_kernal(x_pos, y_pos, radius) / 2) #/ (radius**2)
    return result

# K kernel的导数的表达式
def K_kernal_jac(x_pos, y_pos, radius):
    result = (4* R_kernal(x_pos, y_pos, radius) - R_kernal(x_pos, y_pos, radius)**2) * np.exp(-R_kernal(x_pos, y_pos, radius) / 2) / radius
    return result

# 不重要
# 外部能量项的 offset项，确认无问题
def snake_energy_2D_offset(flattened_pts, img_org, img_grad):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    pts_x = pts[0][0]
    pts_y = pts[0][1]
    pts_r = pts[0][2]

    # 新的参数方程
    theta = np.arange(1e-5, 2 * np.pi, 1 / pts_r + 1e-5)
    # print(theta.shape)
    cl_x = np.zeros([theta.shape[0], 1])
    cl_y = np.zeros([theta.shape[0], 1])

    cl_x[:, 0] = pts_x + pts_r * np.sin(theta)
    cl_y[:, 0] = pts_y + pts_r * np.cos(theta)
    # print(cl_x)


    # 外部能量项 offset
    dist_vals_x = ndimage.interpolation.map_coordinates(img_grad[0], [cl_y, cl_x], order=1)
    dist_vals_y = ndimage.interpolation.map_coordinates(img_grad[1], [cl_y, cl_x], order=1)



    o_x = np.sin(theta)
    o_y = np.cos(theta)
    # print("==x and y==")
    # print(cl_x, cl_y)
    # print(dist_vals_x, dist_vals_y)
    # print(o_x,o_y)
    # print(o_y)
    # print(dist_vals_x[:, 0] * o_x )
    # print(dist_vals_y[:, 0] * o_y)
    # print((theta.shape[0] + 1e-5))
    # pause

    external_energy_offset = np.sum(dist_vals_x[:, 0] * o_x + dist_vals_y[:, 0] * o_y) / (theta.shape[0] + 1e-5)
    # print(external_energy_offset)
    # pause

    return external_energy_offset


# 外部能量项的central项  主要看这个
def snake_energy_2D_central(flattened_pts, img_org, img_grad):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    pts_x = pts[0][0]
    pts_y = pts[0][1]
    pts_r = pts[0][2]

    # X = np.arange(- 3 * pts_r,  3 * pts_r, pts_r / 6 + 1e-5)
    # Y = np.arange(- 3 * pts_r,  3 * pts_r, pts_r / 6 + 1e-5)

    X = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    Y = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)

    X_mesh, Y_mesh = np.meshgrid(X, Y)
    # print(X_mesh.shape)

    X_new = pts_x + X_mesh
    Y_new = pts_y + Y_mesh

    # X_new = np.arange(pts_x - 3 * pts_r, pts_x + 3 * pts_r, pts_r / 6)
    # Y_new = np.arange(pts_y - 3 * pts_r, pts_y + 3 * pts_r, pts_r / 6)
    # X_new_mesh, Y_new_mesh = np.meshgrid(X_new, Y_new)


    # print(X_new)
    # pause
    dist_vals_img_org_new_mesh = ndimage.interpolation.map_coordinates(img_org, [Y_new, X_new], order=1)
    dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Y_mesh, X_mesh], order=1)


    K_value = K_kernal(X_mesh, Y_mesh, pts_r)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,3,1, projection='3d')
    # ax.scatter(X_mesh, Y_mesh, K_value, s=1,c='g')
    # ax = fig.add_subplot(1,3,2, projection='3d')
    # ax.scatter(X_mesh, Y_mesh, dist_vals_img_org, s=1,c='g')
    #
    # ax = fig.add_subplot(1,3,3, projection='3d')
    # ax.scatter(X_mesh, Y_mesh, dist_vals_img_org_new_mesh, s=1,c='g')
    # plt.show()
    # pause

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            # 3R 范围内
            if math.sqrt((X_mesh[i][j]) ** 2 + (Y_mesh[i][j]) ** 2) > 3 * pts_r:
                K_value[i][j] = 0


    # external_energy_central_image = K_value * dist_vals_img_org

    external_energy_central = - np.mean(K_value * dist_vals_img_org_new_mesh)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # ax.scatter(X_mesh, Y_mesh, K_value, s=1, c='g')
    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # ax.scatter(X_mesh, Y_mesh, dist_vals_img_org_new_mesh, s=1, c='g')
    #
    #
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # print(dist_vals_img_org_new_mesh.shape)
    # print(K_value.shape)
    # print(external_energy_central)
    # pause

    return external_energy_central

# 整体的能量项函数
def snake_energy_2D_new(flattened_pts, img_org, img_grad):
    lambda_ = 0.3

    external_energy_offset = snake_energy_2D_offset(flattened_pts, img_org, img_grad)
    # external_energy_offset =0
    external_energy_central = snake_energy_2D_central(flattened_pts, img_org, img_grad)

    external_energy = lambda_ * external_energy_offset + (1 - lambda_) * external_energy_central
    print('Offset: %f  Central: %f  ' % ((lambda_ * external_energy_offset), ((1 - lambda_) * external_energy_central)))


    # pause
    return external_energy


# 整体差分得到的梯度
def snake_energy_2D_jac(flattened_pts, img_org, img_grad):
    der = np.zeros_like(flattened_pts)
    # der_temp = 0.5
    der_temp = 1e-5
    for i in range(flattened_pts.shape[0]):
        x_new_big = copy.deepcopy(flattened_pts)
        x_new_small = copy.deepcopy(flattened_pts)
        x_new_big[i] += der_temp
        x_new_small[i] -= der_temp

        x_new_big_value = snake_energy_2D_new(x_new_big.ravel(), img_org=img_org, img_grad=img_grad)
        x_new_small_value = snake_energy_2D_new(x_new_small.ravel(), img_org=img_org, img_grad=img_grad)
        der[i] = (x_new_big_value - x_new_small_value) / (2 * der_temp)
    return der

# 对每项分别插值得到的梯度
def snake_energy_2D_ext_offset_jac(flattened_pts, img_org, img_grad):
    der = np.zeros_like(flattened_pts)
    der_temp = 0.5
    lambda_ = 0.3

    for i in range(flattened_pts.shape[0]):
        x_new_big = copy.deepcopy(flattened_pts)
        x_new_small = copy.deepcopy(flattened_pts)
        x_new_big[i] += der_temp
        x_new_small[i] -= der_temp

        x_new_big_value = snake_energy_2D_offset(x_new_big.ravel(), img_org=img_org, img_grad=img_grad)
        x_new_small_value = snake_energy_2D_offset(x_new_small.ravel(), img_org=img_org, img_grad=img_grad)
        der[i] += lambda_ * (x_new_big_value - x_new_small_value) / (2 * der_temp)

        x_new_big_value = snake_energy_2D_central(x_new_big.ravel(), img_org=img_org, img_grad=img_grad)
        x_new_small_value = snake_energy_2D_central(x_new_small.ravel(), img_org=img_org, img_grad=img_grad)
        der[i] += (1-lambda_) * (x_new_big_value - x_new_small_value) / (2 * der_temp)

    return der

# 重要
# 理论计算的梯度
def snake_energy_2D_jac_test(flattened_pts, img_org, img_grad, img_jac):
    np.set_printoptions(suppress=True)
    der = np.zeros_like(flattened_pts)
    der_test = np.zeros_like(flattened_pts)
    der_temp = 1e-5
    lambda_ = 0.3
    # print(flattened_pts)
    # for i in range(flattened_pts.shape[0]):
    #     x_new_big = copy.deepcopy(flattened_pts)
    #     x_new_small = copy.deepcopy(flattened_pts)
    #     x_new_big[i] += der_temp
    #     x_new_small[i] -= der_temp
    #
    #     x_new_big_value = snake_energy_2D_central(x_new_big.ravel(), img_org=img_org, img_grad=img_grad)
    #     x_new_small_value = snake_energy_2D_central(x_new_small.ravel(), img_org=img_org, img_grad=img_grad)
        # print(x_new_big_value,x_new_small_value)
    #     # print("===============================================")
    #     der_test[i] += (x_new_big_value - x_new_small_value) / (2 * der_temp)
    # print("==================  central jac  ========================")
    # print("dertest")
    # print(der_test)
    # print("===============================================")

    # print("================== offset test ========================")
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    pts_x = pts[0][0]
    pts_y = pts[0][1]
    pts_r = pts[0][2]

    # 新的参数方程
    theta = np.arange(0.01, 2 * np.pi, 1 / pts_r + 1e-5)

    cl_x = np.zeros([theta.shape[0], 1])
    cl_y = np.zeros([theta.shape[0], 1])
    cl_x[:, 0] = pts_x + pts_r * np.sin(theta)
    cl_y[:, 0] = pts_y + pts_r * np.cos(theta)

    dist_vals_jac = np.zeros([2, 2, theta.shape[0],1])
    dist_vals_jac[0][0] = ndimage.interpolation.map_coordinates(img_jac[0][0], [cl_y, cl_x], order=1)
    dist_vals_jac[1][0] = ndimage.interpolation.map_coordinates(img_jac[1][0], [cl_y, cl_x], order=1)
    dist_vals_jac[0][1] = ndimage.interpolation.map_coordinates(img_jac[0][1], [cl_y, cl_x], order=1)
    dist_vals_jac[1][1] = ndimage.interpolation.map_coordinates(img_jac[1][1], [cl_y, cl_x], order=1)

    dist_vals_jac = dist_vals_jac.astype(np.float64)

    o = np.zeros([2, theta.shape[0]])
    o[0] = np.sin(theta)
    o[1] = np.cos(theta)

    temp_jac = np.zeros([theta.shape[0], 2])
    for i in range(theta.shape[0]):
        temp_jac[i] = dist_vals_jac[:, :, i, 0] @ o[:,i]
    temp_jac_r = np.zeros([theta.shape[0]])
    for i in range(theta.shape[0]):
        temp_jac_r[i] = o[:,i] @ dist_vals_jac[:, :, i, 0] @ o[:,i]

    der[0] += lambda_ * np.mean(temp_jac[:,0])
    der[1] += lambda_ * np.mean(temp_jac[:,1])
    der[2] += lambda_ * np.mean(temp_jac_r)
    # print(x_jac, y_jac, r_jac)

    # print("================== central test ========================")
    # temp_r = 1e-5

    # X = np.arange(- 3 * pts_r, 3 * pts_r, pts_r / 6)
    # Y = np.arange(- 3 * pts_r, 3 * pts_r, pts_r / 6)

    X = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    Y = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    # X_1 = np.arange(-3.0 * (pts_r + temp_r), 3.0 * (pts_r + temp_r), 3 / np.pi + 1e-5)
    # Y_1 = np.arange(-3.0 * (pts_r + temp_r), 3.0 * (pts_r + temp_r), 3 / np.pi + 1e-5)
    # X_mesh_1, Y_mesh_1 = np.meshgrid(X_1, Y_1)
    #
    # X_2 = np.arange(-3.0 * (pts_r - temp_r), 3.0 * (pts_r - temp_r), 3 / np.pi + 1e-5)
    # Y_2 = np.arange(-3.0 * (pts_r - temp_r), 3.0 * (pts_r - temp_r), 3 / np.pi + 1e-5)
    # X_mesh_2, Y_mesh_2 = np.meshgrid(X_2, Y_2)

    #======================================================================================


    # X_new = np.arange(pts_x - 3 * pts_r, pts_x + 3 * pts_r, pts_r / 6)
    # Y_new = np.arange(pts_y - 3 * pts_r, pts_y + 3 * pts_r, pts_r / 6)
    # X_new_mesh, Y_new_mesh = np.meshgrid(X_new, Y_new)
    # X_new_mesh = copy.deepcopy(X_new_mesh[0:36, 0:36])
    # Y_new_mesh = copy.deepcopy(Y_new_mesh[0:36, 0:36])

    X_new_mesh = pts_x + X_mesh
    Y_new_mesh = pts_y + Y_mesh



    # X_new1 = np.arange(pts_x - 3 * (pts_r+temp_r), pts_x + 3 * (pts_r+temp_r), (pts_r+temp_r) / 6)
    # Y_new1 = np.arange(pts_y - 3 * (pts_r+temp_r), pts_y + 3 * (pts_r+temp_r), (pts_r+temp_r) / 6)
    # X_new1 = np.arange(pts_x - 3 * (pts_r + temp_r), pts_x + 3 * (pts_r + temp_r), 3 / np.pi + 1e-5)
    # Y_new1 = np.arange(pts_y - 3 * (pts_r + temp_r), pts_y + 3 * (pts_r + temp_r), 3 / np.pi + 1e-5)
    # X_new_mesh1, Y_new_mesh1 = np.meshgrid(X_new1, Y_new1)


    # X_new2 = np.arange(pts_x - 3 * (pts_r - temp_r), pts_x + 3 * (pts_r - temp_r), (pts_r - temp_r) / 6)
    # Y_new2 = np.arange(pts_y - 3 * (pts_r - temp_r), pts_y + 3 * (pts_r - temp_r), (pts_r - temp_r) / 6)
    # X_new2 = np.arange(pts_x - 3 * (pts_r - temp_r), pts_x + 3 * (pts_r - temp_r), 3 / np.pi + 1e-5)
    # Y_new2 = np.arange(pts_y - 3 * (pts_r - temp_r), pts_y + 3 * (pts_r - temp_r), 3 / np.pi + 1e-5)
    # X_new_mesh2, Y_new_mesh2 = np.meshgrid(X_new2, Y_new2)

    # print(X_new_mesh1[0])
    # print(X_new_mesh2[0])



    # img_org = img_org.astype(np.float64)
    # img_grad = img_grad.astype(np.float64)

    dist_vals_img_org_new_y = ndimage.interpolation.map_coordinates(img_grad[0], [Y_new_mesh, X_new_mesh], order=1)
    dist_vals_img_org_new_x = ndimage.interpolation.map_coordinates(img_grad[1], [Y_new_mesh, X_new_mesh], order=1)
    dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Y_new_mesh, X_new_mesh], order=1)
    # dist_vals_img_org1 = ndimage.interpolation.map_coordinates(img_org, [Y_new_mesh1, X_new_mesh1])
    # dist_vals_img_org2 = ndimage.interpolation.map_coordinates(img_org, [Y_new_mesh2, X_new_mesh2])

    # dist_vals_img_org_new_y = dist_vals_img_org_new_y.astype(np.float64)
    # dist_vals_img_org_new_x = dist_vals_img_org_new_x.astype(np.float64)
    # dist_vals_img_org = dist_vals_img_org.astype(np.float64)
    # dist_vals_img_org1 = dist_vals_img_org1.astype(np.float64)

    # print(dist_vals_img_org1.shape)
    # print(dist_vals_img_org2.shape)

    # dist_vals_img_org_jac = (dist_vals_img_org1 - dist_vals_img_org2)/(2*temp_r)


    K_value = K_kernal(X_mesh, Y_mesh, pts_r)
    K_value_jac = K_kernal_jac(X_mesh, Y_mesh, pts_r)

    # K_value_1 = K_kernal(X_mesh_1, Y_mesh_1, pts_r + temp_r)
    # K_value_2 = K_kernal(X_mesh_2, Y_mesh_2, pts_r - temp_r)
    # K_value_3 = K_kernal(X_mesh_2, Y_mesh_2, pts_r)

    # print(K_value_1.shape)
    # print(K_value_2.shape)
    #
    # K_value_jac_test = (K_value_1[1:-1,1:-1] - K_value_2)/(2*temp_r)

    # K_value_jac_test = (K_value_1[1:-1, 1:-1] - K_value_2) / (2 * temp_r)




    # r_jac_true = - np.mean((K_value_jac_new1*dist_vals_img_org1 - K_value_jac_new2*dist_vals_img_org2)/(2*temp_r))
    # print(-np.mean(K_value_jac_new1*dist_vals_img_org1), -np.mean(K_value_jac_new2*dist_vals_img_org2))



    # for i in range(X_mesh.shape[0]):
    #     for j in range(X_mesh.shape[1]):
    #         # 3R 范围内
    #         if math.sqrt((X_mesh[i][j]) ** 2 + (Y_mesh[i][j]) ** 2) > 3 * pts_r:
    #             K_value[i][j] = 0
                # K_value_jac_test[i][j] = 0
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # K_value_jac * dist_vals_img_org是正确的结果，但实际是K_value_jac * dist_vals_img_org + K_value * dist_vals_img_org_jac
    der[0] += - (1 - lambda_) * np.mean(K_value * dist_vals_img_org_new_y)
    der[1] += - (1 - lambda_) * np.mean(K_value * dist_vals_img_org_new_x)
    der[2] += - (1 - lambda_) * np.mean(K_value_jac * dist_vals_img_org)# + K_value * dist_vals_img_org_jac

    # x_jac = - np.mean(K_value * dist_vals_img_org_new_y)
    # y_jac = - np.mean(K_value * dist_vals_img_org_new_x)
    # r_jac = - np.mean(K_value_jac * dist_vals_img_org)
    #
    # r_1 = - np.mean(K_value * dist_vals_img_org_jac)
    # print(x_jac, y_jac, r_jac,r_1)
    # pause

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 7, 1)
    # plt.imshow(img_org, cmap=plt.cm.gray)
    # # plt.scatter(cl_x, cl_y, c='r')
    # # plt.scatter(X_mesh_new, Y_mesh_new, s=1, c='b')
    # plt.scatter(X_new_mesh, Y_new_mesh, s=1, c='r')
    #
    # ax = fig.add_subplot(1, 7, 2)
    # plt.imshow(img_grad[0, :, :], cmap=plt.cm.gray)
    # plt.scatter(cl_x, cl_y, c='r')
    # ax = fig.add_subplot(1, 7, 3)
    # plt.imshow(img_grad[1, :, :], cmap=plt.cm.gray)
    # # plt.scatter(cl_x, cl_y, c='r')
    # ax = fig.add_subplot(1, 7, 4)
    # plt.imshow(img_jac[0, 0, :, :], cmap=plt.cm.gray)
    # # plt.scatter(cl_x, cl_y, c='r')
    # ax = fig.add_subplot(1, 7, 5)
    # plt.imshow(img_jac[1, 0, :, :], cmap=plt.cm.gray)
    # # plt.scatter(cl_x, cl_y, c='r')
    # ax = fig.add_subplot(1, 7, 6)
    # plt.imshow(img_jac[0, 1, :, :], cmap=plt.cm.gray)
    # # plt.scatter(cl_x, cl_y, c='r')
    # ax = fig.add_subplot(1, 7, 7)
    # plt.imshow(img_jac[1, 1, :, :], cmap=plt.cm.gray)
    # # plt.scatter(cl_x, cl_y, c='r')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.scatter(X_mesh_2, Y_mesh_2, dist_vals_img_org_jac, s=1, c='b')
    # ax.scatter(X_mesh_2, Y_mesh_2, K_value_2, s=1, c='r')
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.scatter(X_mesh, Y_mesh, dist_vals_img_org, s=1,c='g')
    # ax = fig.add_subplot(1, 5, 3, projection='3d')
    # ax.scatter(X_new_mesh, Y_new_mesh, dist_vals_img_org-dist_vals_img_org1, s=1, c='g')
    # ax = fig.add_subplot(1, 5, 4, projection='3d')
    # ax.scatter(X_new_mesh, Y_new_mesh, dist_vals_img_org, s=1, c='g')
    # ax = fig.add_subplot(1, 5, 5, projection='3d')
    # ax.scatter(X_new_mesh1, Y_new_mesh1, dist_vals_img_org1, s=1, c='g')
    # plt.show()
    # pause

    return der

# 构建目标函数的优化方式
def fit_snake_2D(pts, img_org, img_grad, img_jac):


    # 检查初始点的梯度是否正常
    a = snake_energy_2D_jac(pts.ravel(), img_org=img_org, img_grad=img_grad)
    #
    b = snake_energy_2D_ext_offset_jac(pts.ravel(), img_org=img_org, img_grad=img_grad)
    #
    c = snake_energy_2D_jac_test(pts.ravel(), img_org=img_org, img_grad=img_grad, img_jac = img_jac)

    print(a)
    print(b)
    print(c)
    #
    # pause





    # optimize

    begin_time = time.time()
    cost_function = partial(snake_energy_2D_new, img_org=img_org, img_grad=img_grad)
    # cost_function_jac = partial(snake_energy_2D_jac, img_org=img_org, img_grad=img_grad)
    # cost_function_jac = partial(snake_energy_2D_ext_offset_jac, img_org=img_org, img_grad=img_grad)
    cost_function_jac = partial(snake_energy_2D_jac_test, img_org=img_org, img_grad=img_grad, img_jac = img_jac)
    # options['disp'] =  True
    # options['maxiter'] = nits  # FIXME: check convergence
    method = 'SLSQP'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' has very slow convergence
    # res = optimize.minimize(cost_function, pts.ravel(), method='Nelder-Mead', options={'xtol': 1e-2, 'disp': True})
    res = optimize.minimize(cost_function, pts.ravel(), method='SLSQP', jac=cost_function_jac, options={'disp': True})
    # res = optimize.minimize(cost_function, pts.ravel(), method='SLSQP', jac=cost_function_jac,options={'disp': True})
    # res = optimize.fmin_bfgs(cost_function, pts.ravel())

    end_time = time.time()
    print(res)
    print(res.fun)
    print('共耗时： %f' % (end_time-begin_time))
    optimal_pts = np.reshape(res.x, (int(len(res.x) / 3), 3))

    return optimal_pts



def cal_2d_image_grad(img_org, dim):
    # dim = 0 1 2  z x y
    img_org = img_org.astype(np.int16)
    img_grad = np.zeros_like(img_org)

    x_shape, y_shape = img_org.shape

    if dim == 1:
        img_grad[0, :] = img_org[1, :] - img_org[0, :]
        for i in range(1, x_shape-1):
            img_grad[i,:] = (img_org[i+1,:] - img_org[i-1,:])/2
        img_grad[x_shape-1,:] = img_org[x_shape-1,:] - img_org[x_shape-2,:]
    if dim == 2:
        img_grad[:, 0] = img_org[:, 1] - img_org[:, 0]
        for i in range(y_shape - 1):
            img_grad[:,i] = (img_org[:,i + 1] - img_org[:,i-1])/2
        img_grad[:,y_shape - 1] = img_org[:,y_shape - 1] - img_org[:,y_shape - 2]
    return img_grad
#



def main():
    # 原始图像z,x,y, centerline x,y,z
    # 载入图像
    img_name = 'test1'
    # img_name = 'Image4'
    org_dir = 'data/' + img_name + '.tif'
    # skel_dir = 'data/'+ img_name+'.skl.tif'
    img_org = tifffile.imread(org_dir)  # /255
    img_org[img_org > 300] = 300
    img_org = img_org / 300 * 255
    img_org = img_org.astype(np.uint8)
    # img_skel = tifffile.imread(skel_dir)




    # 获取图像梯度
    # img_grad = np.gradient(img_org)
    img_grad = np.zeros([2, img_org.shape[0], img_org.shape[1]])
    img_jac = np.zeros([2, 2, img_org.shape[0], img_org.shape[1]])


    img_grad[0] = cal_2d_image_grad(img_org, 2) # y方向导数
    img_grad[1] = cal_2d_image_grad(img_org, 1) # x方向导数

    img_jac[0][0] = cal_2d_image_grad(img_grad[0], 2)  # yy方向导数
    img_jac[1][0] = cal_2d_image_grad(img_grad[1], 2)  # xy方向导数
    img_jac[0][1] = cal_2d_image_grad(img_grad[0], 1)  # yx方向导数
    img_jac[1][1] = cal_2d_image_grad(img_grad[1], 1)  # xx方向导数

    # pause
    img_org = img_org.astype(np.float64)
    img_grad = img_grad.astype(np.float64)
    img_jac = img_jac.astype(np.float64)
    # img_grad[0] = cv.Sobel(img_org, cv.CV_64F, 1, 0)  # y方向导数
    # img_grad[1] = cv.Sobel(img_org, cv.CV_64F, 0, 1)  # x方向导数
    img_grad_tp = np.zeros_like(img_grad)
    # img_grad_tp[img_grad > 0] = 1
    # img_grad_tp[img_grad <= 0] = -1
    # img_grad_new = np.sqrt(img_grad * img_grad_tp)
    # img_grad = img_grad_new * img_grad_tp
    # print(img_grad[0][29][10])


    # img_jac[0][0] = cv.Sobel(img_grad[0], cv.CV_64F, 1, 0)  # yy方向导数
    # img_jac[1][0] = cv.Sobel(img_grad[1], cv.CV_64F, 1, 0)  # xy方向导数
    # img_jac[0][1] = cv.Sobel(img_grad[0], cv.CV_64F, 0, 1)  # yx方向导数
    # img_jac[1][1] = cv.Sobel(img_grad[1], cv.CV_64F, 0, 1)  # xx方向导数




    # img_jac_tp = np.zeros_like(img_jac)
    # img_jac_tp[img_jac > 0] = 1
    # img_jac_tp[img_jac <= 0] = -1
    # img_jac_new = np.sqrt(img_jac * img_jac_tp)
    # img_jac = img_jac_new * img_jac_tp




    # print(img_grad[0])
    # ax = plt.figure()
    # ax.add_subplot(1, 7, 1)
    # plt.imshow(img_org, cmap=plt.cm.gray)
    # ax.add_subplot(1, 7, 2)
    # plt.imshow(img_grad[0], cmap=plt.cm.gray)
    # ax.add_subplot(1, 7, 3)
    # plt.imshow(img_grad[1], cmap=plt.cm.gray)
    # ax.add_subplot(1, 7, 4)
    # plt.imshow(img_jac[0][0], cmap=plt.cm.gray)
    # ax.add_subplot(1, 7, 5)
    # plt.imshow(img_jac[1][0], cmap=plt.cm.gray)
    # ax.add_subplot(1, 7, 6)
    # plt.imshow(img_jac[0][1], cmap=plt.cm.gray)
    # ax.add_subplot(1, 7, 7)
    # plt.imshow(img_jac[1][1], cmap=plt.cm.gray)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    # pause

    # 获取原始轮廓参数方程
    print("计算初始法平面")
    # r = 10.0
    # init_x = 10.0
    # init_y = 15.0

    r = random.uniform(4, 10)
    init_x = random.uniform(5, 15)
    init_y = random.uniform(15, 20)
    # random.uniform(3, 5)
    # r = 4.0
    # init_x = 10.0
    # init_y = 15.0

    # r = 10.0
    #
    # init_x = 10.0
    # init_y = 25.0
    # 5,13,13
    # 5,6,6

    # x = ndimage.interpolation.map_coordinates(img_grad[0], [[init_x], [init_y]], order=1)
    # print(x)
    # pause

    circle_x, circle_y = get_centerline_circle_2D(init_x - 1, init_y - 1, r)

    pos_r_opt = np.zeros([3, 1])
    pos_r_opt[0][0] = init_x - 1
    pos_r_opt[1][0] = init_y - 1
    pos_r_opt[2][0] = r

    ######################33
    # ax = plt.figure()

    # ax.add_subplot(2,2,1)
    # plt.scatter(init_x, init_y, marker='o',c='g')
    # plt.imshow(img_grad[0], cmap=plt.cm.gray)
    # ax.add_subplot(2,2,2)
    # plt.scatter(init_x, init_y, marker='o',c='g')
    # plt.imshow(img_grad[1], cmap=plt.cm.gray)

    # ax.add_subplot(2,2,3)
    # plt.imshow(img_org, cmap=plt.cm.gray)
    # plt.scatter(init_x, init_y, marker='o',c='g')
    # plt.scatter(circle_x[:,0], circle_y[:,0],s=2, color='salmon')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # pause

    # 优化过程
    snake_opt = fit_snake_2D(pos_r_opt, img_org, img_grad, img_jac)
    print("优化前：")
    print(pos_r_opt)
    print("优化后：")
    print(snake_opt)

    circle_x_new, circle_y_new = get_centerline_circle_2D(snake_opt[0][0], snake_opt[0][1], snake_opt[0][2])

    ax = plt.figure()

    ax.add_subplot(2, 2, 1)
    plt.imshow(img_grad[0], cmap=plt.cm.gray)
    ax.add_subplot(2, 2, 2)
    plt.imshow(img_grad[1], cmap=plt.cm.gray)

    ax.add_subplot(2, 2, 3)
    plt.imshow(img_org, cmap=plt.cm.gray)
    plt.scatter(init_x - 1, init_y - 1, marker='o', c='g')
    plt.scatter(circle_x[:, 0], circle_y[:, 0], s=4, color='salmon')

    plt.scatter(snake_opt[0][0], snake_opt[0][1], marker='o', c='b')
    plt.scatter(circle_x_new[:, 0], circle_y_new[:, 0], s=4, color='red')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Make data.
    X = np.arange(-24, 24, 1)
    Y = np.arange(-24, 24, 1)
    X, Y = np.meshgrid(X, Y)
    r = 8
    R = K_kernal(X, Y, r)

    surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    main()
    # test()


