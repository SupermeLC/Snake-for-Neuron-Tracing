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

# 外部能量项的 offset项，确认无问题
def snake_energy_2D_offset(flattened_pts, img_org, img_grad):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    pts_x = pts[0][0]
    pts_y = pts[0][1]
    pts_r = pts[0][2]

    # 新的参数方程
    theta = np.arange(1e-5, 2 * np.pi, 1 / pts_r + 1e-5)
    cl_x = np.zeros([theta.shape[0], 1])
    cl_y = np.zeros([theta.shape[0], 1])

    cl_x[:, 0] = pts_x + pts_r * np.sin(theta)
    cl_y[:, 0] = pts_y + pts_r * np.cos(theta)

    # 外部能量项 offset
    dist_vals_x = ndimage.interpolation.map_coordinates(img_grad[0], [cl_y, cl_x], order=1)
    dist_vals_y = ndimage.interpolation.map_coordinates(img_grad[1], [cl_y, cl_x], order=1)

    o_x = np.sin(theta)
    o_y = np.cos(theta)
    external_energy_offset = np.sum(dist_vals_x[:, 0] * o_x + dist_vals_y[:, 0] * o_y) / (theta.shape[0] + 1e-5)

    return external_energy_offset


# 外部能量项的central项  主要看这个
def snake_energy_2D_central(flattened_pts, img_org, img_grad):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    pts_x = pts[0][0]
    pts_y = pts[0][1]
    pts_r = pts[0][2]

    X = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    Y = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    X_new = pts_x + X_mesh
    Y_new = pts_y + Y_mesh

    dist_vals_img_org_new_mesh = ndimage.interpolation.map_coordinates(img_org, [Y_new, X_new], order=1)
    K_value = K_kernal(X_mesh, Y_mesh, pts_r)
    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            # 3R 范围内
            if math.sqrt((X_mesh[i][j]) ** 2 + (Y_mesh[i][j]) ** 2) > 3 * pts_r:
                K_value[i][j] = 0
    external_energy_central = - np.mean(K_value * dist_vals_img_org_new_mesh)

    return external_energy_central

# 整体的能量项函数
def snake_energy_2D_new(flattened_pts, img_org, img_grad, lambda_ = 0.3):

    external_energy_offset = snake_energy_2D_offset(flattened_pts, img_org, img_grad)
    external_energy_central = snake_energy_2D_central(flattened_pts, img_org, img_grad)

    external_energy = lambda_ * external_energy_offset + (1 - lambda_) * external_energy_central
    print('Offset: %f  Central: %f  ' % ((lambda_ * external_energy_offset), ((1 - lambda_) * external_energy_central)))

    return external_energy


# 重要
# 理论计算的梯度
def snake_energy_2D_jac(flattened_pts, img_org, img_grad, img_jac,lambda_ = 0.3):
    np.set_printoptions(suppress=True)
    der = np.zeros_like(flattened_pts)

    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 3), 3))
    pts_x = pts[0][0]
    pts_y = pts[0][1]
    pts_r = pts[0][2]

    # 新的参数方程
    theta = np.arange(1e-5, 2 * np.pi, 1 / pts_r + 1e-5)

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

    # print("================== central test ========================")
    X = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    Y = np.arange(-3.0 * pts_r, 3.0 * pts_r, 3 / np.pi + 1e-5)
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    X_new_mesh = pts_x + X_mesh
    Y_new_mesh = pts_y + Y_mesh

    dist_vals_img_org_new_y = ndimage.interpolation.map_coordinates(img_grad[0], [Y_new_mesh, X_new_mesh], order=1)
    dist_vals_img_org_new_x = ndimage.interpolation.map_coordinates(img_grad[1], [Y_new_mesh, X_new_mesh], order=1)
    dist_vals_img_org = ndimage.interpolation.map_coordinates(img_org, [Y_new_mesh, X_new_mesh], order=1)

    K_value = K_kernal(X_mesh, Y_mesh, pts_r)
    K_value_jac = K_kernal_jac(X_mesh, Y_mesh, pts_r)

    der[0] += - (1 - lambda_) * np.mean(K_value * dist_vals_img_org_new_y)
    der[1] += - (1 - lambda_) * np.mean(K_value * dist_vals_img_org_new_x)
    der[2] += - (1 - lambda_) * np.mean(K_value_jac * dist_vals_img_org)# + K_value * dist_vals_img_org_jac

    return der

# 构建目标函数的优化方式
def fit_snake_2D(pts, img_org, img_grad, img_jac, lambda_ = 0.3):
    # optimize
    begin_time = time.time()
    cost_function = partial(snake_energy_2D_new, img_org=img_org, img_grad=img_grad,lambda_ = 0.3)
    cost_function_jac = partial(snake_energy_2D_jac, img_org=img_org, img_grad=img_grad, img_jac = img_jac,lambda_ = 0.3)

    res = optimize.minimize(cost_function, pts.ravel(), method='SLSQP', jac=cost_function_jac, options={'disp': True})

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


def snake_2D(org_img_dir,init_pos,init_r,lambda_,show_result=False):

    img_org = tifffile.imread(org_img_dir)

    # 获取图像梯度
    img_grad = np.zeros([2, img_org.shape[0], img_org.shape[1]])
    img_jac = np.zeros([2, 2, img_org.shape[0], img_org.shape[1]])

    img_grad[0] = cal_2d_image_grad(img_org, 2) # y方向导数
    img_grad[1] = cal_2d_image_grad(img_org, 1) # x方向导数

    img_jac[0][0] = cal_2d_image_grad(img_grad[0], 2)  # yy方向导数
    img_jac[1][0] = cal_2d_image_grad(img_grad[1], 2)  # xy方向导数
    img_jac[0][1] = cal_2d_image_grad(img_grad[0], 1)  # yx方向导数
    img_jac[1][1] = cal_2d_image_grad(img_grad[1], 1)  # xx方向导数

    img_org = img_org.astype(np.float64)
    img_grad = img_grad.astype(np.float64)
    img_jac = img_jac.astype(np.float64)


    # 获取原始轮廓参数方程
    print("计算初始法平面")

    r = float(init_r)
    init_x = float(init_pos[0])
    init_y = float(init_pos[1])

    circle_x, circle_y = get_centerline_circle_2D(init_x - 1, init_y - 1, r)
    pos_r_opt = np.zeros([3, 1])
    pos_r_opt[0][0] = init_x - 1
    pos_r_opt[1][0] = init_y - 1
    pos_r_opt[2][0] = r

    # 优化过程
    snake_opt = fit_snake_2D(pos_r_opt, img_org, img_grad, img_jac, float(lambda_))
    print("优化前：")
    print(pos_r_opt)
    print("优化后：")
    print(snake_opt)

    circle_x_new, circle_y_new = get_centerline_circle_2D(snake_opt[0][0], snake_opt[0][1], snake_opt[0][2])

    if show_result:
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
    org_img_dir = 'data/2D/test.tif'
    init_x = random.uniform(5, 15)
    init_y = random.uniform(15, 20)

    init_pos = [init_x,init_y]
    init_r = random.uniform(4, 10)
    snake_2D(org_img_dir,init_pos,init_r,lambda_ = 0.3,show_result=True)



