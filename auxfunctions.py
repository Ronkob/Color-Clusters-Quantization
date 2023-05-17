import time

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import imageio.v3 as iio
import sklearn as sk
import matplotlib
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        # prints the time in minutes and seconds and to the 3rd digit after the dot
        print("Execution time: ", round((end_time - start_time) / 60, 0), " minutes and ",
              round((end_time - start_time) % 60, 3), " seconds")

        return ret

    return wrapper  # returns the decorated function


def plot_3d_hist(img, txt=''):
    r, g, b = cv2.split(img)
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.savefig(f"3d_histogram{txt}.png")
    plt.show()


def trasform_points(x, y, z):
    """
    take a list of 3d points, and return the polar coordinates of the points in the x-y plane, and the x-z
    plane.
    :param points:
    :return:
    """
    thetas_xy = np.arctan2(y, x)
    thetas_xz = np.arctan2(z, x)
    r_xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    return np.stack([thetas_xy, thetas_xz, r_xyz], axis=1)


# plot all the thetas in the x-y plane, and the x-z plane.
def plot_thetas(thetas_xy, thetas_xz, r_xyz):
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(thetas_xy, thetas_xz, marker=".", c=r_xyz, cmap="Purples")
    axis.set_xlabel("theta_xy")
    axis.set_ylabel("theta_xz")
    plt.show()


def find_centeroids(data_2d, weights):
    # Perform K-means clustering with different numbers of clusters
    max_clusters = 10
    inertia = []

    # make a subplot for each number of clusters
    fig, axes = plt.subplots(2, 5, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data_2d, sample_weight=weights)
        inertia.append(kmeans.inertia_)
        # Plot the clusters and cluster centers
        ax = axes[n_clusters - 1]
        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans.labels_, cmap='viridis')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='r')
        ax.set_title('k = {}'.format(n_clusters))

    plt.show()

    # Plot the inertia to see which number of clusters is best
    plt.plot(range(1, max_clusters + 1), inertia)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


# turn the cluster centers into 3d lines from the origin, and plot them on the 3d histogram
def polar_to_cartesian(theta_xy, theta_xz, r):
    """
    take a point in polar coordinates, and return the point in cartesian coordinates.
    :param theta_xy:
    :param theta_xz:
    :param r:
    :return:
    """
    x = r * np.cos(theta_xy) * np.cos(theta_xz)
    y = r * np.sin(theta_xy) * np.cos(theta_xz)
    z = r * np.sin(theta_xz)
    return x, y, z


# turn a polar representation into a line (start and end points, start at the origin, end at length r)
def polar_to_line(theta_xy, theta_xz, r):
    """
    take a point in polar coordinates, and return the point in cartesian coordinates.
    :param theta_xy:
    :param theta_xz:
    :param r:
    :return:
    """
    x = r * np.cos(theta_xy) * np.cos(theta_xz)
    y = r * np.sin(theta_xy) * np.cos(theta_xz)
    z = r * np.sin(theta_xz)
    return [[np.array([0, 0, 0]), np.array([x, y, z])[:, i]] for i in range(len(x))]


def min_perpendicular_distance(point, lines):
    """
    find the minimum perpendicular distance between a point and a set of lines.
    :param point:
    :param lines: a tuple of two lists, each list contains the start and end points of the lines.
    :return: a tuple of the closest point on the line, and the distance.
    """
    min_dist = float('inf')
    closest_color_ret = None
    starts = lines[0]
    ends = lines[1]
    for idx in range(len(starts)):
        p1, p2 = starts[idx], ends[idx]
        u = p2 - p1
        v = point - p1
        w = np.cross(u, v)
        dist = np.linalg.norm(w) / np.linalg.norm(u)
        if dist < min_dist:
            min_dist = dist
            closest_color_ret = ends[idx]
    return closest_color_ret, min_dist


def closest_point_on_line(point, line):
    """
    find the closest point on a line to a given point.
    :param point:
    :param line: a tuple of two points, the start and end points of the line.
    :return: the closest point on the line, and the distance.
    """
    p1, p2 = line
    u = p2 - p1
    v = point - p1
    w = np.cross(u, v)
    dist = np.linalg.norm(w) / np.linalg.norm(u)
    return (p1 + u * np.dot(u, v) / np.dot(u, u)), dist


# transform each point to it's closest point on the line, and save the new points
@measure_time
def transform_points_to_lines(points, lines):
    """
    transform each point to it's closest point on the line, and save the new points
    :param points:
    :param lines:
    :return:
    """
    r, g, b = points[0], points[1], points[2]
    new_points = []
    for point_idx in range(len(r)):
        closest = [closest_point_on_line(np.array([r[point_idx], g[point_idx], b[point_idx]]), lines[i]) for i
                   in range(len(lines))]
        closest_point = min(closest, key=lambda x: x[1])[0]
        new_points.append(closest_point)
    return np.array(new_points)


# display the original image and the new image
def plot_2_images(img1, img2):
    """
    display the original image and the new image
    :param img1:
    :param img2:
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(img1)
    ax[0].set_title("Original image")
    ax[1].imshow(img2)
    ax[1].set_title("New image")
    fig.tight_layout()
    plt.savefig("original_and_new.png")
    plt.show()
