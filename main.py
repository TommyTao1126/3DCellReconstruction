import numpy as np
from util import plt_z_stack_mesh, show_mesh
from example import example, get_inv_covariance_n_mu, gaussian
import tifffile as tf
from skimage import io

plt_mesh = True
mesh_weight = None
mesh_percentile = 60
volume_weight = 0.002
"""
 Plot img1
"""
# Part 1
# image_align(png_dir1="img1/part1", output_path="img1/part1")
# if plt_mesh:
#     example(gaussian(np.load("img1/part1/volume.npy")))
# plt_z_stack_mesh(gaussian(np.load("img1/part1/volume.npy")), weight=mesh_weight)
# plt_z_stack_mesh("img1/part1/volume.npy")
# else:
#     plt_z_stack_volume("img1/part1/volume.npy", volume_weight)
# Part 2
# stage1
# fusion_image_align(png_dir1="img1/part2/stage1", output_path="img1/part2/stage1")
# fusion_image_align(png_dir1="img1/part2/stage2", output_path="img1/part2/stage2")
# fusion_image_align(png_dir1="img1/part2/stage3", output_path="img1/part2/stage3", padding=64)
#
# if plt_mesh:
#     example("img1/part2/stage1/volume.npy")
#     example("img1/part2/stage2/volume.npy")
#     example("img1/part2/stage3/volume.npy")
# plt_z_stack_mesh("img1/part2/stage1/volume.npy", weight=mesh_weight, percentile=mesh_percentile)
# plt_z_stack_mesh("img1/part2/stage2/volume.npy", weight=mesh_weight, percentile=mesh_percentile)
# plt_z_stack_mesh("img1/part2/stage3/volume.npy", weight=mesh_weight, percentile=mesh_percentile)
# else:
#     plt_z_stack_volume("img1/part2/stage1/volume.npy", weight=volume_weight)
#     plt_z_stack_volume("img1/part2/stage2/volume.npy", weight=volume_weight)
#     plt_z_stack_volume("img1/part2/stage3/volume.npy", weight=volume_weight)

# # # Part 3
# image_align(png_dir1="img1/part3", output_path="img1/part3")
# if plt_mesh:
#     plt_z_stack_mesh("img1/part3/volume.npy", weight=0.003, percentile=50)
# else:
#     plt_z_stack_volume("img1/part3/volume.npy", weight=volume_weight)
"""
 Plot img2
"""
# image_align(png_dir1="img2/part3/TCAB1", png_dir2="img2/part3/nucleolin", output_path="img2/part3")
# plt_multi_class_z_stack("img2/part3/volume.npy")


"""
 Plot img3 (unfinished)
"""
# stack, _ = load_stack("img3/part1", 32)
# np.save("img3/part1/volume.npy", stack)
# plt_z_stack_mesh("img3/part1/volume.npy", weight=0.005)

"""
 Plot 4D Z-stack
"""
# volume1 = nd2.imread('nd2/day3/day3_1.nd2')
# volume1 = volume1[:, 0, :, :]
# grad = filters.sobel(volume1)          # 3-D Sobel magnitude
# edges = grad > filters.threshold_otsu(grad)
# volume1 = morphology.binary_closing(edges, morphology.ball(3))  # fill small gaps
# plt_z_stack_volume(volume1, weight=0.1, z=0.3)


"""
channel: 1: blue 2: green 3: yellow 4: read
"""
z_step = [0.2985, 0.1593, 0.1502, 0.2985, 0.1502, 0.1502, 0.1502]
levels = [0.065, 0.07, 0.09]

colors = ["#dddaeb", "yellow", "magenta"]
# sigma_xys = [4, 1.5, 1.5]
sigma_xys = [3, 1, 0]
water = [False, True, True]
threshold_vals = [0, 0.1, 0.1]
# for i in range(3, 10):

for i in range(8, 9):
    volumes = [np.load("stacks/stack" + str(i) + "channel1.npy"),
               np.load("stacks/stack" + str(i) + "channel2.npy")]
    # np.load("stacks/stack" + str(i) + "channel4.npy")]
    args = zip(volumes,
               colors,
               levels,
               sigma_xys,
               water,
               threshold_vals)
    show_mesh(args, z=z_step[i - 4])
# example("stacks/stack4.npy")

# image_day12 = tf.imread("8_day12.tif")
# plt_z_stack_mesh(image_day12[:, 1, :, :], z=0.1502)
