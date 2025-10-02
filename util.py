import os
import numpy as np
from scipy.ndimage import binary_fill_holes
import pyvista as pv
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology, color, exposure, restoration, registration
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import SimpleITK as sitk
import imageio
import vtk
from scipy.optimize import least_squares
from scipy.interpolate import make_splprep, splev

# sigma = 5
sigma = None


def show3d(path):
    plt_z_stack_volume(np.load(path))


def image_align(png_dir1='img1', png_dir2=None, ref_index=None, padding=32, output_path="volume.npy"):
    stack, files = load_stack(png_dir1, padding)
    if png_dir2 is not None:
        stack, files = merge_n_load_stack(png_dir1, png_dir2)
    n_slices, H, W = stack.shape
    print(f"Loaded {n_slices} slices, shape {H}x{W}")

    if ref_index is None:
        ref_index = n_slices // 2
    print("Reference slice:", ref_index)

    # Normalize to [0,1]
    # stack_norm = (stack - stack.min()) / (stack.max() - stack.min())

    # Convert to SimpleITK images (2D)
    sitk_images = [sitk.GetImageFromArray(slice_i) for slice_i in stack]

    # Reference image
    fixed_img = sitk_images[ref_index]

    # Register all slices to reference space
    resampled_stack = np.zeros_like(stack)

    resampled_stack[ref_index] = stack[ref_index]  # reference remains
    if sigma is not None:
        resampled_stack[ref_index] = smooth_mask(stack[ref_index], sigma)

    transforms = [None] * n_slices
    transforms[ref_index] = sitk.Transform(2, sitk.sitkIdentity)

    for i in range(n_slices):
        if i == ref_index:
            continue
        print(f"Registering slice {i} -> ref {ref_index}")
        fixed = fixed_img
        moving = sitk_images[i]
        resampled_sitk, tr = register_slice(fixed, moving,
                                            affine_iterations=150,
                                            bspline_grid_physical_spacing=(H / 4.0, W / 4.0),
                                            bspline_iterations=50)
        resampled_stack[i] = sitk.GetArrayFromImage(resampled_sitk)
        if sigma is not None:
            resampled_stack[i] = smooth_mask(sitk.GetArrayFromImage(resampled_sitk), sigma)
        transforms[i] = tr

    # Build 3D volume with spacing (z, y, x)
    volume = np.stack([resampled_stack[i] for i in range(n_slices)], axis=0)  # shape (Z, Y, X)

    # You might pick threshold based on histogram; here we use marching_iso param
    # verts, faces, normals, vals = measure.marching_cubes(stack, spacing=(0.3, 0.12, 0.12))

    # np.save(f"{output_volume_path}", stack)
    np.save(f"{output_path}/volume.npy", volume)


def fusion_image_align(png_dir1='img1', padding=32, output_path="volume.npy"):
    stack, files = load_stack(png_dir1, padding)

    n_slices, H, W = stack.shape
    print(f"Loaded {n_slices} slices, shape {H}x{W}")
    np.save(f"{output_path}/volume.npy", stack)


def create_surface_mesh(volume, threshold=0.3):
    """
    Extract surface mesh from volume

    Parameters:
    -----------
    volume : ndarray
        3D volume data
    threshold : float
        Isosurface level (relative to max intensity)

    Returns:
    --------
    verts : ndarray
        Surface vertices
    faces : ndarray
        Surface faces
    """
    # Normalize volume
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-10)

    print(f"Extracting surface at threshold {threshold}...")
    verts, faces, normals, values = measure.marching_cubes(
        volume_norm,
        level=threshold,
        spacing=(1.0, 1.0, 1.0)
    )

    return verts, faces


def watershed_separate_cells(filename):
    image = io.imread(filename, as_gray=True)
    # image = exposure.equalize_adapthist(image, clip_limit=0.03)  # CLAHE

    # 1. Threshold to create binary mask
    # thresh = filters.threshold_otsu(image)
    thresh = filters.threshold_local(image, 51)
    binary = image > thresh

    # 2. Remove small noise
    cleaned = morphology.remove_small_objects(binary, min_size=200)

    # 3. Distance transform
    distance = ndi.distance_transform_edt(cleaned)

    # 4. Find peaks as cell markers
    local_maxi = morphology.local_maxima(distance)
    markers, _ = ndi.label(local_maxi)

    # 5. Watershed to split touching cells
    labels = watershed(-distance, markers, mask=cleaned)

    # 6. Extract each cell separately
    props = measure.regionprops(labels, intensity_image=image)
    for i, prop in enumerate(props):
        minr, minc, maxr, maxc = prop.bbox
        cell_crop = image[minr:maxr, minc:maxc] * (labels[minr:maxr, minc:maxc] == prop.label)
        plt.imshow(cell_crop)
        plt.show()
        # io.imsave(f"cell_{i + 1}.png", (cell_crop * 255).astype(np.uint8))

    # Visualization
    # plt.imshow(labels, cmap="nipy_spectral")
    # plt.show()


def multichannel_seeding(dir, filename, outputdir, channel="cyan"):
    # ---- 1. Load image ----
    # Suppose image is (Z, Y, X, C) or (Y, X, C)
    img = io.imread(f"{dir}/{filename}.png")  # shape = (H, W, C)
    # if 2D RGB-like: img.shape = (Y, X, 2) or (Y, X, 3)
    # Convert to float
    img = img.astype(np.float32) / 255.0

    # Separate channels
    # Assuming nucleus is cyan: Cyan = Green + Blue
    if channel == "cyan":
        nucleus = (img[..., 1] + img[..., 2]) / 2.0  # or just use one channel
        membrane = img[..., 0]  # red channel if outline
    elif channel == "red":
        nucleus = img[..., 0]  # red channel
        membrane = color.rgb2gray(img)  # combine R,G,B

    # can increase contrast by the clip_limit, but resultsi in bad shape
    # nucleus = exposure.equalize_adapthist(nucleus, clip_limit=0.02)
    # membrane = exposure.equalize_adapthist(membrane, clip_limit=0.02)

    nucleus_smooth = filters.gaussian(nucleus, sigma=1)

    # Threshold to get binary nuclei
    nucleus_thresh = nucleus_smooth > filters.threshold_otsu(nucleus_smooth)

    # Remove small blobs (noise)
    nucleus_binary = morphology.remove_small_objects(nucleus_thresh, min_size=50)

    markers, _ = ndi.label(nucleus_binary)
    plt.imshow(markers, cmap="nipy_spectral")
    plt.title("Nucleus markers")
    plt.show()

    # Smooth membrane to reduce noise
    membrane_smooth = filters.gaussian(membrane, sigma=1)

    # Threshold to get binary mask for all cells
    membrane_binary = membrane_smooth > filters.threshold_otsu(membrane_smooth)

    # Distance transform on membrane mask
    distance = ndi.distance_transform_edt(membrane_binary)

    # Watershed: nucleus markers + membrane mask
    labels = watershed(-distance, markers, mask=membrane_binary)

    # Visualization
    # plt.imshow(labels, cmap="nipy_spectral")
    # plt.title("Cells separated by nucleus seeds")
    # plt.show()

    props = measure.regionprops(labels, intensity_image=membrane)

    for i, prop in enumerate(props):
        minr, minc, maxr, maxc = prop.bbox
        cell_crop = membrane[minr:maxr, minc:maxc] * (labels[minr:maxr, minc:maxc] == prop.label)
        io.imsave(f"{outputdir}/{filename}_{i}.png", (cell_crop * 255).astype(np.uint8))


###
### helper
###

# Function: perform affine + bspline registration (2D)
def register_slice(fixed, moving,
                   affine_iterations=200,
                   bspline_grid_physical_spacing=(50.0, 50.0),
                   bspline_iterations=100):
    """
    Returns: transformed moving image resampled into fixed's space.
    """
    # ----- Affine registration -----
    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMeanSquares()
    affine_reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                        minStep=1e-4,
                                                        numberOfIterations=affine_iterations,
                                                        relaxationFactor=0.5)
    affine_reg.SetInterpolator(sitk.sitkLinear)

    # Initial transform: centered affine
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving,
                                                          sitk.AffineTransform(2),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    affine_reg.SetInitialTransform(initial_transform, inPlace=False)
    affine_transform = affine_reg.Execute(fixed, moving)
    # Apply affine to moving
    moving_affine_resampled = sitk.Resample(moving, fixed, affine_transform,
                                            sitk.sitkLinear, 0.0, moving.GetPixelID())

    # ----- B-spline non-rigid registration (refinement) -----
    grid_physical_spacing = bspline_grid_physical_spacing
    image_physical_size = [sz * spc for sz, spc in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh_size = [int(image_physical_size[i] / grid_physical_spacing[i] + 0.5) \
                 for i in range(2)]
    bspline_transform = sitk.BSplineTransformInitializer(image1=fixed,
                                                         transformDomainMeshSize=mesh_size,
                                                         order=3)

    bspline_reg = sitk.ImageRegistrationMethod()
    bspline_reg.SetMetricAsMeanSquares()
    bspline_reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                     numberOfIterations=bspline_iterations)
    bspline_reg.SetInitialTransform(bspline_transform, inPlace=False)
    bspline_reg.SetInterpolator(sitk.sitkLinear)

    # Use the affine-resampled moving as the moving image for bspline refinement
    final_transform = bspline_reg.Execute(fixed, moving_affine_resampled)

    composite = sitk.CompositeTransform(2)
    composite.AddTransform(affine_transform)  # apply affine first
    composite.AddTransform(final_transform)  # then bspline refinement

    # Resample original moving with the composite transform
    moving_resampled = sitk.Resample(moving, fixed, composite, sitk.sitkLinear, 0.0, moving.GetPixelID())
    # return moving_resampled, composite
    return moving_affine_resampled, affine_transform


def smooth_mask(img, sigma=0.8):
    # gentle blur so thin features survive
    blur_small = gaussian_filter(img.astype(float), sigma=sigma)

    # threshold to binary
    bin_small = blur_small > 0.5

    # closing to smooth small indentations, choose radius carefully
    return morphology.closing(bin_small, morphology.disk(2))


def plt_z_stack_volume(path, weight=0.003, z=0.3, xy=0.12):
    volume = np.load(path)
    # If masks are labeled (cell IDs), choose one label or do all
    volume = (volume > 0).astype(np.uint8)  # simple foreground/background

    # Total Variation de-noising while preserving sharp edges
    volume = restoration.denoise_tv_chambolle(volume, weight=weight)
    # volume = ndi.gaussian_filter(volume, sigma=1)  # try sigma=1–2

    # # Plot in 3D
    plotter = pv.Plotter()
    # Create UniformGrid - note the dimension order for PyVista
    grid = pv.ImageData(
        dimensions=volume.shape[::-1],  # PyVista uses (nx, ny, nz)
        spacing=(xy, xy, z),  # Adjust based on your voxel spacing
        origin=(0, 0, 0)
    )

    # Add data to grid (flatten in Fortran order to match PyVista convention)
    grid.point_data["intensity"] = volume.transpose(2, 1, 0).flatten(order='F')
    # grid.cell_data["values"] = volume.transpose(2, 1, 0).flatten(order="F")
    # grid.point_data["values"] = volume.transpose(2, 1, 0).flatten(order="F")
    plotter.add_volume(
        grid,
        scalars="intensity"
    )
    plotter.show()


def normalize(volume, sigma=2):
    min_val = volume.min()
    max_val = volume.max()
    volume = (volume - min_val) / (max_val - min_val)
    return volume
    # return gaussian_filter(volume, sigma=sigma)


def show_mesh(args, sigma=0.5, z=0.22, xy=0.09):
    plotter = pv.Plotter()

    anisotropy_ratio = z / xy

    for volume, color, level, sigma_xy, water, threshold_val in args:
        sigma_z = sigma_xy / anisotropy_ratio
        anisotropic_sigma = (sigma_z, sigma_xy, sigma_xy)
        volume = normalize(volume, sigma)
        volume = gaussian_filter(
            input=volume,
            sigma=anisotropic_sigma,
            mode='nearest'
        )
        if water:
            # Deconvolution (optional if you know PSF)
            # from skimage.restoration import richardson_lucy
            # zstack_deconv = richardson_lucy(zstack_denoised, psf, iterations=10)

            # Contrast enhance
            # volume = filters.rank.equalize(volume.astype(np.uint16))
            volume = watershed_vol(volume, threshold_val)

        verts, faces, normals, values = measure.marching_cubes(volume, level=level, spacing=(z, xy, xy))

        print("Mesh vertices:", verts.shape, "faces:", faces.shape)

        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)

        # Create mesh
        mesh = pv.PolyData(verts, faces_pv)

        # mesh = mesh.smooth(
        #     n_iter=50,  # Number of smoothing iterations (increase for more smoothness)
        #     relaxation_factor=0.1,  # Controls how much vertices move per iteration (0.0 to 1.0)
        #     convergence=0.1,
        #     feature_smoothing=False,  # Set to False for uniform smoothing
        #     boundary_smoothing=False  # Set to False for uniform smoothing
        # )

        plotter.add_mesh(mesh, color=color, opacity=0.7, show_edges=False)
    plotter.show()


def plt_z_stack_mesh(data, weight=0.003, z=0.3, xy=0.12, level=0.1):
    volume = data
    if type(data) == str:
        volume = np.load(data, allow_pickle=True)
    volume = normalize(volume)

    # If masks are labeled (cell IDs), choose one label or do all
    # volume = (volume > 0).astype(np.uint8)  # simple foreground/background

    # Total Variation de-noising while preserving sharp edges
    # volume = np.pad(volume, ((1, 1), (0, 0), (0, 0)), mode='constant')

    # if weight is not None:
    #     volume = restoration.denoise_tv_chambolle(volume, weight=weight)
    # volume = reconstruct_complete_cell(volume, threshold=0, extend_factor=2.0)

    # level = np.percentile(volume, percentile)
    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(volume, level=level, spacing=(z, xy, xy))

    print("Mesh vertices:", verts.shape, "faces:", faces.shape)

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)

    # Create mesh
    mesh = pv.PolyData(verts, faces_pv)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="cyan", opacity=0.7, show_edges=False)
    plotter.show()


def get_z_stack(mask_dir):
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    # Stack into 3D volume (z, y, x)
    pics = []
    # for f in mask_files:
    #     img = io.imread(os.path.join(mask_dir, f), as_gray=True)
    #     pics.append(pad_to_multiple(img))
    # volume = np.stack(pics, axis=0)

    for f in mask_files:
        img = io.imread(os.path.join(mask_dir, f), as_gray=True) > 0
        img = binary_fill_holes(img)
        img = morphology.binary_closing(img, morphology.disk(2))
        pics.append(pad_to_multiple(img.astype(np.uint8)))
    volume = np.stack(pics, axis=0)

    return volume


def load_stack(dirname, padding=32):
    files = [os.path.join(dirname, f) for f in os.listdir(dirname)
             if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))]
    print(files)
    if len(files) == 0:
        raise ValueError("No images found in directory")
    # imgs = [imageio.imread(f).astype(np.float32) for f in files]
    imgs = [pad_to_multiple(imageio.imread(f).astype(np.float32), padding) for f in files]
    # for i in imgs:
    #     plt.imshow(i)
    #     plt.show()

    # If colored, convert to grayscale
    # imgs = [img.mean(axis=2) if img.ndim == 3 else img for img in imgs]

    return np.stack(imgs, axis=0), files


def load_slice(filename):
    imgs = imageio.imread(filename).astype(np.float32)
    return imgs


def merge_n_load_stack(dir1name, dir2name):
    dir1 = [i for i in os.listdir(dir1name) if i.endswith('png')]
    dir2 = [i for i in os.listdir(dir2name) if i.endswith('png')]

    files = [(os.path.join(dir1name, dir1[i]), os.path.join(dir2name, dir2[i])) for i in range(len(dir1))]

    if len(files) == 0:
        raise ValueError("No images found in directory")
    imgs = []
    for i1, i2 in files:
        im1 = pad_to_multiple(imageio.imread(i1).astype(np.float32))
        im2 = pad_to_multiple(imageio.imread(i2).astype(np.float32))

        # --- Contrast Stretching to 2–98 percentile ---
        p2, p98 = np.percentile(im2, (2, 98))
        im2 = exposure.rescale_intensity(im2, in_range=(p2, p98))

        # imgs.append(im1)
        im = im1.copy()
        im[im2 != 0] = im2[im2 != 0]
        imgs.append(im)
        # imgs.append(np.maximum(im1, im2))

    # If colored, convert to grayscale
    # imgs = [img.mean(axis=2) if img.ndim == 3 else img for img in imgs]

    return np.stack(imgs, axis=0), files


def pad_to_multiple(img, mult=32):
    h, w = img.shape[:2]
    pad_h = (mult - h % mult) % mult - 2
    pad_w = (mult - w % mult) % mult - 2
    print(img.shape, pad_w, pad_h)

    return np.pad(img, ((2, pad_h), (2, pad_w)), mode='constant')


def pad_rgb_to_multiple(img, mult=32):
    """
    Pads an RGB image (H, W, C) so its height and width
    are divisible by `mult` (default 16).
    """
    h, w, c = img.shape
    pad_h = (mult - h % mult) % mult
    pad_w = (mult - w % mult) % mult
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')


def show2image(im1, im2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Display first image
    axes[0].imshow(im1)
    axes[0].set_title('Channel 1 - DAPI')
    axes[0].axis('off')  # Hide axes

    # Display second image
    axes[1].imshow(im2)
    axes[1].set_title('Channel 2 - GFP')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def watershed_vol(stack, threshold_value):
    # # Assuming 'stack' is your 3D z-stack (z, y, x)
    # # Threshold if needed
    binary = stack > threshold_value
    # binary = morphology.binary_erosion(binary, morphology.ball(2))
    #
    # # Compute 3D distance transform
    # distance = ndi.distance_transform_edt(binary)
    #
    # # Find local maxima (cell centers)
    # coords = peak_local_max(distance, min_distance=5, labels=binary)
    # markers = np.zeros_like(distance, dtype=int)
    # markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    # # mask = np.zeros(distance.shape, dtype=bool)
    # # mask[tuple(coords.T)] = True
    #
    # # Create markers
    # # markers, _ = ndi.label(mask)
    #
    # # Apply watershed
    # labels = watershed(-distance, markers, mask=binary)

    distance = ndi.distance_transform_edt(binary)

    h = 2

    seeds = morphology.h_maxima(distance, h=h, footprint=morphology.ball(3))
    # Label the seeds
    markers, num_markers = ndi.label(seeds)
    print(f"Found {num_markers} markers using h_maxima with h={h}")

    # Watershed
    labels = watershed(-distance, markers, mask=binary)

    return labels
