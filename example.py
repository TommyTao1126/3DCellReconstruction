import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree, Delaunay
from skimage import measure
import matplotlib.pyplot as plt


# ============================================================================
# STEP 0: Create Synthetic Z-Stack with Two Droplets
# ============================================================================

def create_synthetic_zstack():
    """Create a synthetic z-stack with two droplet-like structures"""
    nz, ny, nx = 20, 50, 50
    z_stack = np.zeros((nz, ny, nx))

    # Droplet 1: center at (25, 25, 10), radius 8
    c1 = {'x': 25, 'y': 25, 'z': 10, 'r': 8}
    # Droplet 2: center at (35, 15, 12), radius 6
    c2 = {'x': 35, 'y': 15, 'z': 12, 'r': 6}

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                d1 = np.sqrt((x - c1['x']) ** 2 + (y - c1['y']) ** 2 + (z - c1['z']) ** 2)
                d2 = np.sqrt((x - c2['x']) ** 2 + (y - c2['y']) ** 2 + (z - c2['z']) ** 2)

                val = 0
                if d1 < c1['r']:
                    val = max(val, 200 * np.exp(-d1 ** 2 / (2 * (c1['r'] / 2) ** 2)))
                if d2 < c2['r']:
                    val = max(val, 180 * np.exp(-d2 ** 2 / (2 * (c2['r'] / 2) ** 2)))

                val += np.random.randn() * 5
                z_stack[z, y, x] = np.clip(val, 0, 255)

    return z_stack


# ============================================================================
# STEP 1: Segment Z-Stack
# ============================================================================

def segment_zstack(z_stack, threshold=80):
    """Threshold-based segmentation"""
    smoothed = gaussian_filter(z_stack, sigma=1.0)
    binary = smoothed > threshold
    return binary.astype(np.uint8)


# ============================================================================
# STEP 2: Extract Boundary Points
# ============================================================================

def extract_boundary_points(binary_volume):
    """Extract surface/boundary points from binary volume"""
    nz, ny, nx = binary_volume.shape
    boundary_points = []

    for z in range(1, nz - 1):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if binary_volume[z, y, x] == 1:
                    neighbors = [
                        binary_volume[z - 1, y, x],
                        binary_volume[z + 1, y, x],
                        binary_volume[z, y - 1, x],
                        binary_volume[z, y + 1, x],
                        binary_volume[z, y, x - 1],
                        binary_volume[z, y, x + 1]
                    ]

                    if 0 in neighbors:
                        boundary_points.append([x, y, z])

    return np.array(boundary_points)


# ============================================================================
# STEP 3: Gaussian-Based Gap Filling
# ============================================================================

def identify_sparse_regions(points, max_distance=2.5, k_neighbors=6):
    """Identify regions where points are sparse"""
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k_neighbors + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    sparse_mask = mean_distances > max_distance
    return sparse_mask, mean_distances


def gaussian_fill_gaps(boundary_points, max_gap=2.5):
    """Fill gaps in boundary using Gaussian RBF interpolation"""
    print(f"Original boundary points: {len(boundary_points)}")

    sparse_mask, mean_distances = identify_sparse_regions(boundary_points, max_gap)
    sparse_points = boundary_points[sparse_mask]

    print(f"Sparse regions identified: {len(sparse_points)} points")

    if len(sparse_points) == 0:
        return boundary_points

    candidates = []
    for point in sparse_points:
        n_candidates = 5
        for _ in range(n_candidates):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = max_gap * 0.5

            offset = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])

            candidates.append(point + offset)

    if len(candidates) == 0:
        return boundary_points

    candidates = np.array(candidates)
    print(f"Generated {len(candidates)} candidate points")

    centroid = np.mean(boundary_points, axis=0)
    distances_to_centroid = np.linalg.norm(boundary_points - centroid, axis=1)

    rbf = Rbf(
        boundary_points[:, 0],
        boundary_points[:, 1],
        boundary_points[:, 2],
        distances_to_centroid,
        function='gaussian',
        epsilon=2.0
    )

    candidate_distances = rbf(
        candidates[:, 0],
        candidates[:, 1],
        candidates[:, 2]
    )

    actual_distances = np.linalg.norm(candidates - centroid, axis=1)
    error = np.abs(candidate_distances - actual_distances)
    on_surface_mask = error < max_gap * 0.3

    filled_points = candidates[on_surface_mask]
    print(f"Filled points added: {len(filled_points)}")

    complete_boundary = np.vstack([boundary_points, filled_points])
    return complete_boundary


# ============================================================================
# STEP 4: Create Mesh from Point Cloud
# ============================================================================

def create_mesh_from_points(points, method='delaunay'):
    """
    Create a mesh surface from point cloud

    Methods:
    - 'delaunay': 3D Delaunay triangulation
    - 'poisson': Poisson surface reconstruction (requires normals)
    """
    if method == 'delaunay':
        # Create PyVista point cloud
        cloud = pv.PolyData(points)

        # Delaunay 3D triangulation
        mesh = cloud.delaunay_3d(alpha=3.0)

        # Extract surface
        surface = mesh.extract_geometry()

        return surface

    elif method == 'alpha_shape':
        # Alpha shape for better surface reconstruction
        cloud = pv.PolyData(points)

        # Compute normals (needed for better reconstruction)
        cloud.compute_normals(inplace=True)

        # Delaunay with alpha parameter
        mesh = cloud.delaunay_3d(alpha=2.5)
        surface = mesh.extract_geometry()

        return surface


# ============================================================================
# STEP 5: PyVista Visualization
# ============================================================================

def visualize_with_pyvista(z_stack, binary, boundary, filled, show_mesh=True):
    """
    Interactive 3D visualization with PyVista
    """
    print("\n[Visualization] Creating PyVista scene...")

    # Create plotter with multiple viewports
    plotter = pv.Plotter(shape=(2, 2), window_size=[1600, 1200])

    # ========================================================================
    # Subplot 1: Original Z-Stack Volume
    # ========================================================================
    plotter.subplot(0, 0)
    plotter.add_text("Original Z-Stack Volume", font_size=12, color='white')

    # Create volumetric grid
    grid = pv.ImageData(dimensions=z_stack.shape)
    grid['intensity'] = z_stack.flatten(order='F')

    # Volume rendering with opacity
    plotter.add_volume(
        grid,
        cmap='gray',
        opacity='sigmoid',
        scalar_bar_args={'title': 'Intensity'}
    )
    plotter.camera_position = 'iso'

    # ========================================================================
    # Subplot 2: Binary Segmentation (Volumetric)
    # ========================================================================
    plotter.subplot(0, 1)
    plotter.add_text("Binary Segmentation (Volume)", font_size=12, color='white')

    # Extract isosurface from binary volume using marching cubes
    verts, faces, normals, values = measure.marching_cubes(binary, level=0.5)

    # Create mesh
    faces_pv = np.column_stack([np.full(len(faces), 3), faces])
    volume_mesh = pv.PolyData(verts, faces_pv)

    plotter.add_mesh(
        volume_mesh,
        color='lightblue',
        opacity=0.7,
        smooth_shading=True,
        show_edges=False
    )
    plotter.camera_position = 'iso'

    # ========================================================================
    # Subplot 3: Boundary Points Only
    # ========================================================================
    plotter.subplot(1, 0)
    plotter.add_text(f"Boundary Points ({len(boundary)} pts)", font_size=12, color='white')

    # Create point cloud
    boundary_cloud = pv.PolyData(boundary)

    # Color by Z coordinate
    boundary_cloud['z_coord'] = boundary[:, 2]

    plotter.add_points(
        boundary_cloud,
        scalars='z_coord',
        cmap='viridis',
        point_size=8,
        render_points_as_spheres=True,
        scalar_bar_args={'title': 'Z Depth'}
    )
    plotter.camera_position = 'iso'

    # ========================================================================
    # Subplot 4: Gap-Filled Surface with Mesh
    # ========================================================================
    plotter.subplot(1, 1)
    plotter.add_text(f"Gap-Filled Surface ({len(filled)} pts)", font_size=12, color='white')

    # Show original boundary points
    boundary_cloud = pv.PolyData(boundary)
    plotter.add_points(
        boundary_cloud,
        color='blue',
        point_size=5,
        opacity=0.3,
        render_points_as_spheres=True
    )

    # Show filled points
    n_original = len(boundary)
    if len(filled) > n_original:
        filled_only = filled[n_original:]
        filled_cloud = pv.PolyData(filled_only)
        plotter.add_points(
            filled_cloud,
            color='lime',
            point_size=8,
            render_points_as_spheres=True
        )

    # Create and show mesh if requested
    if show_mesh:
        try:
            print("  Creating surface mesh...")
            mesh = create_mesh_from_points(filled, method='delaunay')

            plotter.add_mesh(
                mesh,
                color='cyan',
                opacity=0.6,
                smooth_shading=True,
                show_edges=True,
                edge_color='white',
                line_width=0.5
            )
            print("  Mesh created successfully!")
        except Exception as e:
            print(f"  Warning: Could not create mesh: {e}")

    plotter.camera_position = 'iso'

    # Add legend for last subplot
    plotter.add_legend(
        [['Original Boundary', 'blue'],
         ['Filled Points', 'lime'],
         ['Surface Mesh', 'cyan']],
        size=(0.2, 0.1)
    )

    # ========================================================================
    # Global settings
    # ========================================================================
    plotter.link_views()  # Synchronize camera across all views

    print("\n[Controls]")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Zoom")
    print("  - Middle mouse: Pan")
    print("  - Press 's' to save screenshot")
    print("  - Press 'q' to quit")

    plotter.show()


def visualize_single_view_detailed(z_stack, binary, boundary, filled):
    """
    Single detailed view with side-by-side comparison
    """
    print("\n[Visualization] Creating detailed comparison...")

    plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 600])

    # View 1: Volumetric rendering
    plotter.subplot(0, 0)
    plotter.add_text("Volume Segmentation", font_size=14)

    verts, faces, normals, values = measure.marching_cubes(binary, level=0.5)
    faces_pv = np.column_stack([np.full(len(faces), 3), faces])
    volume_mesh = pv.PolyData(verts, faces_pv)

    plotter.add_mesh(
        volume_mesh,
        color='lightblue',
        opacity=0.8,
        smooth_shading=True,
        show_edges=True,
        edge_color='darkblue',
        line_width=0.5
    )
    plotter.add_axes()
    plotter.camera_position = 'iso'

    # View 2: Boundary only
    plotter.subplot(0, 1)
    plotter.add_text(f"Boundary Extraction\n{len(boundary)} points", font_size=14)

    boundary_cloud = pv.PolyData(boundary)
    boundary_cloud['z'] = boundary[:, 2]

    plotter.add_points(
        boundary_cloud,
        scalars='z',
        cmap='plasma',
        point_size=10,
        render_points_as_spheres=True,
        scalar_bar_args={'title': 'Z Coordinate'}
    )
    plotter.add_axes()
    plotter.camera_position = 'iso'

    # View 3: Gap-filled with mesh
    plotter.subplot(0, 2)
    n_filled = len(filled) - len(boundary)
    plotter.add_text(f"Gap Filling + Mesh\n+{n_filled} points", font_size=14)

    # Original points (smaller, transparent)
    boundary_cloud = pv.PolyData(boundary)
    plotter.add_points(
        boundary_cloud,
        color='royalblue',
        point_size=6,
        opacity=0.4,
        render_points_as_spheres=True
    )

    # Filled points (larger, bright)
    if n_filled > 0:
        filled_only = filled[len(boundary):]
        filled_cloud = pv.PolyData(filled_only)
        plotter.add_points(
            filled_cloud,
            color='lime',
            point_size=12,
            render_points_as_spheres=True
        )

    # Create smooth mesh
    try:
        mesh = create_mesh_from_points(filled, method='delaunay')
        plotter.add_mesh(
            mesh,
            color='cyan',
            opacity=0.5,
            smooth_shading=True,
            show_edges=True,
            edge_color='white',
            line_width=0.3
        )
    except:
        pass

    plotter.add_axes()
    plotter.camera_position = 'iso'

    plotter.link_views()
    plotter.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("3D DROPLET RECONSTRUCTION WITH PYVISTA")
    print("=" * 70)

    # Step 0: Create synthetic data
    print("\n[Step 0] Creating synthetic z-stack with 2 droplets...")
    z_stack = create_synthetic_zstack()

    for i in z_stack:
        plt.imshow(i)
        plt.show()

    # print(f"  Shape: {z_stack.shape}")
    # print(f"  Intensity: [{z_stack.min():.1f}, {z_stack.max():.1f}]")
    #
    # # Step 1: Segment
    # print("\n[Step 1] Segmenting volume...")
    # binary = segment_zstack(z_stack, threshold=80)
    # print(f"  Voxels: {np.sum(binary):,} / {binary.size:,}")
    #
    # # Step 2: Extract boundary
    # print("\n[Step 2] Extracting boundary points...")
    # boundary = extract_boundary_points(binary)
    # print(f"  Points: {len(boundary):,}")
    # print(f"  Reduction: {100 * (1 - len(boundary) / np.sum(binary)):.1f}%")
    #
    # # Step 3: Fill gaps
    # print("\n[Step 3] Filling gaps with Gaussian RBF...")
    # filled = gaussian_fill_gaps(boundary, max_gap=2.5)
    # print(f"  Final: {len(filled):,} points (+{len(filled) - len(boundary)})")
    #
    # # Step 4: Visualize - Multiple views
    # print("\n[Step 4] Creating interactive 3D visualization...")
    # print("\nChoose visualization mode:")
    # print("  1. Four-panel overview (recommended)")
    # print("  2. Three-panel detailed comparison")
    #
    # choice = input("Enter 1 or 2 (default: 1): ").strip() or "1"
    #
    # if choice == "2":
    #     visualize_single_view_detailed(z_stack, binary, boundary, filled)
    # else:
    #     visualize_with_pyvista(z_stack, binary, boundary, filled, show_mesh=True)
    #
    # print("\n" + "=" * 70)
    # print("VISUALIZATION COMPLETE!")
    # print("=" * 70)

    return z_stack, binary, boundary, filled


if __name__ == "__main__":
    # Run the pipeline
    z_stack, binary, boundary, filled = main()