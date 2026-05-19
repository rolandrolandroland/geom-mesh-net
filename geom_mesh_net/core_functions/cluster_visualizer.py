## plot cluster
import plotly.graph_objs as go
import numpy as np
import pyvista as pv
from geom_mesh_net.core_functions import data_loader as dl
from torch.utils.data import DataLoader

def cluster_visualizer(pp3, marks_to_plot, colors, names,
                       sizes = None, opacities = None,
                       fig = None, legend_size = 14):
    if fig is None:
        fig = go.Figure()
    if sizes is None:
        sizes = np.ones(len(marks_to_plot))* 3
    if opacities is None:
        opacities = np.ones(len(marks_to_plot))* 0.8
    for mark, col, nm, sz, op in zip(marks_to_plot, colors, names, sizes, opacities):
        inds = pp3.labels == mark
        #Create the trace (the data layer)
        trace = go.Scatter3d(
            x=pp3.coords['x'][inds],
            y=pp3.coords['y'][inds],
            z=pp3.coords['z'][inds],
            mode='markers',
            marker=dict(
                size=sz,
                color= col,
                opacity=op  # Make the host almost invisible so you can see through it
            ),
            name= nm
        )
        fig.add_trace(trace)

        # 3. Critical: Fix the aspect ratio so x, y, z scales are equal
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'  # This forces 1 unit in X to equal 1 unit in Y and Z
            ),
            margin=dict(l=0, r=0, b=0, t=0),  # Tight layout
        legend = dict(
            itemsizing='constant',  # Forces legend markers to be a standard, readable size (10px)
            font=dict(size=legend_size)  # Optional: Make the text bigger too
        )
        )
    return fig


## This script will visualize the data produced in train_network.py




def plot_side_by_side_comparison(orig_coords_dict, orig_labels, thinned_coords_dict, thinned_labels, predicted_volume,
                                 grid_origin, style_key, resolution=0.5):
    """
    Renders a 1x3 PyVista plot comparing Ground Truth, Input, and Prediction.
    Filters coordinates using an external labels array, matching Plotly logic.
    """
    print("Initializing Styled Side-by-Side 3D Viewer...")

    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))
    plotter.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.1)  # Add this!
    # Helper function to filter and style points
    def add_styled_points(coords_dict, labels_array):
        # Stack the dict (x, y, z) into a clean (N, 3) matrix
        pts_matrix = np.vstack(list(coords_dict.values())).T
        labels = np.array(labels_array).flatten()

        for mark in style_key["marks"]:
            # Find which rows in the matrix belong to this specific mark
            inds = (labels == mark)

            # If no points exist for this mark, skip it
            if not np.any(inds):
                continue

            # Extract just the points for this mark
            mark_points = pts_matrix[inds]

            # Get styling from the key
            idx = list(style_key["marks"]).index(mark)
            color = style_key["colors"][idx]
            name = style_key["names"][idx]
            size = style_key["sizes"][idx]
            opacity = style_key["opacities"][idx]

            # Render to PyVista
            cloud = pv.PolyData(mark_points)
            plotter.add_mesh(
                cloud,
                color=color,
                point_size=size,
                opacity=opacity,
                label=name,
                render_points_as_spheres=True
            )

    # --- Window 1: The Original Point Cloud (Ground Truth) ---
    plotter.subplot(0, 0)
    plotter.add_text("1. Original Unthinned Cloud", font_size=12)
    add_styled_points(orig_coords_dict, orig_labels)
    plotter.add_legend(bcolor="white", face="circle")  # Fixed legend formatting!

    # --- Window 2: The Thinned Point Cloud (The Network's Input) ---
    plotter.subplot(0, 1)
    plotter.add_text("2. Thinned Input (10% Retained)", font_size=12)
    add_styled_points(thinned_coords_dict, thinned_labels)

    # --- Window 3: The Neural Field Reconstruction ---
    plotter.subplot(0, 2)
    plotter.add_text("3. Continuous AI Reconstruction", font_size=12)

    grid = pv.ImageData()
    grid.dimensions = predicted_volume.shape
    grid.spacing = (resolution, resolution, resolution)
    grid.origin = grid_origin
    grid.point_data["Density"] = predicted_volume.flatten(order="F")

    min_val = predicted_volume.min()
    max_val = predicted_volume.max()
    # define threshold for isosurface
    threshold_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    iso_vals = min_val + (max_val - min_val) * threshold_vals
    isosurfaces = grid.contour(isosurfaces= iso_vals)

    plotter.add_mesh(isosurfaces, color="red", opacity=0.4, label="Predicted Density Boundary")
    add_styled_points(thinned_coords_dict, thinned_labels)

    plotter.link_views()
    plotter.show()