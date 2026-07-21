## plot cluster
import plotly.graph_objs as go
import numpy as np
import pyvista as pv
from geom_mesh_net.core_functions import data_loader as dl
from torch.utils.data import DataLoader, Subset
import torch

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




# -----------------------------
# Helper: run inference for one pattern
# -----------------------------

def predict_volume_for_pattern(
    pattern_ind,
    dataset,
    model_path,
    data_dir,
    data_file_name="clust_pattern_",
):
    """
    Loads one dataset item, loads the matching trained model, predicts the volume,
    and returns the objects needed for plotting.
    """

    single_pattern_dataset = Subset(dataset, [pattern_ind])

    dataloader = DataLoader(
        single_pattern_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dl.point_cloud_collate,
    )

    batch = next(iter(dataloader))

    thinned_coords, domain, thinned_labs, xx, yy, zz, full_upp_probs = batch

    model = dl.ContinuousNeuralField2()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x_flat = xx.flatten()
    y_flat = yy.flatten()
    z_flat = zz.flatten()

    inputs = torch.stack([x_flat, y_flat, z_flat], dim=1).float()

    with torch.no_grad():
        preds = model(inputs)

    grid_shape = xx.squeeze(0).shape
    predicted_volume = preds.reshape(grid_shape).cpu().numpy()

    original_vox = full_upp_probs.squeeze(0).cpu().numpy()

    grid_origin = (
        xx.min().item(),
        yy.min().item(),
        zz.min().item(),
    )

    thinned_coords_dict = thinned_coords[0]
    thinned_labels_array = thinned_labs[0]

    raw_data_path = data_dir / f"{data_file_name}{pattern_ind}.npz"
    raw_data = np.load(raw_data_path, allow_pickle=True)

    original_coords_dict = raw_data["coords"].item()
    original_labels_array = raw_data["labels"]

    return {
        "original_coords_dict": original_coords_dict,
        "original_labels_array": original_labels_array,
        "thinned_coords_dict": thinned_coords_dict,
        "thinned_labels_array": thinned_labels_array,
        "original_vox": original_vox,
        "predicted_volume": predicted_volume,
        "grid_origin": grid_origin,
    }


def plot_side_by_side_comparison(
    plots_to_make,
    style_key,
    grid_origin,
    resolution=0.5,
    orig_coords_dict=None,
    orig_labels=None,
    thinned_coords_dict=None,
    thinned_labels=None,
    original_vox=None,
    thinned_vox=None,
    predicted_volume=None,
    voxel_opacity=0.4,
    threshold_vals=None,
    iso_colors=None,
    screenshot_path=None,
    show=True,
    off_screen=False,
):
    """
    Dynamically renders selected point-pattern and voxel/neural-field plots.

    Parameters
    ----------
    plots_to_make : list[str]
        Which panels to include. Valid options are:
        - "original_pp3"
        - "thinned_pp3"
        - "original_vox"
        - "thinned_vox"
        - "predicted_volume"

    style_key : dict
        Styling dictionary for point clouds. Expected keys:
        "marks", "colors", "names", "sizes", "opacities"

    grid_origin : tuple
        Origin for PyVista ImageData.

    resolution : float
        Grid spacing.

    orig_coords_dict, thinned_coords_dict : dict
        Coordinate dictionaries with keys "x", "y", "z".

    orig_labels, thinned_labels : array-like
        Point labels corresponding to original and thinned coordinates.

    original_vox, thinned_vox, predicted_volume : np.ndarray
        3D scalar volumes to render as isosurfaces.

    screenshot_path : str or None
        If provided, saves a screenshot to this path.

    show : bool
        If True, display the viewer. If False, close after saving screenshot.
    """

    valid_plot_options = {
        "original_pp3",
        "thinned_pp3",
        "original_vox",
        "thinned_vox",
        "predicted_volume",
    }

    invalid_options = set(plots_to_make) - valid_plot_options
    if invalid_options:
        raise ValueError(
            f"Invalid plot option(s): {invalid_options}. "
            f"Valid options are: {valid_plot_options}"
        )

    if len(plots_to_make) == 0:
        raise ValueError("plots_to_make must contain at least one plot option.")

    if threshold_vals is None:
        threshold_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    default_iso_colors = [
        "navy",
        "blue",
        "deepskyblue",
        "limegreen",
        "gold",
        "orange",
        "red",
    ]

    if iso_colors is None:
        iso_colors = default_iso_colors

    # Make sure threshold_vals is a NumPy array
    threshold_vals = np.array(threshold_vals)

    # Adjust iso_colors to match the number of thresholds.
    # If too short, repeat colors cyclically.
    # If too long, truncate.
    n_thresholds = len(threshold_vals)

    if len(iso_colors) < n_thresholds:
        repeats = int(np.ceil(n_thresholds / len(iso_colors)))
        iso_colors = (iso_colors * repeats)[:n_thresholds]
    else:
        iso_colors = iso_colors[:n_thresholds]

    print(f"Initializing PyVista viewer with {len(plots_to_make)} panel(s)...")

    n_plots = len(plots_to_make)
    plotter = pv.Plotter(shape=(1, n_plots), window_size=(600 * n_plots, 600), off_screen=off_screen)
    plotter.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.1)

    def add_styled_points(coords_dict, labels_array, add_legend_labels=True):
        """
        Add labeled point cloud to the current PyVista subplot.
        """

        if coords_dict is None or labels_array is None:
            raise ValueError("Point-cloud plot requested, but coordinates or labels are missing.")

        pts_matrix = np.column_stack([
            coords_dict["x"],
            coords_dict["y"],
            coords_dict["z"],
        ])

        labels = np.array(labels_array).flatten()

        for mark in style_key["marks"]:
            inds = labels == mark

            if not np.any(inds):
                continue

            mark_points = pts_matrix[inds]

            idx = list(style_key["marks"]).index(mark)
            color = style_key["colors"][idx]
            name = style_key["names"][idx]
            size = style_key["sizes"][idx]
            opacity = style_key["opacities"][idx]

            cloud = pv.PolyData(mark_points)

            plotter.add_mesh(
                cloud,
                color=color,
                point_size=size,
                opacity=opacity,
                label=name if add_legend_labels else None,
                render_points_as_spheres=True,
            )

    def add_voxel_volume(volume, title_for_legend="Isosurface"):
        """
        Add a 3D scalar volume as multiple separately-labeled isosurfaces,
        with a legend showing both the relative threshold and the actual value.
        """

        if volume is None:
            raise ValueError(f"Voxel plot requested for {title_for_legend}, but volume is missing.")

        grid = pv.ImageData()
        grid.dimensions = volume.shape
        grid.spacing = (resolution, resolution, resolution)
        grid.origin = grid_origin
        grid.point_data["Density"] = volume.flatten(order="F")

        min_val = float(volume.min())
        max_val = float(volume.max())

        if np.isclose(max_val, min_val):
            print(f"Skipping {title_for_legend}: volume has no variation.")
            return

        iso_vals = min_val + (max_val - min_val) * threshold_vals

        # One distinct color per threshold


        for i, (thr, iso_val) in enumerate(zip(threshold_vals, iso_vals)):
            color = iso_colors[i]

            surface = grid.contour(isosurfaces=[float(iso_val)])

            if surface.n_points == 0:
                continue

            legend_label = f"{thr:.2f} : {iso_val:.3f}"

            plotter.add_mesh(
                surface,
                color=color,
                opacity=voxel_opacity,
                label=legend_label,
            )

    plot_titles = {
        "original_pp3": "Original Unthinned Point Cloud",
        "thinned_pp3": "Thinned Point Cloud",
        "original_vox": "Original Voxel Target",
        "thinned_vox": "Thinned Voxel Target",
        "predicted_volume": "Neural Field Prediction",
    }

    for panel_ind, plot_name in enumerate(plots_to_make):
        plotter.subplot(0, panel_ind)
        plotter.add_text(f"{panel_ind + 1}. {plot_titles[plot_name]}", font_size=12)

        if plot_name == "original_pp3":
            add_styled_points(orig_coords_dict, orig_labels, add_legend_labels=True)
            plotter.add_legend(bcolor="white", face="circle")

        elif plot_name == "thinned_pp3":
            add_styled_points(thinned_coords_dict, thinned_labels, add_legend_labels=True)
            plotter.add_legend(bcolor="white", face="circle")

        elif plot_name == "original_vox":
            add_voxel_volume(original_vox, title_for_legend="Original Voxel Field")
            plotter.add_legend(bcolor="white", face="circle")
            plotter.add_text(
                "Threshold : Value    ",
                font_size=8,
                position="upper_right",
            )

        elif plot_name == "thinned_vox":
            add_voxel_volume(thinned_vox, title_for_legend="Thinned Voxel Field")
            plotter.add_legend(bcolor="white", face="circle")

        elif plot_name == "predicted_volume":
            add_voxel_volume(predicted_volume, title_for_legend="Predicted Density Boundary")
            plotter.add_text(
                "Threshold : Value    ",
                font_size=8,
                position="upper_right",
            )
            # Overlay thinned points, but do not add them to the legend
            if thinned_coords_dict is not None and thinned_labels is not None:
                add_styled_points(thinned_coords_dict, thinned_labels, add_legend_labels=False)

            plotter.add_legend(bcolor="white", face="circle")

        plotter.add_axes()

    plotter.link_views()

    if screenshot_path is not None:
        plotter.screenshot(str(screenshot_path))

    if show:
        return plotter.show()
    else:
        plotter.close()
        return None