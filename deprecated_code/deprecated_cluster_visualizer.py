# old plotting functions

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


### this version deprecated 5/28/26
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
    voxel_color="red",
    threshold_vals=None,
    screenshot_path=None,
    show=True,
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
        Origin for PyVista ImageData, usually:
        (xx.min().item(), yy.min().item(), zz.min().item())

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

    print(f"Initializing PyVista viewer with {len(plots_to_make)} panel(s)...")

    n_plots = len(plots_to_make)
    plotter = pv.Plotter(shape=(1, n_plots), window_size=(600 * n_plots, 600))
    plotter.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.1)

    def add_styled_points(coords_dict, labels_array):
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
                label=name,
                render_points_as_spheres=True,
            )

    def add_voxel_volume(volume, label="Density Boundary", color=voxel_color):
        """
        Add a 3D scalar volume as isosurfaces to the current PyVista subplot.
        """

        if volume is None:
            raise ValueError(f"Voxel plot requested for {label}, but volume is missing.")

        grid = pv.ImageData()
        grid.dimensions = volume.shape
        grid.spacing = (resolution, resolution, resolution)
        grid.origin = grid_origin
        grid.point_data["Density"] = volume.flatten(order="F")

        min_val = volume.min()
        max_val = volume.max()

        if np.isclose(max_val, min_val):
            print(f"Skipping {label}: volume has no variation.")
            return

        iso_vals = min_val + (max_val - min_val) * threshold_vals
        isosurfaces = grid.contour(isosurfaces=iso_vals)

        plotter.add_mesh(
            isosurfaces,
            color=color,
            opacity=voxel_opacity,
            label=label,
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
            add_styled_points(orig_coords_dict, orig_labels)
            plotter.add_legend(bcolor="white", face="circle")

        elif plot_name == "thinned_pp3":
            add_styled_points(thinned_coords_dict, thinned_labels)
            plotter.add_legend(bcolor="white", face="circle")

        elif plot_name == "original_vox":
            add_voxel_volume(original_vox, label="Original Voxel Field", color=voxel_color)

        elif plot_name == "thinned_vox":
            add_voxel_volume(thinned_vox, label="Thinned Voxel Field", color=voxel_color)

        elif plot_name == "predicted_volume":
            add_voxel_volume(predicted_volume, label="Predicted Density Boundary", color=voxel_color)

            # Optional: overlay thinned points on the prediction if provided
            if thinned_coords_dict is not None and thinned_labels is not None:
                add_styled_points(thinned_coords_dict, thinned_labels)

        plotter.add_axes()

    plotter.link_views()

    if screenshot_path is not None:
        plotter.screenshot(str(screenshot_path))

    if show:
        return plotter.show()
    else:
        plotter.close()
        return None

####################

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
    plotter = pv.Plotter(shape=(1, n_plots), window_size=(600 * n_plots, 600))
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