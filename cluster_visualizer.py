## plot cluster
import plotly.graph_objs as go
import numpy as np

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

