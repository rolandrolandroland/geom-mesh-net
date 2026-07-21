## plot cluster
import pickle
import plotly.graph_objs as go
import numpy as np
import cluster_visualizer as cv
with open("data/clust_01.pkl", 'rb') as f:
    clust_pattern_01 = pickle.load(f)

with open("data/clust_02.pkl", 'rb') as f:
    clust_pattern_02 = pickle.load(f)

print(f"loaded pattern with {clust_pattern_01.n_points} points")
print(f"loaded pattern with {clust_pattern_02.n_points} points")

# alright lets define the function parameters
key = {"marks": [0, 1, 2, 3],
       'colors': ["blue", "mediumblue", "brown", "firebrick"],
       'names': ["Host Background", "Host Clusters", "Guest Clusters", "Guest Background"],
       'sizes': [2, 2, 2, 2],
       'opacities': [0.8, 0.8, 0.8, 0.8]}

key_arrays = {k: np.array(v) for k, v in key.items()}
# just looking at the clustered domain at the moment
to_use = [1, 2]
fig_01 = cv.cluster_visualizer(clust_pattern_01,
                               marks_to_plot=key_arrays["marks"][to_use],
                               colors=key_arrays["colors"][to_use],
                               names=key_arrays["names"][to_use],
                               sizes=key_arrays["sizes"][to_use],
                               opacities=key_arrays["opacities"][to_use])

fig_01.show()


# now just plot clustered type, both inside and outside of clusters
to_use = [2,3]
fig_02 = cv.cluster_visualizer(clust_pattern_01,
                               marks_to_plot=key_arrays["marks"][to_use],
                               colors=key_arrays["colors"][to_use],
                               names=key_arrays["names"][to_use],
                               sizes=key_arrays["sizes"][to_use],
                               opacities=key_arrays["opacities"][to_use])

fig_02.show()

# now just plot clustered type, both inside and outside of clusters
to_use = [0, 1, 2, 3]
fig_03= cv.cluster_visualizer(clust_pattern_01,
                               marks_to_plot=key_arrays["marks"][to_use],
                               colors=key_arrays["colors"][to_use],
                               names=key_arrays["names"][to_use],
                               sizes=key_arrays["sizes"][to_use],
                               opacities=key_arrays["opacities"][to_use])

fig_03.show()


####################
# alright, lets redo for the vertical stacks!

to_use = [1, 2]
fig_01b = cv.cluster_visualizer(clust_pattern_02,
                               marks_to_plot=key_arrays["marks"][to_use],
                               colors=key_arrays["colors"][to_use],
                               names=key_arrays["names"][to_use],
                               sizes=key_arrays["sizes"][to_use],
                               opacities=key_arrays["opacities"][to_use])

fig_01b.show()


# now just plot clustered type, both inside and outside of clusters
to_use = [2,3]
fig_02b = cv.cluster_visualizer(clust_pattern_02,
                               marks_to_plot=key_arrays["marks"][to_use],
                               colors=key_arrays["colors"][to_use],
                               names=key_arrays["names"][to_use],
                               sizes=key_arrays["sizes"][to_use],
                               opacities=key_arrays["opacities"][to_use])

fig_02b.show()

# now just plot clustered type, both inside and outside of clusters
to_use = [0, 1, 2, 3]
fig_03b= cv.cluster_visualizer(clust_pattern_02,
                               marks_to_plot=key_arrays["marks"][to_use],
                               colors=key_arrays["colors"][to_use],
                               names=key_arrays["names"][to_use],
                               sizes=key_arrays["sizes"][to_use],
                               opacities=key_arrays["opacities"][to_use])

fig_03b.show()