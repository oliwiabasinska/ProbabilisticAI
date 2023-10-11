#subsampled_indices = self.rng.integers(low = 0, high = train_x_2D.shape[0], size = 2000)
#subsampled_x = train_x_2D[subsampled_indices]
#subsampled_y = train_y[subsampled_indices]


from sklearn_extra.cluster import KMedoids
import numpy as np

k = 2000  # Number of clusters / new datapoints
kmedoids = KMedoids(n_clusters=k)
kmedoids.fit(train_x)

subsampled_x = train_x[kmedoids.medoid_indices_]
subsampled_y = train_y[kmedoids.medoid_indices_]
