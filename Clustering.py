from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

nn = NearestNeighbors(n_neighbors=6).fit(train_x_2D)
distances, indices = nn.kneighbors(train_x_2D)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(10,8))
plt.plot(distances)
plt.savefig("Clusters")

eps = np.arange(0.001,0.015, 0.001)
min_samples = range(1,20)

output = []

for ms in min_samples:
    for ep in eps:
        labels = DBSCAN(min_samples=ms, eps = ep).fit(train_x_2D).labels_
        score = silhouette_score(train_x_2D, labels)
        output.append((ms, ep, score))

plt.clf()

plt.hist(train_y, color = 'blue', edgecolor = 'black',bins = int(len(train_y)/20))
plt.savefig("DistributionY")

plt.clf()

plt.hist(train_y, color = 'blue', edgecolor = 'black',bins = int(len(train_y)/20))
plt.savefig("DistributionY")
