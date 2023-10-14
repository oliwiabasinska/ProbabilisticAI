import os
import typing
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        #self.rng = np.random.default_rng(seed=0)
        #self.gpr = GaussianProcessRegressor()
        #self.gprs = [ [GaussianProcessRegressor() for i in range(4)] for j in range(4) ]
        self.k = 16
        self.cluster_gprs = [GaussianProcessRegressor() for i in range(self.k)]
        self.alpha = 1.21
        self.fiveNN = KNeighborsClassifier(n_neighbors=5)

    def convert_coordinates(self, train_x_2D):

        transformed_x1 = ( train_x_2D[:,0] / 0.25 ).astype(int)
        transformed_x2 = ( train_x_2D[:,1] / 0.25 ).astype(int)

        vector = [ str(transformed_x1[i])+str(transformed_x2[i]) for i in range(train_x_2D.shape[0])]

        print("preview of vector of ints:", vector[0:5])

        return np.array(vector)



    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        # print("Trained GPR has", self.gpr.kernel_, 
        #       "and marginal log likelihood", self.gpr.log_marginal_likelihood_value_)
        predictions,gp_mean, gp_std = np.zeros(test_x_2D.shape[0]), np.zeros(test_x_2D.shape[0]), np.zeros(test_x_2D.shape[0])

        print("the alpha scaling param used is", self.alpha)

        int_coords_test = self.convert_coordinates(test_x_2D) 

        print("------Printing trained kernel information:")
        for cluster in range(self.k):
            print("Trained GPR in section", cluster, "has kernel with params:")
            print(self.cluster_gprs[cluster].kernel_)


        test_cluster_labels = self.fiveNN.predict(test_x_2D)

        print("------Iterating for", len(test_cluster_labels), "-clusters and the shape of test is", test_x_2D.shape)


        for cluster in range(self.k):
            mask = cluster == test_cluster_labels
            gp_mean[mask], gp_std[mask] = self.cluster_gprs[cluster].predict(test_x_2D[mask], return_std=True)


        # TODO: Use the GP posterior to form your predictions here
        predictions = np.zeros(test_x_2D.shape[0])
        predictions[test_x_AREA==1] = gp_mean[test_x_AREA==1] + self.alpha*gp_std[test_x_AREA==1]  #overestimation
        predictions[test_x_AREA==0] = gp_mean[test_x_AREA==0] # no overestimation

        #predictions = gp_mean + 1.21*gp_std[test_x_AREA==1]

        # plots
        plt.scatter(test_x_2D[:, 0],test_x_2D[:, 1], c = test_x_AREA )
        plt.savefig("Test coordinates and residential areas")
        plt.clf()


        plt.scatter(test_x_2D[:, 0],test_x_2D[:, 1], c = predictions )
        plt.savefig("Test coordinates colored by predicted pollution level")
        plt.clf()

        return predictions, gp_mean, gp_std
    
    # cross val for kernel

    def cross_val(self, kernels, X, y, k):
        """
            Return the best kernel from cross validation over K folds.
            :param X: training data (NUM_SAMPLES, 2)
            :param y: training labels (NUM_SAMPLES,)
            :return:
                best kernel
            """

        n_folds = k

        score_mat = np.zeros((n_folds, len(kernels)))

        print("----> initialize Kfold")
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = 0)

        print("----> Entering loop...")
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print("----------- Fold {} ----------".format(i))
            X_train, X_test  = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for j in range(len(kernels)):
                kern = kernels[j]
                print("Kernel is", kern)
                model = GaussianProcessRegressor(kernel=kern,
                                                random_state=0, 
                                                n_restarts_optimizer = 3).fit(X_train, y_train)

                score = model.log_marginal_likelihood_value_ 
                score_mat[i,j] = score

        print("Score matrix", score_mat)
        avg_score = np.mean(score_mat, axis = 0)    
        print("Average score", avg_score) # choose kernel maximising R2

        index_max = np.argmax(avg_score)
        print("Final chosen kernel based on CV will be", kernels[index_max])

        return kernels[index_max]

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # Clusters for x
        #clustered_x = DBSCAN(eps=0.01, min_samples=29).fit_predict(train_x_2D)
        #clusters = np.unique(clustered_x)
        #print("total number of clusters: ", len(clusters))
        #for cluster in clusters:
        #    row_ix = np.where(clustered_x == cluster)
        #    print(row_ix)

        # subsampled_indices = self.rng.integers(low = 0, high = train_x_2D.shape[0], size = 500)
        # subsampled_x = train_x_2D[subsampled_indices]
        # subsampled_y = train_y[subsampled_indices]

        # int_coords_train = self.convert_coordinates(train_x_2D)


        print("train_x:", train_x_2D.shape)
        print("train_y:", train_y.shape)
        train_y_reshaped = train_y.reshape(-1,1)
        data = np.concatenate((train_x_2D, train_y_reshaped), axis = 1)
        print(data.shape)
        kmeans = KMeans(n_clusters=self.k, random_state=69, init='k-means++', n_init="auto").fit(data)
        labels = kmeans.labels_
        print("labels:", labels.shape)

        length_scale_bounds=(1e-05, 10000000.0)
        kernel = Matern(length_scale_bounds=length_scale_bounds)*DotProduct() + ConstantKernel() + WhiteKernel() 
        noise_std = 10

        for cluster in range(self.k):
            train_k = train_x_2D[labels == cluster]
            y_k = train_y[labels == cluster]
            self.cluster_gprs[cluster] = GaussianProcessRegressor(kernel=kernel, random_state=69, alpha = noise_std**2, n_restarts_optimizer=5).fit(train_k, y_k)

        self.fiveNN.fit(train_x_2D, labels)
        


        #k = 2000  # Number of clusters / new datapoints
        #kmedoids = KMedoids(n_clusters=k)
        #kmedoids.fit(train_x_2D)

        #subsampled_x = train_x_2D[kmedoids.medoid_indices_]
        #subsampled_y = train_y[kmedoids.medoid_indices_]


        #plt.violinplot([train_x_2D[:,0],subsampled_x[:,0]])
        #plt.savefig("Ion for train vs subsampled train")
        #plt.clf()

        #plt.violinplot([train_x_2D[:,1],subsampled_x[:,1]])
        #plt.savefig("Lat for train vs subsampled train")


        # ---- uncomment to run cross-validation
        length_scale_bounds=(1e-05, 10000000.0)
        #kernels = np.array([ DotProduct(), ExpSineSquared(length_scale_bounds=length_scale_bounds), RBF(length_scale_bounds=length_scale_bounds), Matern(length_scale_bounds=length_scale_bounds), RationalQuadratic(length_scale_bounds=length_scale_bounds) ]) + WhiteKernel() + ConstantKernel()
        #kernel = self.cross_val(kernels, subsampled_x, subsampled_y, 5) # get best kernel from cross validation

        kernel = Matern(length_scale_bounds=length_scale_bounds)*DotProduct() + ConstantKernel() + WhiteKernel() 
        noise_std = 10
        #self.gpr = GaussianProcessRegressor(kernel=kernel,
        #                                    random_state=0, 
        #                                    alpha = noise_std**2,
        #                                    #normalize_y = True,
        #                                    n_restarts_optimizer = 5).fit(subsampled_x, subsampled_y)
        
        # finding the optimal alpha

        # loop for training GPRs

        # for i in range(4):
        #     for j in range(4):
        #         coord_str = str(i)+str(j)
        #         mask = int_coords_train == coord_str
        #         print("GPR is being trained on ", sum(mask*1),"samples")
        #         self.gprs[i][j]  = GaussianProcessRegressor(kernel=kernel,
        #                                     random_state=0, 
        #                                     alpha = noise_std**2,
        #                                     n_restarts_optimizer = 5).fit(train_x_2D[mask], train_y[mask])

        #self.gpr = []
        #for cluster in clusters:
        #    row_ix = np.where(clustered_x == cluster)
        #    self.gpr[cluster+1] = GaussianProcessRegressor(kernel=kernel,
        #                                                   random_state=0,
        #                                                   alpha = noise_std**2,
        #                                                   n_restarts_optimizer = 5).fit(train_x_2D[row_ix], 
        #                                                                                 train_y[row_ix])
        
        
        #print("Kernel optimized params", self.gpr.kernel_, 
        #      "with log marginal likelihood", self.gpr.log_marginal_likelihood_value_ )

# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function 
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """    
    
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features

    train_x_2D = train_x[:,[0,1]]
    train_x_AREA = train_x[:, 2] 
    test_x_2D = test_x[:,[0,1]]
    test_x_AREA = test_x[:, 2] 

    #plt.scatter(train_x_2D[:, 0],train_x_2D[:, 1], c = train_x_AREA )
    #plt.savefig("Coordinates and residential areas")
    #plt.clf()




    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA



def find_optimal_alpha(model, train_x_AREA, train_x_2D, train_y):
    """
    Function that finds the optimal scaling parameter alpha for overestimation.
    

    """
    alphas = np.linspace(0,10,num=100)
    min_alpha = 0
    min_score = 40000

    for a in alphas:
        gp_mean, gp_std = model.gpr.predict(train_x_2D, return_std=True)
        predictions = gp_mean + a * gp_std
        score = cost_function(train_y, predictions, train_x_AREA)
        if score < min_score:
            print(score,a)
            min_score = score
            min_alpha = a

    return(min_alpha)


# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # shuffle data
    np.random.seed(43)
    idx = np.arange(0,train_x.shape[0], 1)
    np.random.shuffle(idx)
    train_x, train_y = train_x[idx,:] , train_y[idx]

    #plt.scatter(train_x[:, 0],train_x[:, 1], c = train_y )
   # plt.savefig("Coordinates colored by pollution level")
   # plt.clf()

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)
    
    # clustering

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_y,train_x_2D)

    # find optimal alpha
    #min_alpha = find_optimal_alpha(model, train_x_AREA, train_x_2D, train_y)
    #model.alpha = min_alpha

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)


    # make plots for exploratory data analysis

    #predictions_x = model.make_predictions(train_x_2D, train_x_AREA) # predictions on the train set itself

    #plt.violinplot((predictions[0], train_y))
    #plt.savefig("predictions distribution.png")
    #plt.clf()

    #plt.violinplot((predictions_x[0], train_y))
    #plt.savefig("predictions made on train_x, vs the actual labels train_y.png")
    #plt.clf()

    #plt.violinplot((train_x_2D[:,0], test_x_2D[:,0]))
    #plt.savefig("Ion concentration on train vs test data.png")
    #plt.clf()

    #plt.violinplot((train_x_2D[:,1], test_x_2D[:,1]))
    #plt.savefig("Lat on train vs test data.png")
    #plt.clf()

    #plt.scatter(train_x_2D[:,0], train_y)
    #plt.savefig("[ion] vs pm25.png")
    #plt.clf()

    #plt.scatter(train_x_2D[:,1], train_y)
    #plt.savefig("lat vs pm25.png")
    #plt.clf()

    #plt.violinplot((train_x_2D[:,0], test_x_2D[:,0]))
    #plt.savefig("Ions distribution for training and test data.png")
    #plt.clf()

    #plt.violinplot((train_x_2D[:,1], test_x_2D[:,1]))
    #plt.savefig("Latitude distribution for training and test data.png")

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
