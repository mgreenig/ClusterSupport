import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from functools import wraps
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

# class of decorators for functions in other scripts
class Decorators:

    # decorator for checking class instance for existing attributes (so they do not need to be calculated twice)
    def return_attr_if_exists(attr_name):
        def actual_decorator(function):
            @wraps(function)
            def function_wrapper(*args, **kwargs):
                if hasattr(args[0], attr_name) and not callable(getattr(args[0], attr_name)):
                    return getattr(args[0], attr_name)
                else:
                    return function(*args, **kwargs)

            return function_wrapper

        return actual_decorator

    # decorator for scikit-learn's cluster fit function
    def cluster_wrapper(cluster_func):
        @wraps(cluster_func)
        def new_cluster_func(self, X, *args, **kwargs):
            clustering_output = cluster_func(self, X, *args, **kwargs)
            # save the data and labels in an instance of the ClusteringResult class
            clustering_result = ClusteringResult(X, labels=clustering_output.labels_)
            return clustering_result

        return new_cluster_func

    # check if an inputted parameter is valid for a clustering algorithm
    def check_for_attr(function):
        @wraps(function)
        def function_wrapper(self, X, parameter, *args, **kwargs):
            if parameter in self.valid_parameters:
                return function(self, X, parameter, *args, **kwargs)
            else:
                raise ValueError('Please select a valid parameter from: {}'.format(', '.join(self.valid_parameters)))

        return function_wrapper

## class for clustering results ##

class ClusteringResult:

    def __init__(self, X, labels):
        self.X = X
        self.array_X = np.asarray(X)
        self.scaled_X = StandardScaler().fit_transform(self.array_X)
        self.labels_ = labels
        self.n_clusters = len(set(labels))

    # calculates the sum of distances around the cluster centroid for a cluster specified by cluster_mask
    def distance_around_mean(X, cluster_mask, scale = True, dist_metric='sqeuclidean'):

        viable_dist_metrics = {'euclidean', 'sqeuclidean', 'cosine', 'manhattan'}
        if dist_metric not in viable_dist_metrics:
            raise ValueError('Please choose a viable distance metric from: {}'.format(', '.join(viable_dist_metrics)))

        X = X.scaled_X if scale else X.array_X

        centroid = np.mean(X[cluster_mask], axis=0)

        # calculate distances around the mean, depending on the metric
        if dist_metric == 'euclidean':
            dists_around_mean = np.sum(np.linalg.norm(X[cluster_mask] - centroid, axis = 1))
        elif dist_metric == 'sqeuclidean':
            dists_around_mean = np.sum(np.linalg.norm(X[cluster_mask] - centroid, axis = 1) ** 2)
        elif dist_metric == 'cosine':
            dot_product = np.dot(centroid, np.transpose(X[cluster_mask]))
            norms_multiplied = np.linalg.norm(centroid) * np.linalg.norm(X[cluster_mask], axis=1)
            dists_around_mean = 1 - (dot_product / norms_multiplied)
        elif dist_metric == 'manhattan':
            dists_around_mean = np.sum(np.abs(np.subtract(X[cluster_mask], centroid), axis=1))

        # sum the distances
        sum_of_dists = np.sum(dists_around_mean)
        return sum_of_dists

    # calculate within cluster sum of squares
    @Decorators.return_attr_if_exists('inertia')
    def inertia(self, scale = True, dist_metric='sqeuclidean'):
        '''
        Calculates the sum of the sums of distances around each of the cluster means

        Parameters
        -----------
        scale : bool, whether to apply z-score normalisation on X

        dist_metric : str, metric used to calculate of the sum of distances around the means, one from {'euclidean', 'sqeuclidean', 'cosine', 'manhattan'}

        Returns
        -----------
        total_inertia : float, sum of distances across all cluster means
        '''

        # list for storing all the distances
        all_dists = []

        # loop through clusters and calculate the sum of distances around the mean for each cluster
        for label in set(self.labels_):
            # get indices in data that belong to the cluster
            cluster_mask = self.labels_ == label
            # calculate distances around the mean of the cluster
            dist_around_mean = self.distance_around_mean(cluster_mask, scale = scale, dist_metric=dist_metric)
            all_dists.append(dist_around_mean)

        total_inertia = sum(all_dists)
        self.inertia = total_inertia
        return total_inertia

    @Decorators.return_attr_if_exists('C_index')
    def C_index(self, scale = True, dist_metric='sqeuclidean'):
        '''
        Calculates the C_index of the clustering

        Parameters
        -----------
        scale : bool, whether to apply z-score normalisation on X

        dist_metric : str, metric used to calculate distances between points, one from {'euclidean', 'sqeuclidean', 'cosine', 'manhattan'}

        Returns
        -----------
        C_index : float, C_index value for the clusteirng
        '''

        X = self.scaled_X if scale else self.array_X

        # get distance matrix
        if not hasattr(self, 'distance_matrix'):
            self.distance_matrix = pdist(X, dist_metric)

        distance_matrix_sorted = np.sort(self.distance_matrix)

        unique_labels = set(self.labels_)
        C_values = []
        n_points_per_cluster = []

        for label in unique_labels:
            cluster_mask = self.labels_ == label
            number_of_points = len(self.labels_[cluster_mask])
            n_points_per_cluster.append(number_of_points)
            dists_around_mean = self.distance_around_mean(cluster_mask, scale = scale, dist_metric=dist_metric)
            # C value is the sum of pairwise distances within the clustering, equal to the distances around the mean*2*N
            C = dists_around_mean * 2 * number_of_points
            C_values.append(C)

        n_within_cluster_pairs = int(0.5 * (np.sum([n ** 2 for n in n_points_per_cluster]) - np.sum(n_points_per_cluster)))
        C = np.sum(C_values)
        Cmin = np.sum(distance_matrix_sorted[:n_within_cluster_pairs])
        Cmax = np.sum(distance_matrix_sorted[-n_within_cluster_pairs:])

        C_index = (C - Cmin) / (Cmax - Cmin)
        self.C_index = C_index

        return C_index

    @Decorators.return_attr_if_exists('silhouette_score')
    def silhouette_score(self, scale = True, dist_metric='euclidean'):
        '''
        Calculates the silhouette score of the clustering - wrapper for sklearn.metrics.silhouette_score()

        Parameters
        -----------
        scale : bool, whether to apply z-score normalisation on X

        dist_metric : str, metric used to calculate distances between points, can be any metric supported by sklearn.metrics.pairwise.pairwise_distances()

        Returns
        -----------
        score : float, silhouette score for the clustering
        '''
        X = self.scaled_X if scale else self.array_X
        score = silhouette_score(self.array_X, self.labels_, metric=dist_metric)
        self.silhouette = score
        return score

    @Decorators.return_attr_if_exists('CH_score')
    def CH_score(self, scale = True, dist_metric='sqeuclidean'):
        '''
        Calculates the Calinski-Harabasz score of the clustering, also known as the variance-ratio criterion

        Parameters
        -----------
        scale : bool, whether to apply z-score normalisation on X

        dist_metric : str, metric used to calculate distances between points, one from {'euclidean', 'sqeuclidean', 'cosine', 'manhattan'}

        Returns
        -----------
        CH_score : float, Calinski-Harabasz score for the clustering
        '''

        no_of_points = len(self.array_X)
        no_of_clusters = self.n_clusters
        within_cluster_variance = self.inertia()
        total_variance = self.distance_around_mean(np.arange(no_of_points), dist_metric=dist_metric)
        between_cluster_variance = total_variance - within_cluster_variance

        CH_score = (between_cluster_variance / within_cluster_variance) * (no_of_points - no_of_clusters) / (
                    no_of_clusters - 1)
        self.CH_score = CH_score

        return CH_score

    # get a summary of each cluster, including number of points and average distance around the centroid
    @Decorators.return_attr_if_exists('summary')
    def get_summary(self, scale=True, dist_metric='sqeuclidean'):
        '''
        Produces a summary of the clustering, containing information on the number of points in each cluster and the average distance around the mean

        Parameters
        -----------
        scale : bool, whether to apply z-score normalisation on X

        dist_metric : str, metric used to calculate distances between points and cluster means, one from {'euclidean', 'sqeuclidean', 'cosine', 'manhattan'}

        Returns
        -----------
        summary : Pandas DataFrame of size (n_clusters, 2), containing the number of points in each cluster and the cluster's average distance around the mean
        '''

        clustering_summary = {'cluster': [], 'n': [], 'avg_dist_around_mean': []}

        for label in set(self.labels_):
            clustering_summary['cluster'].append(label)
            cluster_mask = self.labels_ == label
            cluster_size = len(np.flatnonzero(cluster_mask))
            clustering_summary['n'].append(cluster_size)
            total_dist_around_mean = self.distance_around_mean(cluster_mask, scale = scale, dist_metric=dist_metric)
            clustering_summary['avg_dist_around_mean'].append(total_dist_around_mean / cluster_size)

        summary = pd.DataFrame(clustering_summary).round(decimals=2)
        summary = summary.set_index('cluster')
        self.summary = summary

        return summary

    # function for assessing each cluster with a classifier
    @ignore_warnings(category = ConvergenceWarning)
    def classifier_assessment(self, classifier = 'logreg', labels = None, scale = True, n = 0, grid_search = False, roc_plot = True, save_fig = False, random_state = None, *args, **kwargs):
        '''
        For each cluster in the clustering in X, train a classifier to discriminate instances within the cluster from instances outside

        This can be used to assess the quality of a clustering run on a reduced feature space

        Parameters
        -----------
        classifier : str, classifier model to be used, one from {'logreg', 'random_forest', 'svm'}
            logreg - sklearn.linear_model.LogisticRegression()
            random_forest - sklearn.ensemble.RandomForestClassifier()
            svm - sklearn.svm.SVC()

        labels : iterable or None, cluster labels for all instances (if None, defaults to using the labels used in initialising the class)

        scale : bool, whether to apply z-score normalisation on X

        n : int, number of instances from each cluster to be used for training the classifier, if n = 0 all instances are used

        grid_search : bool, whether to conduct a grid search on classifier hyperparameters before calculating accuracy

        roc_plot : bool, whether to show a ROC plot for classifier test set performance

        save_fig : bool, whether to save the figure (saves to current directory under file name 'example_cluster_ROC_curves.png')

        random_state : int or None, random state used in train/test split

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        assessment_summary : Pandas DataFrame of size (n_clusters, 4), containing test set precision, recall, f1 scores,
        and number of test set instances for the classifier for each cluster
        '''

        # list of viable classifiers
        viable_classifiers = {'logreg', 'random_forest', 'svm'}
        if classifier not in viable_classifiers:
            raise ValueError('Please choose a viable distance metric from: {}'.format(', '.join(viable_classifiers)))

        # create instance of the classifier depending on which classifier was specified
        if classifier == 'logreg':
            hyperparams = {'C': [10**i for i in range(-2, 2)]}
            clf = LogisticRegression(*args, **kwargs)
        elif classifier == 'random_forest':
            hyperparams = {'max_depth': [3, 6, 9], 'max_samples': [0.5, 0.7, 0.9]}
            clf = RandomForestClassifier(*args, **kwargs)
        elif classifier == 'svm':
            hyperparams = {'C': [10**i for i in range(-3, 3)]}
            clf = SVC(probability = True, *args, **kwargs)

        if labels is not None:
            if len(labels) != len(self.labels_):
                raise ValueError('Length of the labels passed to the function ({}) does not match the number of instances ({})'.format(len(labels), len(self.labels_)))
            self.labels_ = labels

        # scale if specified
        X = self.scaled_X if scale else self.array_X

        # set random state
        np.random.seed(random_state)

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, self.labels_)

        # if n is specified, sample points from each cluster
        if n > 0:
            min_cluster_size = min([len(X_train[y_train == lab]) for lab in set(y_train)])
            n = min(n, min_cluster_size)
            X_train_sample = np.empty(shape = [n*len(set(y_train)), X.shape[1]])
            y_train_sample = np.empty(shape = [n*len(set(y_train))])
            # loop through clusters and sample n points from each
            for i, lab in enumerate(set(y_train)):
                cluster_mask = y_train == lab
                in_cluster = X_train[cluster_mask,:]
                sample_idx = np.random.choice(len(in_cluster), size = n, replace = False)
                in_cluster_sample = in_cluster[sample_idx]
                X_train_sample[n*i:n*(i+1),:] = in_cluster_sample
                y_train_sample[n*i:n*(i+1)] = [lab]*n
            X_train = X_train_sample
            y_train = y_train_sample

        # set grid search if specified
        if grid_search:
            clf = GridSearchCV(clf, hyperparams, scoring = 'balanced_accuracy')

        # fit the model
        clf.fit(X_train, y_train)

        # get test set preds and classification report
        test_set_preds = clf.predict(X_test)
        test_set_probs = clf.predict_proba(X_test)
        assessment_summary = classification_report(y_test, test_set_preds, output_dict=True)
        assessment_summary = {'cluster_' + str(cluster): [stat for stat in assessment_summary[str(cluster)].values()] for cluster in self.labels_}
        assessment_summary = pd.DataFrame.from_dict(assessment_summary, orient = 'index', columns = ['precision', 'recall', 'f1_score', 'n'])

        if roc_plot:
            # get tprs and fprs for different thresholds
            plt.figure(figsize = (12, 8))

            # loop through clusters and plot tpr/fpr for each cluster
            for label in set(self.labels_):
                cluster_mask = clf.classes_ == label
                target = np.where(y_test == label, 1, 0)
                probs = test_set_probs[:,cluster_mask].flatten()
                fpr, tpr, _ = roc_curve(target, probs, drop_intermediate=False)
                area_under_curve = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='cluster {} (area = {:.3f})'.format(label, area_under_curve))

            plt.plot([0, 1], [0, 1], 'k--')
            lgd = plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 12, title = 'ROC curves')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize = 14)
            plt.ylabel('True Positive Rate', fontsize = 14)
            plt.tick_params(which = 'both', labelsize = 11)
            plt.title('ROC curves for different clusters', fontsize = 16)

            if save_fig:
                plt.savefig('cluster_ROC_curves.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

            plt.show()

        return assessment_summary