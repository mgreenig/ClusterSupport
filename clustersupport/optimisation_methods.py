import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster

from clustersupport.clustering_results import Decorators

## Functions for optimizing clustering hyperparameters ##

class OptimisationMethods:

    viable_metrics = {'inertia', 'CH_score', 'silhouette_score', 'C_index'}

    # gap statistic calculation
    @Decorators.check_for_attr
    def gap_statistic(self, X, parameter, parameter_range, metric='inertia', pca=True, n_bootstraps=50, verbose=True, random_state = None, *args,
                      **kwargs):
        '''
        Calculates a gap statistic over a hyperparameter range for the clustering (see Tibshirani et al, 2001)

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        parameter : str, parameter to be optimized, choices depend on clustering class being used
            KMeans() - {'n_clusters'}
            AffinityPropagation() - {'damping'}
            MiniBatchKMeans() - {'n_clusters'}
            AgglomerativeClustering() - {'n_clusters', 'affinity'}
            OPTICS() - {'min_samples'}
            DBSCAN() - {'min_samples', 'eps'}
            MeanShift() - {'bandwidth'}
            Birch() - {'threshold', 'branching_factor', 'n_clusters'}
            SpectralClustering() - {'n_clusters'}

        parameter_range : iterable, values of the parameter over which the gap statistic should be calculated

        metric : str, clustering metric to use in gap statistic calculation, one from {'inertia', 'CH_score', 'silhouette_score', 'C_index'}

        pca : bool, whether to run PCA on data before calculating gap statistics

        n_bootstraps : int, number of bootstrapping rounds to be used to estimate null gap statistic standard error

        verbose : bool, whether to print progress

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        gap_stats : Pandas DataFrame of size (len(parameter_range), 2), containing the gap statistic and its standard error for each value in parameter_range
        '''

        if metric not in self.viable_metrics:
            raise ValueError('Please choose a clustering metric from: {}'.format(', '.join(self.viable_metrics)))

        X = np.asarray(X)

        # first PCA transform data
        if pca:
            X = sklearn.decomposition.PCA(n_components=min(100, X.shape[1])).fit_transform(X)

        # create objects for storing data
        results = {parameter: [], 'gap_statistic': [], 'standard_error': []}
        random_data_sets = np.empty(shape=[n_bootstraps, X.shape[0], X.shape[1]])

        # set random seed
        np.random.seed(random_state)

        for i in range(n_bootstraps):
            random_data_sets[i, :, :] = self.generate_random_data(X)

        for value in parameter_range:
            self.__dict__[parameter] = value
            clustering_real_data = self.fit(X, *args, **kwargs)
            real_data_score = getattr(clustering_real_data, metric)()

            random_data_scores = []

            # bootstrapping random data generation
            for data_set in random_data_sets:

                clustering_random_data = self.fit(data_set, *args, **kwargs)
                random_data_score = getattr(clustering_random_data, metric)()

                # taking logs makes computation easier for inertias
                if metric == 'inertia':
                    random_data_scores.append(np.log(random_data_score))
                else:
                    random_data_scores.append(random_data_score)

            # calculate the gap statistic as the difference between the real score and the mean of the scores for the random data
            real_score = np.log(real_data_score) if metric == 'inertia' else real_data_score
            null_score = np.average(random_data_scores)
            se_null_score = np.std(random_data_scores) * np.sqrt(1 + (1 / n_bootstraps))
            gap_statistic = null_score - real_score

            # add to the results dictionary
            results[parameter].append(value)
            results['gap_statistic'].append(gap_statistic)
            results['standard_error'].append(se_null_score)

            if verbose == True:
                print('Iteration for {} = {} completed.'.format(parameter, value))

        gap_stats = pd.DataFrame(data=results).round(decimals=5)
        gap_stats = gap_stats.set_index(parameter)

        return gap_stats

    @staticmethod
    def generate_random_data(X):
        minimums = np.amin(X, axis=0)
        maximums = np.amax(X, axis=0)
        random_data = np.random.uniform(low=minimums, high=maximums, size=X.shape)
        return random_data

    # generate elbow plot
    @Decorators.check_for_attr
    def elbow_plot(self, X, parameter, parameter_range, metric='inertia', save_fig = False, *args, **kwargs):
        '''
        Plots hyperparameter values against values of a clustering metric

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        parameter : str, parameter to be plotted, choices depend on clustering class being used
            KMeans() - {'n_clusters'}
            AffinityPropagation() - {'damping'}
            MiniBatchKMeans() - {'n_clusters'}
            AgglomerativeClustering() - {'n_clusters', 'affinity'}
            OPTICS() - {'min_samples'}
            DBSCAN() - {'min_samples', 'eps'}
            MeanShift() - {'bandwidth'}
            Birch() - {'threshold', 'branching_factor', 'n_clusters'}
            SpectralClustering() - {'n_clusters'}

        parameter_range : iterable, values of the parameter over which the metric should be plotted

        metric : str, clustering metric to plot for parameter values, one from {'inertia', 'CH_score', 'silhouette_score', 'C_index'}

        save_fig : bool, whether to save figure (only applies when plot = True, saves figure as 'elbow_plot.png')

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        plot : matplotlib plot() object
        '''

        if metric not in self.viable_metrics:
            raise ValueError('Please choose a clustering metric from: {}'.format(', '.join(self.viable_metrics)))

        # get within-cluster sum of squares across range of k values to determine optimal k
        scores = []

        X = np.asarray(X)

        for value in parameter_range:
            self.__dict__[parameter] = value
            clustering = self.fit(X, *args, **kwargs)
            clustering_score = getattr(clustering, metric)()

            # add within cluster sum of squares
            scores.append(clustering_score)

        # plot and look for elbow
        fig = plt.figure(figsize = (12,6))
        metric = metric[0].upper() + metric[1:]
        metric = metric.replace('_', ' ')
        plt.xlabel(parameter, fontsize = 14)
        plt.ylabel(metric, fontsize = 14)
        plt.title('{} for different values of {}'.format(metric, parameter), fontsize = 16)
        plt.plot(parameter_range, scores)

        if save_fig:
            plt.savefig('elbow_plot.png')

    # consensus clustering
    @Decorators.check_for_attr
    def consensus_cluster(self, X, parameter, parameter_range, subsample=0.75, n_bootstraps=50, verbose=True, plot=True,
                          save_fig=False, random_state=None, *args, **kwargs):
        '''
        Runs consensus clustering (Monti et al., 2003) on the data over a hyperparameter range, calculating the
        proportion of unambiguous clusterings and the area under the CDF for each value of the hyperparameter

        Can also plot the empirical cdfs for different hyperparameter values

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        parameter : str, parameter to be optimized, choices depend on clustering class being used
            KMeans() - {'n_clusters'}
            AffinityPropagation() - {'damping'}
            MiniBatchKMeans() - {'n_clusters'}
            AgglomerativeClustering() - {'n_clusters', 'affinity'}
            OPTICS() - {'min_samples'}
            DBSCAN() - {'min_samples', 'eps'}
            MeanShift() - {'bandwidth'}
            Birch() - {'threshold', 'branching_factor', 'n_clusters'}
            SpectralClustering() - {'n_clusters'}

        parameter_range : iterable, values of the parameter over which consensus values should be calculated

        subsample : float, proportion of data to subsample when bootstrapping

        n_bootstraps : int, number of bootstrapping rounds to be used to estimate consensus values

        plot : bool, whether to plot empirical CDFs for each hyperparameter value

        save_fig : bool, whether to save figure (only applies when plot = True, saves figure as 'cdf_plot.png')

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        consensus_df : Pandas DataFrame of size (len(parameter_range), 2), containing the proportion of unambiguous clusterings and the area under the consensus CDF for each value in parameter_range
        '''

        n_points = len(X)
        X = np.asarray(X)
        sample_size = round(len(X) * subsample)
        results = {parameter: [], 'proportion_unambiguous_clusterings': [], 'area_under_cdf': []}

        if plot:
            fig = plt.figure(figsize = (12,6))
            plt.ylim(0, 1)
            plt.xlim(0, 1)

        # set seed
        np.random.seed(random_state)

        # loop through parameter values
        for value in parameter_range:

            # connectivity matrix indicates whether two points appear in a cluster together
            connectivity_matrix = np.zeros([n_points, n_points], dtype='uint8')
            # appearance matrix keeps track of how many times two points are sampled together
            appearance_matrix = np.zeros([n_points, n_points], dtype='uint8')

            self.__dict__[parameter] = value

            # for each resample, randomly-sample data, cluster, and record co-appearances (in both clusters and samples)
            for i in range(n_bootstraps):

                random_ints_for_idx = np.random.choice(n_points, size=sample_size, replace = False)
                sample = X[random_ints_for_idx]
                cluster_result = self.fit(sample, *args, **kwargs)
                labs = cluster_result.labels_

                appearance_matrix[random_ints_for_idx[:, None], random_ints_for_idx] += 1

                for lab in set(labs):
                    cluster_mask = labs == lab
                    original_data_idx = random_ints_for_idx[cluster_mask]
                    connectivity_matrix[original_data_idx[:, None], original_data_idx] += 1

            # get upper triangulars
            connectivity_matrix = np.triu(connectivity_matrix, k=1)
            appearance_matrix = np.triu(appearance_matrix, k=1)

            # for identifying element pairs that never appeared together (zeros in appearance matrix), so as not to divide by zero
            zero_idx = appearance_matrix == 0

            # stability of a clustering is the number of same-group clusterings divided by the total number of clusterings
            consensus_values = np.divide(connectivity_matrix[~zero_idx], appearance_matrix[~zero_idx])
            ambigious_clusterings = consensus_values[(consensus_values > 0.1) & (consensus_values < 0.9)]
            proportion_unambiguous_clusterings = 1 - len(ambigious_clusterings) / len(consensus_values)

            # plot the empirical cdf of consensus values
            ecdf_x_values = np.unique(consensus_values)
            ecdf_y_values = [len(consensus_values[consensus_values <= value]) / len(consensus_values) for value in
                             ecdf_x_values]

            if plot:
                plt.step(ecdf_x_values, ecdf_y_values)

            # calculate change in area under CDF for each parameter value
            area_components = np.empty(len(ecdf_x_values) - 1)
            for i, x_val in enumerate(ecdf_x_values):
                if i != 0:
                    x_component = x_val - ecdf_x_values[i - 1]
                    y_component = ecdf_y_values[i-1]
                    area_components[i - 1] = x_component * y_component
            area_under_cdf = np.sum(area_components)
            results[parameter].append(value)
            results['proportion_unambiguous_clusterings'].append(proportion_unambiguous_clusterings)
            results['area_under_cdf'].append(area_under_cdf)

            if verbose == True:
                print('Iteration for {} = {} completed.'.format(parameter, value))

        if plot:
            lgd = plt.legend([parameter + ' = ' + str(i) for i in parameter_range], bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 12, title = 'Values of {}'.format(parameter))
            plt.xlabel('Consensus value', fontsize = 14)
            plt.ylabel('Cumulative frequency', fontsize = 14)
            plt.tick_params(which = 'both', labelsize = 11)
            plt.title('Empirical CDFs of consensus values across different values of {}'.format(parameter), fontsize = 16)
            if save_fig:
                fig.savefig('cdf_plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

        consensus_df = pd.DataFrame(data=results).round(decimals=3)
        consensus_df = consensus_df.set_index(parameter)
        return consensus_df