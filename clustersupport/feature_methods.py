import numpy as np
import pandas as pd
from scipy.stats import t, norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

## Functions for feature analysis ##

class FeatureMethods:

    viable_metrics = {'inertia', 'CH_score', 'silhouette_score', 'C_index'}

    def t_test(self, X, scale=True, output ='p-value', *args, **kwargs):
        '''
        Run a two-sample t-test for each feature, between cells inside and outside each cluster

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        scale : bool, whether to apply z-score normalisation on X

        output : str, statistic to calculate and return for each feature/cluster combination, one from {'t-statistic', 'p-value'}

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        results : Pandas DataFrame of shape (n_clusters, n_features), containing the calculated statistic for each cluster/feature combination
        '''

        # output argument should be either t statistic or p value
        if output not in {'t-statistic', 'p-value'}:
            raise ValueError('Please choose an output from: {}'.format(', '.join(['t-statistic', 'p-value'])))

        X = np.asarray(X)
        # scale data if specified
        if scale:
            X = StandardScaler().fit_transform(X)

        # if clustering has not been run, run it
        if not hasattr(self, 'labels_'):
            self.fit(X, *args, **kwargs)

        results_dict = {}

        # Welch's two sample t test
        for label in set(self.labels_):
            # get feature values for points inside and outside the cluster
            cluster_mask = self.labels_ == label
            cluster_members = X[cluster_mask]
            cluster_nonmembers = X[~cluster_mask]
            cluster_feature_values = cluster_members.transpose()
            cluster_feature_means = np.mean(cluster_feature_values, axis=1)
            cluster_feature_stdevs = np.std(cluster_feature_values, axis=1, ddof=1)
            nonmember_feature_values = cluster_nonmembers.transpose()
            nonmember_feature_means = np.mean(nonmember_feature_values, axis=1)
            nonmember_feature_stdevs = np.std(nonmember_feature_values, axis=1, ddof=1)
            combined_stdevs = np.sqrt((cluster_feature_stdevs ** 2 / len(cluster_members)) + (
                        nonmember_feature_stdevs ** 2 / len(cluster_nonmembers)))
            # calculate t-statistics
            t_statistics = (cluster_feature_means - nonmember_feature_means) / combined_stdevs
            # for p-values we also need to estimate the combined degrees of freedom for the t-distributed null hypothesis
            if output == 'p-value':
                dof_denominator_term1 = (cluster_feature_stdevs ** 4 / (len(cluster_members) ** 2 * (len(cluster_members) - 1)))
                dof_denominator_term2 = (nonmember_feature_stdevs ** 4 / (len(cluster_nonmembers) ** 2 * (len(cluster_nonmembers) - 1)))
                degrees_of_freedom = combined_stdevs ** 4 / (dof_denominator_term1 + dof_denominator_term2)
                pvalues = t(degrees_of_freedom).pdf(t_statistics)
                results_dict[label] = pvalues
            else:
                results_dict[label] = t_statistics

        results = pd.DataFrame.from_dict(results_dict, orient='index')
        return results

    # remove each feature and calculate a metric of clustering structure
    def leave_one_out(self, X, scale=True, metric='inertia', *args, **kwargs):
        '''
        For each feature in X, run a clustering with the feature omitted and calculate the difference in a clustering metric compared to the clustering with all features

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        scale : bool, whether to apply z-score normalisation on X

        metric : str, metric to use for comparison, one from {'inertia', 'CH_score', 'silhouette_score', 'C_index'}

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        results : Pandas Series of length n_features, containing the difference in the metrics calculated between the
        clustering of the full and reduced data sets for each feature
        '''

        if metric not in self.viable_metrics:
            raise ValueError('Please choose a clustering metric from: {}'.format(', '.join(self.viable_metrics)))

        X = np.asarray(X)
        # scale data if specified
        if scale:
            X = StandardScaler().fit_transform(X)

        # if clustering has not been run, run it
        if not hasattr(self, metric):
            clustering = self.fit(X, *args, **kwargs)

        # get baseline metric value
        baseline_metric_value = getattr(clustering, metric)()

        # transpose data to get features as first index in the array
        feature_values = X.transpose()
        number_of_features = feature_values.shape[0]
        results = {}
        for i in range(number_of_features):
            feature_mask = np.arange(number_of_features) == i
            # remove the feature
            feature_dropped = feature_values[~feature_mask]
            # transpose back into data form
            new_data = feature_dropped.transpose()
            # cluster
            new_clustering = self.fit(new_data)
            # calculate new metric
            metric_value = getattr(new_clustering, metric)()
            difference = metric_value - baseline_metric_value
            results[i] = difference

        results = pd.Series(list(results.values()), index=results.keys())
        results.name = 'change_in_{}'.format(metric)
        results.index.name = 'feature'
        return results

    # fits a logistic regression model for cluster membership for each of the clusters
    def logistic_regression(self, X, scale=True, output = 'p-value', subsample = 0.75, n_bootstraps = 50, random_state = None, penalty='l2', C=1.0, *args, **kwargs):
        '''
        Fits a logistic regression model for each cluster using the features in X, fitting to the target variable of cluster membership (1 if in the cluster, 0 if not)

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        scale : bool, whether to apply z-score normalisation on X

        output : str, values to return from the logistic regression models, one from {'coef', 'p-value'}
            coef - returns model coefficients
            p-value - returns p-values for model coefficients (calculated using a bootstrap procedure)

        subsample : float, proportion of data to subsample when bootstrapping coefficients (only applies when output = 'p-value')

        n_bootstraps : int, number of re-sampling rounds using in bootstrapping (only applies when output = 'p-value')

        random_state : int, seed for random state used in bootstrapping (only applies when output = 'p-value')

        penalty : str, argument to sklearn.linear_model.LogisticRegression(), can be one of {'l2', 'l1', 'elasticnet', 'none'}

        C : float, argument to sklearn.linear_model.LogisticRegression() (smaller values specify stronger regularization)

        *args, **kwargs : arguments to fit() method of the clustering clas
        Returns
        -----------
        results : Pandas DataFrame of shape (n_clusters, n_features), containing the calculated coefficient or coefficient p-value for each cluster/feature combination
        '''

        if output not in {'coef', 'p-value', 'z-score'}:
            raise ValueError('Please choose an output type from: {}'.format(', '.join(['coef', 'p-value'])))

        X = np.asarray(X)
        # scale data if specified
        if scale:
            X = StandardScaler().fit_transform(X)

        # if clustering has not been run, run it
        if not hasattr(self, 'labels_'):
            self.fit(X, *args, **kwargs)

        # initialise logistic regression model
        LR = LogisticRegression(penalty=penalty, C=C)

        # loop through clusters, fit logistic regression model for each one and save coefficients
        regression_coefficients = {}
        for label in set(self.labels_):
            target = np.where(self.labels_ == label, 1, 0)
            LR.fit(X, target)
            regression_coefficients[label] = LR.coef_[0]

        # if output is specified as p-value, bootstrap X and fit coefficients to the sampled data to estimate standard errors
        if output == 'p-value' or output == 'z-score':
            # dictionary for coefficients calculated during bootstrapping
            bootstrap_coefs = {label: [] for label in set(self.labels_)}
            np.random.seed(random_state)
            # loop through clusters and bootstrap for each cluster
            for label in set(self.labels_):
                target = np.where(self.labels_ == label, 1, 0)
                for i in range(n_bootstraps):
                    sample = np.random.choice(X.shape[0], size = round(X.shape[0]*subsample), replace = False)
                    Xb = X[sample]
                    targetb = target[sample]
                    LR.fit(Xb, targetb)
                    bootstrap_coefs[label].append(LR.coef_[0])
            coef_ses = {label: np.std(bootstrap_coefs[label], ddof = 1) for label in bootstrap_coefs}
            if output == 'p-value':
                output = {label: norm.pdf(regression_coefficients[label] / coef_ses[label]) for label in regression_coefficients}
            else:
                output = {label: regression_coefficients[label] / coef_ses[label] for label in regression_coefficients}
        else:
            output = regression_coefficients

        # put coefficients into a data frame
        results = pd.DataFrame(output).T
        results.index.name = 'cluster'

        return results

    # non-parametric mann_whitney U-test
    def mann_whitney(self, X, output = 'p-value', *args, **kwargs):
        '''
        Calculates a Mann-Whitney U statistic for each cluster/feature combination, for the null hypothesis that a
        feature value randomly selected from within the cluster is equally likely to be greater/less than a value randomly
        selected from outside the cluster

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features), input data

        scale : bool, whether to apply z-score normalisation on X

        output : str, values to return from the tests, one from {'z-score', 'p-value'}
            z-score - returns z-scores of the mann-whitney U statistic
            p-value - returns p-values for mann-whitney U statistics (calculated using normal approximation)

        *args, **kwargs : arguments to fit() method of the clustering class

        Returns
        -----------
        results : Pandas DataFrame of shape (n_clusters, n_features), containing the calculated U statistic or p-value for each cluster/feature combination
        '''

        # output argument should be either a z-score or a p-value
        if output not in {'z-score', 'p-value'}:
            raise ValueError('Please choose an output from: {}'.format(', '.join(['z-score', 'p-value'])))

        X = pd.DataFrame(X)

        # if clustering has not been run, run it
        if not hasattr(self, 'labels_'):
            self.fit(X, *args, **kwargs)

        # dictionary for test results
        results_dict = {}

        # loop through clusters
        for label in set(self.labels_):
            cluster_mask = self.labels_ == label
            results_dict[label] = []
            # loop through features
            for column in X.columns:
                # get ranks for the feature
                ranks = X[column].rank(axis = 0, method = 'average')
                # ranks for points in the cluster
                cluster_ranks = ranks[cluster_mask]
                # ranks for points outside the cluster
                non_cluster_ranks = ranks[~cluster_mask]
                # mean for null distribution
                mu = (len(cluster_ranks)*len(non_cluster_ranks))/2
                # test statistics
                U1 = cluster_ranks.sum() - (len(cluster_ranks) * (len(cluster_ranks) + 1) / 2)
                U2 = non_cluster_ranks.sum() - (len(non_cluster_ranks) * (len(non_cluster_ranks) + 1) / 2)
                # calculate the term for tied ranks used in the variance term
                tied_ranks_term = 0
                for rank in ranks.unique():
                    n_at_rank = len([r for r in ranks if r == rank])
                    tied_ranks_term += (n_at_rank**3 - n_at_rank) / (len(ranks) * (len(ranks) - 1))
                # variance for the null distribution
                var = ((len(cluster_ranks) * len(non_cluster_ranks))/12)*((len(ranks) + 1) - tied_ranks_term)
                sigma = np.sqrt(var)
                # z score for the smaller of the two U statistics
                z_score = (U1 - mu) / sigma if U1 < U2 else (U2 - mu) / sigma
                # if p-value output, use scipy's norm pdf function
                if output == 'p-value':
                    pvalue = norm.pdf(z_score)
                    results_dict[label].append(pvalue)
                else:
                    results_dict[label].append(z_score)

        results = pd.DataFrame.from_dict(results_dict, orient='index')
        return results