# ClusterSupport 

ClusterSupport is a small package designed to enhance scikit-learn's clustering capabilities. The package combines scikit-learn's clustering algorithm classes like `KMeans()`, `AgglomerativeClustering()`, and `DBSCAN()`,
with additional functions for the analysis and optimisation of clustering results.

## Dependencies

ClusterSupport requires the following packages:
- scikit-learn~=0.22.1
- numpy~=1.18.1
- pandas~=1.0.3
- scipy~=1.4.1
- matplotlib~=3.1.3

## Getting started

To install ClusterSupport, just use:

```
pip install clustersupport
```

## Analysing clustering results

ClusterSupport inherits clustering classes from scikit-learn and wraps their `.fit()` methods so that calling `.fit()`
returns an instance of the `ClusteringResult()` class. Let's load in one of scikit-learn's toy datasets to see some of the functionality.

``` python
from sklearn.datasets import load_boston
data = pd.DataFrame(load_boston()['data'], columns = load_boston()['feature_names'])
```

And then we can call the `.fit()` method to return an instance of our `ClusteringResult()` class.

``` python
import clustersupport as cs
results = cs.KMeans(n_clusters = 3).fit(data)
```

We can calculate a metric of clustering structure, for example the Calinski-Harabasz score or C-index, using:

``` python
CH_score = results.CH_score()
C_index = results.C_index()
```

Currently, ClusterSupport supports the following clustering metrics:
- Silhouette score
- Calinski-Harabasz score
- C-index
- Inertia (sum of intra-cluster distances)

For another view we can get a summary of the clustering using:

``` python
summ = results.get_summary(dist_metric = 'sqeuclidean')
```

which returns a Pandas DataFrame containing values for the number of points in each cluster as well as the average distance 
around the cluster mean.

|   cluster |   n |   avg_dist_around_mean |
|----------:|-------------------:|-----------------------:|
|         0 |                102 |                   6.18 |
|         1 |                366 |                   8.66 |
|         2 |                 38 |                   6.69 |

The distance metric can be specified with the `dist_metric` argument. ClusterSupport currently supports `'sqeuclidean'`, `'euclidean'`, `'cosine'`, and `'manhattan'` distances.

Finally, we can conduct a classifier assessment of the clustering. This involves training a classifier to predict 
the cluster to which each data point belongs and assessing the classifier's accuracy on a 'test set' of points that were
not used in the training. This is particularly useful in contexts where the clustering is run on a reduced version of the full 
feature space, and we wish to analyse how effectively the clustering captures the full detail of the complete feature space.

``` python
from sklearn.decomposition import PCA

# set seed and apply PCA on the data
data_reduced = PCA(n_components = 3, random_state = 123).fit_transform(data)

# fit to the reduced data and save the labels
reduced_fit_labels = cs.KMeans(n_clusters = 8, random_state = 123).fit(data_reduced).labels_

# run fit with the full data set
clustering = cs.KMeans(n_clusters = 8).fit(data)

# run classifier assessment with the reduced fit labels
clf_assessment = clustering.classifier_assessment(classifier = 'logreg', labels = reduced_fit_labels, roc_plot = True, n = 50, save_fig = True, random_state = 123)
```

![Example ROC curves](https://github.com/mgreenig/ClusterSupport/raw/master/docs/artwork/example_cluster_ROC_curves.png)

Specifying `roc_plot = True` uses matplotlib to plot an ROC curve for each cluster so that the user can see how well each cluster is classified. 
The function also outputs a Pandas DataFrame with classification metrics (precision, recall, and f1 score) calculated for each cluster.
The AUC for each cluster reflects the classifier's ability to distinguish instances in that cluster, thus providing an estimate of:

1. How well the reduced feature space represents the complete feature space
2. How well-separated each cluster is

## Analysing feature importances in clustering

ClusterSupport also provides methods for analysing the importance of different features in the cluster, either on at a global or per-cluster level. 
These functions are called as methods under the clustering classes inherited from scikit-learn (e.g. `clustersupport.KMeans()`).

The `t_test` method calculates a t-statistic for each feature in each cluster, calculated as a scaled difference-of-means 
between feature values for instances inside the cluster compared to instances outside the cluster. This returns Pandas DataFrame of size
(n_clusters, n_features), with the calculated t-statistic or p-value for the respective cluster/feature combination in each cell. 
Welch's two-sample t-test with unequal variances is used for the calculation.

``` python
feature_t_tests = cs.KMeans(n_clusters = 3).t_test(X = data, output = 'p-value')
```

You can also output the raw t-statistics with `output = 't-statistic'`. 
Alternatively you can conduct a non-parametric Mann-Whitney U test to test the ranks of feature values inside/outside each cluster.

``` python
feature_MW = cs.KMeans(n_clusters = 3).mann_whitney(X = data, output = 'p-value')
```

This also returns a DataFrame of size (n_clusters, n_features), with the calculated statistic or p-value for the cluster/feature combination in each cell.

The `leave_one_out()` function assesses the global contribution of each feature to the clustering by calculating a global metric like the Calinski-Harabasz score
or the sum of intra-cluster distances (a.k.a inertia).

``` python
feature_LOO = cs.KMeans(n_clusters = 3).leave_one_out(X = data, metric = 'CH_score')
```

This generates a Pandas Series showing the change in the clustering metric that was calculated when each feature was removed.

|   feature |   change_in_CH_score |
|----------:|---------------------:|
|         0 |             12.3713  |
|         1 |              8.71682 |
|         2 |             -8.93893 |
|         3 |             36.7465  |
|         4 |             -9.20392 |
|         5 |             26.1994  |
|         6 |             -3.35618 |
|         7 |             -7.85918 |
|         8 |            -15.0409  |
|         9 |            -21.3239  |
|        10 |             21.898   |
|        11 |             20.8256  |
|        12 |              3.82213 |

Finally we can build a logistic regression model to calculate a coefficient for each feature in each cluster and return the p-value of the coefficient under the null hypothesis:

![Eq1](https://latex.codecogs.com/svg.latex?\frac{\beta}{\text{SE}(\beta)}%20\sim%20\mathcal{N}(\mu%20=%200,%20\sigma^{2}%20=%201))

Where SE() denotes the standard error of the coefficient. This is done using the `logistic_regression()` function, which builds a logistic regression model for each cluster with y = 1 if an instance is in the cluster, and y = 0 if not:

``` python
feature_LR = cs.KMeans(n_clusters = 3).logistic_regression(X = data, output = 'p-value')
```

which returns a DataFrame of size (n_clusters, n_features), with the calculated value for the cluster/feature combination in each cell. 
Types of output can be chosen from 'coef', 'z-score', and 'p-value'.

| cluster   |        0 |          1 |        2 |        3 |         4 |        5 |        6 |         7 |         8 |          9 |       10 |       11 |       12 |
|---:|---------:|-----------:|---------:|---------:|----------:|---------:|---------:|----------:|----------:|-----------:|---------:|---------:|---------:|
|  0 | 0.398823 | 0.0147451  | 0.259453 | 0.398709 | 0.21047   | 0.391893 | 0.104119 | 0.0995283 | 0.396368  | 0.308628   | 0.308155 | 0.389234 | 0.292119 |
|  1 | 0.358399 | 0.398903   | 0.105571 | 0.37434  | 0.0327718 | 0.30657  | 0.385749 | 0.302824  | 0.0947453 | 0.00961577 | 0.202549 | 0.369796 | 0.169429 |
|  2 | 0.277988 | 0.00147331 | 0.198039 | 0.396121 | 0.0952189 | 0.35859  | 0.159246 | 0.191815  | 0.299553  | 0.124311   | 0.362565 | 0.390554 | 0.354773 |

These p-values are not adjusted.

## Optimizing clustering hyperparameters

ClusterSupport also provides functions for optimizing clustering hyperparameters. 
Like the feature methods, these functions are called as methods under the clustering classes inherited from scikit-learn (e.g. `clustersupport.KMeans()`)

The simplest optimization method is `elbow_plot()`, which constructs a plot of hyperparameter values compared to a clustering metric.

``` python
cs.KMeans().elbow_plot(X = data, parameter = 'n_clusters', parameter_range = range(2,10), metric = 'silhouette_score')
```

![Example elbow plot](https://github.com/mgreenig/ClusterSupport/raw/master/docs/artwork/example_elbow_plot.png)

The `gap_statistic()` method is another function can be used to optimise hyperparameters. 
It calculates the [gap statistic](https://statweb.stanford.edu/~gwalther/gap) and its standard errors across a range of hyperparameter values.
For example, to optimise the number of clusters used in K-means clustering, we call the following:

``` python
gap_statistics = cs.KMeans().gap_statistic(X = data, parameter = 'n_clusters', parameter_range = range(2,10), metric = 'inertia', random_state = 123)
```

Note that the `n_clusters` argument of `KMeans()` is not used since the function is iterating through a specified range of hyperparameter values.
The `parameter` argument is passed as a string and the `parameter_range` argument should be an iterable containing values of the hyperparameter over which the gap statistic should be calculated.
The function defaults to calculating the gap statistic in terms of changes in inertia but can also be modified to calculate the change in Calinski-Harabasz score, silhouette score, or C-index by changing the `metric` argument.

[Tibshirani and colleagues](https://statweb.stanford.edu/~gwalther/gap) propose taking the first value of k clusters at which the value of the gap statistic at k clusters is greater than the value for k+1 clusters minus the standard error at k+1 clusters.

The function returns a data frame of size `(len(parameter_range), 2)` which contains the gap statistic for each hyperparameter value and the gap statistic's standard error.

|   n_clusters |   gap_statistic |   standard_error |
|-------------:|----------------:|-----------------:|
|            2 |         0.01415 |          0.00134 |
|            3 |         0.04581 |          0.0019  |
|            4 |         0.06136 |          0.0018  |
|            5 |         0.06562 |          0.00212 |
|            6 |         0.07478 |          0.00244 |
|            7 |         0.118   |          0.0031  |
|            8 |         0.11725 |          0.00321 |
|            9 |         0.1476  |          0.00314 |

We can also use the `consensus_cluster()` function to run [Monti consensus clustering](https://link.springer.com/content/pdf/10.1023/A:1023949509487.pdf) over a hyperparameter value range. The function is passed in a similar way to the `gap_statistic()` function.

``` python
consensus_data = cs.KMeans().consensus_cluster(X = data, parameter = 'n_clusters', parameter_range = range(2,10), plot = True, random_state = 123)
```

![Example CDF plot](https://github.com/mgreenig/ClusterSupport/raw/master/docs/artwork/example_cdf_plot.png)

Consensus clustering does not rely on any particular clustering metric. The `plot` argument defaults to `True` and causes the function to output the empirical CDFs for consensus values for different hyperparameter values. 
[Monti et al](https://link.springer.com/content/pdf/10.1023/A:1023949509487.pdf) suggesting picking the number of clusters k at which the largest increase is seen in the area under the CDF between k clusters and k-1 clusters. 
[Șenbabaoğlu et al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4145288/) suggest a different method, involving selecting the number of clusters k at which the proportion of unambiguous consensus values (values <0.1 or >0.9) is greatest.  
The `consensus_cluster()` function returns a Pandas DataFrame of size `(len(parameter_range), 2)`, containing columns for the proportion of unambiguous clusterings and the area under the CDF for every value of the hyperparameter of interest.

|   n_clusters |   proportion_unambiguous_clusterings |   area_under_cdf |
|-------------:|-------------------------------------:|-----------------:|
|            2 |                                1     |            0.396 |
|            3 |                                0.999 |            0.431 |
|            4 |                                0.975 |            0.64  |
|            5 |                                0.827 |            0.704 |
|            6 |                                0.87  |            0.766 |
|            7 |                                0.918 |            0.81  |
|            8 |                                0.902 |            0.827 |
|            9 |                                0.896 |            0.845 |

