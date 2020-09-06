import numpy as np
import pandas as pd
import unittest
from sklearn.datasets import load_iris, load_boston
from clustersupport import KMeans, AgglomerativeClustering
import matplotlib.axes


class self(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # mock numpy array data set
        cls.boston = load_boston()['data']
        cls.scaled_boston = (cls.boston - np.mean(cls.boston, axis=0)) / np.std(cls.boston, axis=0)

        # mock pandas dataframe data set
        cls.iris = pd.DataFrame(load_iris()['data'], columns=load_iris()['feature_names'])
        cls.scaled_iris = (cls.iris - np.mean(cls.iris, axis=0)) / np.std(cls.iris, axis=0)

    # testing gap statistic
    def test_gap_statistic(self):

        boston_gap_stat = KMeans().gap_statistic(X = self.boston, parameter= 'n_clusters', parameter_range=range(2,10))
        iris_gap_stat = AgglomerativeClustering().gap_statistic(X = self.iris, parameter= 'n_clusters', parameter_range=range(2,10), metric = 'CH_score')

        # check if the output df is the correct shape
        self.assertEqual(boston_gap_stat.shape, (len(range(2,10)), 2))
        self.assertEqual(iris_gap_stat.shape, (len(range(2,10)), 2))

        # check if data frame is positive
        self.assertTrue(np.all(boston_gap_stat['standard_error'] > 0))
        self.assertTrue(np.all(iris_gap_stat['standard_error'] > 0))

    # testing
    def test_elbow_plot(self):

        boston_elbow_plot = AgglomerativeClustering().elbow_plot(X = self.boston, parameter= 'n_clusters', parameter_range=range(2,10), metric = 'C_index')
        iris_elbow_plot = KMeans().elbow_plot(X = self.boston, parameter= 'n_clusters', parameter_range=range(2,10))

        # test that the distance around mean calculations work
        self.assertTrue(issubclass(type(boston_elbow_plot), matplotlib.axes.SubplotBase))
        self.assertTrue(issubclass(type(iris_elbow_plot), matplotlib.axes._subplots.SubplotBase))

    # testing inertia calculations with the ones done by sklearn
    def test_consensus_cluster(self):

        boston_consensus_df = KMeans().consensus_cluster(X=self.boston, parameter='n_clusters', parameter_range=range(2, 10))
        iris_consensus_df = AgglomerativeClustering().consensus_cluster(X=self.boston, parameter='n_clusters', parameter_range=range(2, 10))

        # check if shape is correct
        self.assertEqual(boston_consensus_df.shape, (len(range(2,10)), 2))
        self.assertEqual(iris_consensus_df.shape, (len(range(2, 10)), 2))

        # check if all values are in correct range
        self.assertTrue(np.all(boston_consensus_df['proportion_unambiguous_clusterings'] >= 0) and np.all(boston_consensus_df['proportion_unambiguous_clusterings'] <= 1))
        self.assertTrue(np.all(iris_consensus_df['proportion_unambiguous_clusterings'] >= 0) and np.all(iris_consensus_df['proportion_unambiguous_clusterings'] <= 1))
        self.assertTrue(np.all(boston_consensus_df['area_under_cdf'] > 0))
        self.assertTrue(np.all(iris_consensus_df['area_under_cdf'] > 0))

if __name__ == '__main__':
    unittest.main()