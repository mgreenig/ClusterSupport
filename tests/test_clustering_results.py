import numpy as np
import pandas as pd
import unittest
from sklearn.datasets import load_iris, load_boston
from sklearn.cluster import KMeans
from clustersupport.clustering_results import ClusteringResult

class self(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        # mock numpy array data set
        cls.boston = load_boston()['data']
        cls.scaled_boston = (cls.boston - np.mean(cls.boston, axis=0)) / np.std(cls.boston, axis=0)
        cls.boston_clustering = KMeans(n_clusters=3).fit(cls.boston)
        cls.boston_labels = cls.boston_clustering.labels_

        # mock pandas dataframe data set
        cls.iris = pd.DataFrame(load_iris()['data'], columns=load_iris()['feature_names'])
        cls.scaled_iris = (cls.iris - np.mean(cls.iris, axis=0)) / np.std(cls.iris, axis=0)
        # fit this one with scaled data
        cls.iris_clustering = KMeans(n_clusters=4).fit(cls.scaled_iris)
        cls.iris_labels = cls.iris_clustering.labels_

        # initialise class instances
        cls.boston_instance = ClusteringResult(X=cls.boston, labels=cls.boston_labels)
        cls.iris_instance = ClusteringResult(X=cls.scaled_iris, labels=cls.iris_labels)

    # testing class initialisation
    def test_init(self):

        # check if the array data we transform into has the same shape as the raw data
        self.assertEqual(self.boston_instance.array_X.shape, self.boston.shape)
        self.assertEqual(self.iris_instance.array_X.shape, self.iris.shape)

        # check if the scaling works
        self.assertTrue(np.all(self.boston_instance.scaled_X == self.scaled_boston))
        self.assertTrue(np.allclose(self.iris_instance.scaled_X, self.scaled_iris))

        # check if number of clusters is correct
        self.assertEqual(self.boston_instance.n_clusters, 3)
        self.assertEqual(self.iris_instance.n_clusters, 4)

    # testing sum of squares calculations
    def test_distance_around_mean(self):

        # ground truth for sum of squares for cluster 1 in boston data set
        boston_c1_mask = self.boston_labels == 1
        boston_c1 = self.boston[self.boston_labels == 1]
        boston_c1_ss = np.sum(np.linalg.norm(boston_c1 - np.mean(boston_c1, axis = 0), axis = 1) ** 2)

        # ground truth for sum of squares for cluster 3 in iris data set
        iris_c3_mask = self.iris_labels == 3
        iris_c3 = self.scaled_iris[iris_c3_mask]
        iris_c3_ss = np.sum(np.linalg.norm(iris_c3 - np.mean(iris_c3, axis = 0), axis = 1) ** 2)

        # test that the distance around mean calculations work
        self.assertEqual(boston_c1_ss, self.boston_instance.distance_around_mean(cluster_mask = boston_c1_mask, scale = False))
        self.assertEqual(iris_c3_ss, self.iris_instance.distance_around_mean(cluster_mask = iris_c3_mask, scale = False))

        # check that a non-valid distance metric raises a value error
        self.assertRaises(ValueError, lambda: self.boston_instance.distance_around_mean(cluster_mask = boston_c1_mask, dist_metric = 'jaccard'))
        self.assertRaises(ValueError, lambda: self.iris_instance.distance_around_mean(cluster_mask = iris_c3_mask, dist_metric = 'jaccard'))

    # testing inertia calculations with the ones done by sklearn
    def test_inertia(self):

        self.assertAlmostEqual(self.boston_instance.inertia(scale = False), self.boston_clustering.inertia_)
        self.assertAlmostEqual(self.iris_instance.inertia(scale = False), self.iris_clustering.inertia_)

    # testing C index function to see if answer lies between 0 and 1
    def test_C_index(self):

        self.assertTrue(0 < self.boston_instance.C_index() < 1)
        self.assertTrue(0 < self.iris_instance.C_index() < 1)

    # testing silhouette score function to see if answer lies between -1 and 1
    def test_silhouette(self):

        self.assertTrue(-1 < self.boston_instance.silhouette_score() < 1)
        self.assertTrue(-1 < self.iris_instance.silhouette_score() < 1)

    # testing summary function to see if the outputted data frame is the correct shape and returns all positive values
    def test_get_summary(self):

        boston_summary = self.boston_instance.get_summary()
        iris_summary = self.iris_instance.get_summary()

        self.assertEqual(boston_summary.shape, (3, 2))
        self.assertTrue(np.all(boston_summary > 0))
        self.assertEqual(iris_summary.shape, (4, 2))
        self.assertTrue(np.all(iris_summary > 0))

    # testing classifier assessment function to see if the outputted data frame is the correct shape and returns all positive values
    def test_classifier_assessment(self):

        clf_assessment_boston = self.boston_instance.classifier_assessment(grid_search = False, roc_plot = False)
        clf_assessment_iris = self.iris_instance.classifier_assessment(roc_plot = False)

        # check the sizes are correct
        self.assertEqual(clf_assessment_boston.shape, (3, 4))
        self.assertEqual(clf_assessment_iris.shape, (4, 4))

        # check the values of the assessment data frame
        self.assertTrue(np.all(clf_assessment_boston > 0))
        self.assertTrue(np.all(clf_assessment_iris > 0))


if __name__ == '__main__':
    unittest.main()