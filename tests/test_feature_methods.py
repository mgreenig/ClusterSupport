import numpy as np
import pandas as pd
import unittest
from sklearn.datasets import load_iris, load_boston
from clustersupport import KMeans, AgglomerativeClustering

class self(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # mock numpy array data set
        cls.boston = load_boston()['data']
        cls.scaled_boston = (cls.boston - np.mean(cls.boston, axis=0)) / np.std(cls.boston, axis=0)
        cls.boston_clustering = KMeans(n_clusters=3)
        cls.boston_clustering.fit(cls.boston)

        # mock pandas dataframe data set
        cls.iris = pd.DataFrame(load_iris()['data'], columns=load_iris()['feature_names'])
        cls.scaled_iris = (cls.iris - np.mean(cls.iris, axis=0)) / np.std(cls.iris, axis=0)
        # do not call fit on this one to test if functions execute properly
        cls.iris_clustering = AgglomerativeClustering(n_clusters=4)

    # testing gap statistic
    def test_t_test(self):

        boston_t_test = self.boston_clustering.t_test(X = self.boston, output = 'p-value')
        iris_t_test = self.iris_clustering.t_test(X = self.iris, output = 't-statistic')

        # check if the output df is the correct shape (n_clusters, n_features)
        self.assertEqual(boston_t_test.shape, (3, self.boston.shape[1]))
        self.assertEqual(iris_t_test.shape, (4, self.iris.shape[1]))

        # check if all p-values are between 0 and 1
        self.assertTrue(np.all(boston_t_test > 0) and np.all(boston_t_test < 1))
        # check if t-statistic computes differently than p-values
        self.assertTrue(np.any(iris_t_test < 0) or np.any(iris_t_test > 1))

    # testing leave one out function
    def test_leave_one_out(self):

        boston_LOO = self.boston_clustering.leave_one_out(X=self.boston, metric='CH_score')
        iris_LOO = self.iris_clustering.leave_one_out(X=self.iris, metric='inertia')

        # check if the output df is the correct shape (n_clusters, n_features)
        self.assertEqual(len(boston_LOO), self.boston.shape[1])
        self.assertEqual(len(iris_LOO), self.iris.shape[1])

        # check if the output df is the correct shape (n_clusters, n_features)
        self.assertTrue(isinstance(boston_LOO, pd.Series))
        self.assertTrue(isinstance(iris_LOO, pd.Series))

    # testing logistic regression calculations
    def test_logistic_regression(self):

        boston_LR = self.boston_clustering.logistic_regression(X=self.boston, output='coef')
        iris_LR = self.iris_clustering.logistic_regression(X=self.iris, output='p-value', n_bootstraps=20)

        # check if the output df is the correct shape (n_clusters, n_features)
        self.assertEqual(boston_LR.shape, (3, self.boston.shape[1]))
        self.assertEqual(iris_LR.shape, (4, self.iris.shape[1]))

        # check if all p-values are between 0 and 1
        self.assertTrue(np.all(iris_LR > 0) and np.all(iris_LR < 1))
        # check if coefficients compute differently than p-values
        self.assertTrue(np.any(boston_LR < 0) or np.any(boston_LR > 1))

    # testing logistic regression calculations
    def test_mann_whitney(self):

        boston_MW = self.boston_clustering.mann_whitney(X=self.boston, output = 'p-value')
        iris_MW = self.iris_clustering.mann_whitney(X=self.iris, output = 'z-score')

        # check if the output df is the correct shape (n_clusters, n_features)
        self.assertEqual(boston_MW.shape, (3, self.boston.shape[1]))
        self.assertEqual(iris_MW.shape, (4, self.iris.shape[1]))

        # check if all p-values are between 0 and 1
        self.assertTrue(np.all(boston_MW > 0) and np.all(boston_MW < 1))
        # check if coefficients compute differently than p-values
        self.assertTrue(np.any(iris_MW < 0) or np.any(iris_MW > 1))

if __name__ == '__main__':
    unittest.main()