import sklearn.cluster

from clustersupport.clustering_results import Decorators
from clustersupport.optimisation_methods import OptimisationMethods
from clustersupport.feature_methods import FeatureMethods

# overwrite class methods with the decorator
class KMeans(sklearn.cluster.KMeans, OptimisationMethods, FeatureMethods):
    valid_parameters = {'n_clusters'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.KMeans.fit)
    
class AffinityPropagation(sklearn.cluster.AffinityPropagation, OptimisationMethods, FeatureMethods):
    valid_parameters = {'damping'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.AffinityPropagation.fit)
        
class MiniBatchKMeans(sklearn.cluster.MiniBatchKMeans, OptimisationMethods, FeatureMethods):
    valid_parameters = {'n_clusters'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.MiniBatchKMeans.fit)
    
class AgglomerativeClustering(sklearn.cluster.AgglomerativeClustering, OptimisationMethods, FeatureMethods):
    valid_parameters = {'n_clusters', 'affinity'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.AgglomerativeClustering.fit)
    
class OPTICS(sklearn.cluster.OPTICS, OptimisationMethods, FeatureMethods):
    valid_parameters = {'min_samples'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.OPTICS.fit)
    
class DBSCAN(sklearn.cluster.DBSCAN, OptimisationMethods, FeatureMethods):
    valid_parameters = {'eps', 'min_samples'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.DBSCAN.fit)
    
class MeanShift(sklearn.cluster.MeanShift, OptimisationMethods, FeatureMethods):
    valid_parameters = {'bandwidth'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.MeanShift.fit)
    
class Birch(sklearn.cluster.Birch, OptimisationMethods, FeatureMethods):
    valid_parameters = {'threshold', 'branching_factor', 'n_clusters'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.Birch.fit)

class SpectralClustering(sklearn.cluster.SpectralClustering, OptimisationMethods, FeatureMethods):
    valid_parameters = {'n_clusters'}
    fit = Decorators.cluster_wrapper(sklearn.cluster.SpectralClustering.fit)
