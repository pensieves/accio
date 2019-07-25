# An Abstract Base Class for fast / approximate nearest neighbour wrapper.
# Don't instantiate, but subclass.

from abc import ABCMeta, abstractmethod
from sklearn.utils.extmath import softmax
import numpy as np

def id_to_class_func(ids, num_classes, id_class_array):
    r'''Takes id class list with class of all point ids saved at their 
    corresponding index positions and returns the ids mapped to their 
    classes.
    '''
    
    return num_classes, id_class_array[ids]

class NNBase(object, metaclass = ABCMeta):
    r'''Base wrapper class for fast / approximate nearest neighbours.'''
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        r'''A list and dictionary of parameters.'''
        pass
    
    @abstractmethod
    def add_batch(self, vectors, **kwargs):
        r'''Add vectors in the given batch to the NN object.'''
        pass
    
    @abstractmethod
    def build(self, **kwargs):
        r'''build the NN object (probably if it has C / C++ backend)'''
        pass
    
    @abstractmethod
    def save(self, save_path, **kwargs):
        r'''Method to save the NN to save path provided.'''
        pass
    
    @abstractmethod
    def load(self, load_path, **kwargs):
        r'''Method to load the NN to save path provided.'''
        pass
    
    @abstractmethod
    def get_knn(self, queries, k=1, query_as_vector=True, **kwargs):
        r'''Method to return nearest neighbours for a given query which  
        can either be an iterable of ids already indexed or an iterable 
        of query vectors. If the query is for already indexed ids then 
        set query_as_vector as False.
        '''
        pass
    
    def knn_classify(self, queries, k=1, query_as_vector=True, 
                     smoothening=0.001, omit_first_match=False,
                     avg_over_queries=False, 
                     id_to_class_func=id_to_class_func, **fkwargs):
        r'''Method to return classification probabilities based on knn
        and the provided classes of the points.
        omit_first_match should be set as True if the queries are of 
        items from the indexed dataset and hence needs to be omitted to
        get a soft probabilities based on it's nearest neighbour 
        excluding itself.
        '''
        
        if query_as_vector is True:
            queries = np.atleast_2d(queries)
            
        ids, distances = self.get_knn(queries, k, query_as_vector, 
                                      include_distances=True)
        
        if omit_first_match is True:
            ids, distances = ids[:,1:], distances[:,1:]
        
        num_classes, id_classes = id_to_class_func(np.ravel(ids), **fkwargs)
        id_classes = id_classes.reshape(ids.shape)
        
        probabilities_wrt_ids = softmax(-distances)
        proba = np.full((len(queries), num_classes), smoothening)
        
        # obtain class wise boolean in id_classes to sum up the probabilities
        class_bool = (id_classes == np.arange(num_classes)[:,None,None])
        instance_class_proba = class_bool*probabilities_wrt_ids
        
        #sum up the class-wise probabilities and add then up to proba
        proba += instance_class_proba.sum(axis=2).transpose()
        
        # obtain final smoothened probabilities
        proba /= proba.sum(axis=1, keepdims=True)
        
        if avg_over_queries is True:
            proba = proba.mean(axis=0)

        return proba