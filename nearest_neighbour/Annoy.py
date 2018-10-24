from annoy import AnnoyIndex
from .NNBase import NNBase
from copy import deepcopy
from pathlib import Path
import numpy as np
from sklearn.externals import joblib

class Annoy(NNBase):
    r'''Approximate Nearest Neighbour wrapper around Annoy library
    at https://github.com/spotify/annoy'''

    def __init__(self, *args, **kwargs):
        r'''should satisfy the function signature of AnnoyIndex i.e. 
        (f, metric='angular') where f is dimension of the features 
        and metric can be "angular", "euclidean", "manhattan", 
        "hamming", or "dot".'''

        self.args = deepcopy(args)
        self.kwargs = deepcopy(kwargs)
        if args:
            self.nn = AnnoyIndex(*args, **kwargs)
        else:
            self.nn = None
        self.n_items = 0

    def add_batch(self, vectors, **kwargs):
        r'''add vectors to ANN'''

        for vector in vectors:
            self.nn.add_item(self.n_items, vector)
            self.n_items += 1
        
        return self

    def build(self, **kwargs):
        r'''Build the Annoy Index'''

        self.nn.build(**kwargs)
        return self

    def save(self, save_path, **kwargs):
        r'''Save the AnnoyIndex in the provided path'''

        self.nn.save(save_path)
        save_path = Path(save_path)
        helper_path = save_path.parent/(save_path.stem+'_helper'+save_path.suffix)
        helper = {'init_args' : self.args, 'init_kwargs' : self.kwargs}
        joblib.dump(helper, helper_path)

    def load(self, load_path, **kwargs):
        r'''Load the AnnoyIndex from the provided path. The provided 
        path should be of the AnnoyIndex and not the helper object.'''

        load_path = Path(load_path)
        helper_path = load_path.parent/(load_path.stem+'_helper'+load_path.suffix)
        helper = joblib.load(helper_path)

        self.args = helper['init_args']
        self.kwargs = helper['init_kwargs']
        self.nn = AnnoyIndex(*self.args, **self.kwargs)
        self.nn.load(str(load_path))
        self.n_items = self.nn.get_n_items()
        
        return self

    def get_knn(self, queries, k=-1, query_as_vector=True, **kwargs):
        r'''Method to return nearest neighbours for a given query which 
        can either be an iterable of ids already indexed or an iterable 
        of query vectors. If the query is for already indexed ids then 
        set query_as_vector as False.
        '''

        if k == -1:
            k = self.nn.get_n_items()

        if query_as_vector:
            queries = np.atleast_2d(queries)
            
        ids = np.empty((len(queries),k), dtype='int32')
        distances = None
        if kwargs.get('include_distances'):
            distances = np.empty((len(queries),k))

        fetch_knn = self.nn.get_nns_by_vector if query_as_vector \
                        else self.nn.get_nns_by_item
        
        if kwargs.get('include_distances'):
            for i, query in enumerate(queries):
                result = fetch_knn(query, k, **kwargs)
                ids[i], distances[i] = result[0], result[1]
        else:
            for i, query in enumerate(queries):
                result = fetch_knn(query, k, **kwargs)
                ids[i] = result

        return ids, distances