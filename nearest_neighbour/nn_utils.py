import numpy as np
from functools import reduce
from pathlib import Path
from .. import nearest_neighbour as nn

def combine_knns(knn1, knn1_src_id, knn2, knn2_src_id, k=-1):
    r'''Combines results from two KNNs w.r.t the distance values'''
    
    ids1, distances1 = knn1
    ids2, distances2 = knn2
    
    if k == -1:
        k = ids1.shape[1] + ids2.shape[1]
    
    if isinstance(knn1_src_id, int):
        knn1_src_id = np.full(ids1.shape, knn1_src_id)
    
    if isinstance(knn2_src_id, int):
        knn2_src_id = np.full(ids2.shape, knn2_src_id)
    
    ids = np.concatenate((ids1, ids2), axis=1)
    src_ids = np.concatenate((knn1_src_id, knn2_src_id), axis=1)
    distances = np.concatenate((distances1, distances2), axis=1)
    
    # argsort the distances and use the returned indices to obtain
    # ids, src_ids and distances for combined knns
    dist_argsort = np.argsort(distances, axis=1)
    
    # sort the distances, src_ids and ids.
    # ravel() provides speedup over flatten() or reshape(-1) or
    # the direct indexing as:
    # a[np.arange(a.shape[0])[:,np.newaxis], np.argsort(a, axis=1)]
    
    distances = distances.ravel()[dist_argsort+\
                                (np.arange(distances.shape[0])\
                                [:,np.newaxis]*distances.shape[1])]
    src_ids = src_ids.ravel()[dist_argsort+\
                                (np.arange(src_ids.shape[0])\
                                [:,np.newaxis]*src_ids.shape[1])]
    ids = ids.ravel()[dist_argsort+(np.arange(ids.shape[0])\
                                [:,np.newaxis]*ids.shape[1])]
    
    return (ids[:,:k], distances[:,:k]), src_ids[:,:k]

def get_combined_knn(queries, nn_list, k=-1, nn_algo='Annoy'):
    r'''Combines results from multiple KNNs w.r.t the distance
    values. If nn_list is a list of nn objects then nn_algo will
    be ignored. If nn_list is a list of saved nn object paths 
    then nn_algo needs to be correctly provided else the default 
    value of Annoy will be considered.'''
    
    if isinstance(nn_list[0], str) or isinstance(nn_list[0], Path):
        
        return reduce(lambda si, sj: combine_knns(*si,*sj,k=k), 
                          ((nn.__dict__[nn_algo]().load(nn_list[i]).\
                            get_knn(queries, k=k, query_as_vector=True,\
                                    include_distances=True), i) for i \
                           in range(len(nn_list))))
    
    else:
        
        return reduce(lambda si, sj: combine_knns(*si,*sj,k=k), 
                          ((nn_list[i].get_knn(queries, k=k, \
                                               query_as_vector=True,\
                                               include_distances=True), i)\
                           for i in range(len(nn_list))))