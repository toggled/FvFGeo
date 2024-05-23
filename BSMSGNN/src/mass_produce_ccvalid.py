import torch
import datasets as cdata
from multiprocessing import Pool
num_threads = 8
num_la = 5
def func2(id):
    mdata = cdata.ShapeFV('../data/shapes',
                                   instance_id=id,
                                   layer_num=num_la,
                                   stride=1,
                                   mode = 'valid',
                                   recal_mesh=1,
                                   consist_mesh=0,
                                   in_normal_feature_list=['mesh_pos','saf','dsdf'])
# samples/airfoils
with Pool(processes=num_threads) as pool:
    pool.map(func2, range(200))
def func22(id):
    mdata = cdata.Shape('../data/shapes',
                                   instance_id=id,
                                   layer_num=num_la,
                                   stride=1,
                                   mode = 'valid',
                                   recal_mesh=1,
                                   consist_mesh=0,
                                   in_normal_feature_list=['mesh_pos','saf','dsdf'])
# samples/airfoils
with Pool(processes=num_threads) as pool:
    pool.map(func22, range(200))
