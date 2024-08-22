import torch
import datasets as cdata
from multiprocessing import Pool
num_threads = 32
num_la = 9
dataset = 'airfrans'
if dataset == 'shape':
    feat_list = ['mesh_pos','saf','dsdf']
    root = '../data/shapes'
    fvclass = cdata.ShapeFV
    normclass = cdata.Shape
else:
    feat_list = ['mesh_pos','saf','dsdf']
    root = '../data/airfrans'
    fvclass = cdata.airfransFV
    normclass = cdata.airfrans

def func(id):
    mdata = fvclass(root,
                                   instance_id=id,
                                   layer_num=num_la,
                                   stride=1,
                                   mode = 'test',
                                   recal_mesh=1,
                                   consist_mesh=0,
                                   in_normal_feature_list=feat_list)
# samples/airfoils
with Pool(processes=num_threads) as pool:
    pool.map(func, range(200))

def func2(id):
    mdata = normclass(root,
                                   instance_id=id,
                                   layer_num=num_la,
                                   stride=1,
                                   mode = 'test',
                                   recal_mesh=1,
                                   consist_mesh=0,
                                   in_normal_feature_list=feat_list)
# samples/airfoils
with Pool(processes=num_threads) as pool:
    pool.map(func2, range(200))

