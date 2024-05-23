from process_vtk import save_vtpvtu,save_node_markers,save_edge_indices,save_coarse_data,save_SDFSAFDSDF_data
import json,os
from multiprocessing import Pool
# from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fine", action='store_true') # add -f to generate node based fine mesh post-processed data
parser.add_argument('-d','--device',type = str, default = 'cuda:0')
parser.add_argument('-T','--num_threads', type = int, default=8)
args = parser.parse_args()

fine = args.fine
if fine:
    raw_prefix =  os.path.join('/home1/OF_dataset/')
    assert os.path.exists(raw_prefix), "Raw Airfrans Openfoam cases must be saved in a directory OF_dataset (currently missing)"
    savepath = 'Fine_vtk22/'
else:
    raw_prefix = os.path.join(os.getcwd(),'AIRFRANS_coarse/')
    assert os.path.exists(raw_prefix), "Coarse version of Airfrans openfoam cases must be saved in a directory AIRFRANS_coarse (currently missing)"
    savepath = 'Coarse_vtk22/'
def runcase(c):
    if args.fine:
        save_vtpvtu(casepath=c,savepath=savepath)
        save_node_markers(c,savepath)
        save_edge_indices(c,savepath)
        save_SDFSAFDSDF_data(c,savepath,device = args.device)
    else:
        save_vtpvtu(casepath=c,savepath=savepath)
        save_SDFSAFDSDF_data(c,savepath,device = args.device)
        save_coarse_data(c,savepath)
    
if __name__=='__main__':
    with open('manifest.json', 'r') as f:
        manifest = json.load(f)
    input_list = []
    for name in (manifest['full_test']+manifest['full_train']):
        input_list.append(os.path.join(raw_prefix,name))
    if args.fine is False:
        for i in input_list[:100]:
            runcase(i)
        #with Pool(processes=args.num_threads) as pool:
        #    pool.map(runcase, input_list)
    else:
        for c in tqdm(input_list[:100], 'Post-processing fine AirfRANS'):
            runcase(c)
