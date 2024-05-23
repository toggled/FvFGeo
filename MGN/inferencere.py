import argparse, yaml, os, json, glob
import torch
import train, metrics
from dataset import Dataset,FVDataset,GUNetDataset,FVDatasetreplaceSAF,NSResDataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-M', '--model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet', type = str)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-m', '--modelpath', help='Path to the model we test', default = 'GUNetFVGCN_SAF_dSDF', type = str)
parser.add_argument('-saf','--saf',default = 1, type = int, help = 'IF saf was used for training')
parser.add_argument('-dsdf','--dsdf', default = 1, type=int, help = 'If dsdf was used for training')
parser.add_argument('-rm', '--remeshing', help = 'If you want to train in dp. (default: 0)', default = 0, type = int)
args = parser.parse_args()

use_fv = args.model.startswith('FVGraphSAGE')
gunet_ssg = args.model.startswith('GUNetSGraphSAGE')  or args.model.startswith('GUNetGCN_SAF_dSDF')
use_fv_subsam = args.model.startswith('GNetFVGraphSAGE')
fvnew = args.model.startswith('GNetFVnewGraphSAGE') or args.model.startswith('GNetFVnewGCN_FVnew_SAF_dSDF')
fvnew_res = args.model.startswith('GNetFVnewGraphSAGE_residual')
if __name__=='__main__':
#    with open('/home/jessica/Downloads/submission/AirfRANS/Dataset/manifest.json', 'r') as f:
#        manifest = json.load(f)
#    with open('/home/jessica/Downloads/submission/AirfRANS/Dataset/manifest_dsdf.json', 'r') as f:
#        manifest_dsdf = json.load(f)
#    with open('/home/jessica/Downloads/submission/AirfRANS/Dataset/manifest_saf.json', 'r') as f:
#        manifest_saf = json.load(f)
    with open('Dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    if use_fv or use_fv_subsam:
        with open('Dataset/Fvattrs_surf_bi/FVattrs.json', 'r') as f:
            manifest_fv = json.load(f)
    elif fvnew or fvnew_res:
        pass
    else:
        if gunet_ssg: #GUNetSGraphSAGE
            with open('Dataset/manifest_saf2.json', 'r') as f:
                manifest_saf = json.load(f) # Load updated SAF (not the old one)
        else:
            with open('Dataset/manifest_saf.json', 'r') as f:
                manifest_saf = json.load(f)
        with open('Dataset/manifest_dsdf.json', 'r') as f:
                manifest_dsdf = json.load(f)

    # n_data = 200 #for 1/4 data
    # n_data = 400 #for half data
    # n_data = 600 #for 3/4 data
    n_data = 800 #for FULL data
    manifest_train = manifest[args.task + '_train']#[:n_data]
    # test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
    n = int(.1*len(manifest_train))
    train_dataset = manifest_train[:-n] 
    if use_fv:
        _, coef_norm = FVDataset(train_dataset, norm = True, manifest_fv = manifest_fv)

    elif use_fv_subsam:
        if args.remeshing: # Fv + subsampling + remeshing
            train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None)
        else: # FV + no subsampling or remeshing
            train_dataset, coef_norm = FVDatasetreplaceSAF(train_dataset, norm = True, manifest_fv = manifest_fv)
    elif fvnew or fvnew_res:
        train_dataset, coef_norm = NSResDataset(train_dataset, norm = True)
    else:
        if gunet_ssg: #  python inference.py -t scarce -M GUNet -m GUNet // python inference.py -t scarce -M GUNetSGraphSAGE_SAF_dSDF -m GUNetSGraphSAGE_SAF_dSDF -saf 1 -dsdf 1
                # _, coef_norm = GUNetDataset(train_dataset, norm = True, \
                #     manifest_dsdf = manifest_dsdf,manifest_saf = manifest_saf)
                # print(coef_norm)
                coef_norm = (np.array([ 4.99704629e-01,  1.16213085e-02,  6.07769814e+01,  5.06739235e+00,
                -1.68674637e-03, -8.38373380e-05,  3.68214585e-02,  4.45225369e-03,
                    2.48446131e+00,  2.53843737e+00,  2.79392791e+00,  3.00866175e+00,
                    3.10984993e+00,  3.06120133e+00,  2.81851792e+00,  2.61446238e+00],
                dtype=np.float32), np.array([ 0.70514375,  0.33673635, 17.171833  ,  6.146387  ,  0.04210775,
                    0.06199972,  0.4150414 ,  0.3262758 ,  1.2856725 ,  1.5891755 ,
                    1.5941806 ,  1.475755  ,  0.91610646,  1.4538018 ,  1.5798049 ,
                    1.5741476 ], dtype=np.float32), np.array([ 4.1501225e+01,  1.0193408e+01, -4.2369653e+02,  8.1722403e-04],
                dtype=np.float32), np.array([2.9134247e+01, 3.0195290e+01, 2.5707593e+03, 2.9320728e-03],
                dtype=np.float32))
        else:
            _, coef_norm = Dataset(train_dataset, norm = True, sample = None, \
                use_saf =False,use_dsdf = False, manifest_dsdf = None,manifest_saf = None)
    # Cuda
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #with open('params.yaml', 'r') as f: # hyperparameters of the model
    with open('paramsNEW.yaml', 'r') as f: #EDITED####
        hparams = yaml.safe_load(f)[args.model]
    model = torch.load(args.modelpath, map_location = device)

    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    if use_fv:
        import metricsfv
        coefs = metricsfv.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
             criterion = 'MSE', s = s, manifest_fv = manifest_fv)
    elif use_fv_subsam:
        if args.remeshing:
            raise ValueError("Not implemented (TODO)")
        else:
            import metricsfv
            coefs = metricsfv.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
             criterion = 'MSE', s = s, manifest_fv = manifest_fv,use_fv_subsamp=True)
    elif fvnew or fvnew_res:
        import metricsfvnew
        if fvnew and (not fvnew_res):
            coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
             criterion = 'MSE', s = s)
        else:
            #coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
            # criterion = 'MSE', s = s, res_est = torch.load('test_ests.pickle'))
            #coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
            # criterion = 'MSE', s = s, res_est = torch.load('test_ests_pos.pickle'))
            #coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
            # criterion = 'MSE', s = s, res_est = torch.load('test_ests1.pickle'))
            coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
             criterion = 'MSE', s = s, res_est = torch.load('test_ests_fvnew.pickle'))
            #coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
            # criterion = 'MSE', s = s, res_est = torch.load('test_ests1_uvp.pickle'))
            #coefs = metricsfvnew.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
            # criterion = 'MSE', s = s, res_est = torch.load('test_ests_ucm.pickle'))
    else:
        if gunet_ssg:
            # import metricsfv
            import metricsre

            coefs = metricsre.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/',\
                 criterion = 'MSE', s = s, use_saf = args.saf,use_dsdf = args.dsdf, \
                    manifest_dsdf = manifest_dsdf, manifest_saf = manifest_saf,gunet_ssg=True)
        else:
            coefs = metricsre.Results_test(device, [model], hparams, coef_norm, n_test = 1, path_in = 'Dataset/', \
                criterion = 'MSE', s = s,use_saf = args.saf,use_dsdf = args.dsdf, \
                    manifest_dsdf = manifest_dsdf, manifest_saf = manifest_saf)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
    icml_scores = 'icmlscores5/'
    os.system('mkdir -p '+icml_scores)
    icml_scores = icml_scores+args.modelpath+"/"
    os.system('mkdir -p '+icml_scores)
    os.system('mkdir -p '+icml_scores + args.task)
    np.save(icml_scores + args.task + '/true_coefs', coefs[0])
    np.save(icml_scores + args.task + '/pred_coefs_mean', coefs[1])
    np.save(icml_scores + args.task + '/pred_coefs_std', coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(icml_scores + args.task + '/true_surf_coefs_' + str(n), file)
    for n, file in enumerate(coefs[4]):
        np.save(icml_scores + args.task + '/surf_coefs_' + str(n), file)
    np.save(icml_scores + args.task + '/true_bls', coefs[5])
    np.save(icml_scores + args.task + '/bls', coefs[6])
    if not use_fv:
        for aero in glob.glob('airFoil2D*'):
            os.rename(aero, icml_scores + args.task + '/' + aero)
    os.rename('score.json', icml_scores + args.task + '/score.json')
    os.system('mv Cl_model#* '+icml_scores+args.task+'/')
    os.system('mv Cd_model#* '+icml_scores+args.task+'/')


# python inference.py -t scarce -M GUNetFVGCN -m GUNetFVGCN_SAF_dSDF -saf 1 -dsdf 1
# python inference.py -t scarce -M GUNetFVGCN -m GUNetFVGCN_SAF -saf 1 -dsdf 0
