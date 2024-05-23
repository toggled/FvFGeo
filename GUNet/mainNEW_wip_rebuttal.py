#@title Main script, EDITED to use dSDF data and train GUNet with aSGCN_conv

#edited from: https://github.com/Extrality/AirfRANS/blob/main/main.py

import argparse, yaml, os, json, glob
import torch
#import train, metrics
# import trainNEW, metrics
import trainNEW_correct as trainNEW
import metrics
from dataset import Dataset,FVDataset,GUNetDataset,FVDatasetreplaceSAF
import sys
import numpy as np

sys.path.append('../../submission/AirfRANS/')

parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-s', '--score', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = int)
parser.add_argument('-cuda', '--cuda', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = str)
parser.add_argument('-dp', '--dp', help = 'If you want to train in dp. (default: 0)', default = 0, type = int)
# parser.add_argument('-saf','--saf',default = 1, type = int, help = 'Use saf for training')
# parser.add_argument('-dsdf','--dsdf', default = 1, type=int, help = 'Use dsdf for training')
parser.add_argument('-p','--prec', help = 'The precision you want to train, choose between half,full', default = 'half', type = str)
args = parser.parse_args()
use_fv = args.model.startswith('FVGraphSAGE') 
use_fv_subsam = args.model.startswith('GNetFVGraphSAGE') or args.model.startswith('GNetFVnewGraphSAGE')
gunet_ssg = args.model.startswith('GUNetSGraphSAGE') or args.model.startswith('GUNetGCN_SAF_dSDF')
print('Running: ',args)
task = args.task
if task=='full':
    nb_epochs = 400
elif task == 'scarce':
    nb_epochs = 1600
elif task == 'reynolds':
    nb_epochs = 635
else: # AOA
    nb_epochs = 398 

with open('paramsNEW.yaml', 'r') as f: #EDITED####
    #print(yaml.safe_load(f))
    hparams = yaml.safe_load(f)[args.model]
hparams['nb_epochs'] = nb_epochs
hparams['half_prec'] = (args.prec=='half')

with open('Dataset/manifest.json', 'r') as f:
    manifest = json.load(f)
if use_fv or use_fv_subsam:
    #with open('Dataset/Fvattrs_surf/FVattrs.json', 'r') as f: # does not include surface points as nodes (cell-centers only)
    with open('Dataset/Fvattrs_surf_bi/FVattrs.json', 'r') as f: # include surface points as nodes in the graph
        manifest_fv = json.load(f)
else:
    with open('Dataset/manifest_dsdf.json', 'r') as f:
        manifest_dsdf = json.load(f)
    with open('Dataset/manifest_saf.json', 'r') as f:
        manifest_saf = json.load(f)


manifest_train = manifest[args.task + '_train']#[0:10]
test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(.1*len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]

#n = 2
#train_dataset = manifest_train[:3]
#val_dataset = manifest_train[3:3+n]
if use_fv: # Load our own Data()
    train_dataset, coef_norm = FVDataset(train_dataset, norm = True, manifest_fv = manifest_fv)
    val_dataset = FVDataset(val_dataset, manifest_fv = manifest_fv, coef_norm = coef_norm)
elif use_fv_subsam:
    # train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None)
    # val_dataset = Dataset(val_dataset, sample = None, coef_norm = coef_norm)
    # with open('Dataset/manifest_saf2.json', 'r') as f:
    #         manifest_saf = json.load(f) # Load updated SAF (not the old one)
    train_dataset, coef_norm = FVDatasetreplaceSAF(train_dataset, norm = True, manifest_fv = manifest_fv)
    val_dataset = FVDatasetreplaceSAF(val_dataset, manifest_fv = manifest_fv, coef_norm = coef_norm)
else:
    if gunet_ssg: #GUNetSGraphSAGE
        with open('Dataset/manifest_saf2.json', 'r') as f:
            manifest_saf = json.load(f) # Load updated SAF (not the old one)
        train_dataset, coef_norm = GUNetDataset(train_dataset, norm = True, \
                manifest_dsdf = manifest_dsdf,manifest_saf = manifest_saf)
        val_dataset = GUNetDataset(val_dataset, coef_norm = coef_norm,\
            manifest_dsdf = manifest_dsdf,manifest_saf = manifest_saf)
        # print('example: \n\r',val_dataset[0])
    else: # Original authors
        # if os.path.exists('Dataset/train_dataset'):
        #     train_dataset = torch.load('Dataset/train_dataset')
        #     val_dataset = torch.load('Dataset/val_dataset')
        #     coef_norm = torch.load('Dataset/normalization')
        # else:
        train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None, \
                use_saf = hparams['SAF'],use_dsdf = hparams['dSDF'], manifest_dsdf = manifest_dsdf,manifest_saf = manifest_saf)
        # torch.save(train_dataset, 'Dataset/train_dataset')
        # torch.save(coef_norm, 'Dataset/normalization')
        val_dataset = Dataset(val_dataset, sample = None, coef_norm = coef_norm)
        # torch.save(val_dataset, 'Dataset/val_dataset')

print('train_dataset[0]: ',train_dataset[0])
# import sys 
# sys.exit(1)
# Cuda
use_cuda = torch.cuda.is_available()
device = args.cuda if use_cuda else 'cpu' #EDITED####
if use_cuda:
    print('Using: ',device) #print('Using GPU')
else:
    print('Using CPU')

#with open('params.yaml', 'r') as f: # hyperparameters of the model
# with open('paramsNEW.yaml', 'r') as f: #EDITED####
#     hparams = yaml.safe_load(f)[args.model]
from models.MLP import MLP

models = []
for i in range(args.nmodel):
    if (hparams['encoder'] is not False):
        encoder = MLP(hparams['encoder'], batch_norm = False)
        decoder = MLP(hparams['decoder'], batch_norm = False)
    else:
        encoder, decoder = None, None

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model = GraphSAGE(hparams, encoder, decoder)
    
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model = PointNet(hparams, encoder, decoder)

    elif args.model == 'MLP':
        from models.NN import NN
        model = NN(hparams, encoder, decoder)
        
    elif args.model.startswith('GUNetSGraphSAGE'):
        from models.GUNetSGraphSAGE import GUNetSGraphSAGE
        model = GUNetSGraphSAGE(hparams, encoder, decoder)
    
    elif args.model.startswith('GNetFVGraphSAGE'):
        from models.GNetFVGraphSAGE import GNetFVGraphSAGE
        model = GNetFVGraphSAGE(hparams, encoder, decoder)
    
    elif args.model.startswith('FVGraphSAGE'):
        from models.FVGraphSAGE import FVGraphSAGE
        model = FVGraphSAGE(hparams, encoder, decoder)
    
    elif args.model.startswith('FVGCN'):
        from models.FVGCN import FVGCN
        model = FVGCN(hparams, encoder, decoder)
    
    elif args.model.startswith('GraphSAGEFVGCN'):
        from models.GraphSAGEFVGCN import GraphSAGEFVGCN
        model = GraphSAGEFVGCN(hparams, encoder, decoder)

    elif args.model == 'GUNet':
        from models.GUNet import GUNet
        model = GUNet(hparams, encoder, decoder)
        
    elif args.model.startswith('GUNetGCN'):
        from models.GUNetGCN import GUNetGCN
        model = GUNetGCN(hparams, encoder, decoder)
        
    else: #GUNetFVGCN or GUNetFVGCN_FV_aSGCN_A2_SAF_dSDF etc...
        from models.GUNetFVGCN import GUNetFVGCN
        model = GUNetFVGCN(hparams, encoder, decoder)

    path = 'metrics/'+args.model+"/" # path where you want to save log and figures
    os.system('mkdir -p '+path)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('model_parameters : ',sum([np.prod(p.size()) for p in model_parameters]))
    
    # model = train.main(device, train_dataset, val_dataset, model, hparams, path, 
    #             criterion = 'MSE_weighted', val_iter = None, reg = args.weight, name_mod = args.model, val_sample = True, use_saf = hparams['SAF'], use_dsdf = hparams['dSDF'])
    if use_fv_subsam:
        # import trainNEW_wip
        # model = trainNEW_wip.fvmainsubsam(device, train_dataset, val_dataset, model, hparams, path, 
        #         criterion = 'MSE_weighted', val_iter = 25, reg = args.weight, name_mod = args.model, val_sample = True,dp = (args.dp==1))
        model = trainNEW.main(device, train_dataset, val_dataset, model, hparams, path, 
                criterion = 'MSE_weighted', val_iter = 25, reg = args.weight, name_mod = args.model, val_sample = True, use_saf = hparams['SAF'], use_dsdf = hparams['dSDF'],dp = (args.dp == 1))

    else:
        model = trainNEW.main(device, train_dataset, val_dataset, model, hparams, path, 
                criterion = 'MSE_weighted', val_iter = 25, reg = args.weight, name_mod = args.model, val_sample = True, use_saf = hparams['SAF'], use_dsdf = hparams['dSDF'],dp = (args.dp == 1))
    
    models.append(model)
    torch.save(model,args.model+args.prec+"_"+str(i))
torch.save(models, args.model+args.prec+"_final_clip1") #EDITED####
print(args,' done')

# if bool(args.score):
#     s = args.task + '_test' if args.task != 'scarce' else 'full_test'
#     coefs = metrics.Results_test(device, [models], hparams, coef_norm, n_test = 3, path_in = 'Dataset/', criterion = 'MSE', s = s)
#     # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
#     # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
#     np.save('scores/' + args.task + '/true_coefs', coefs[0])
#     np.save('scores/' + args.task + '/pred_coefs_mean', coefs[1])
#     np.save('scores/' + args.task + '/pred_coefs_std', coefs[2])
#     for n, file in enumerate(coefs[3]):
#         np.save('scores/' + args.task + '/true_surf_coefs_' + str(n), file)
#     for n, file in enumerate(coefs[4]):
#         np.save('scores/' + args.task + '/surf_coefs_' + str(n), file)
#     np.save('scores/' + args.task + '/true_bls', coefs[5])
#     np.save('scores/' + args.task + '/bls', coefs[6])
#     for aero in glob.glob('airFoil2D*'):
#         os.rename(aero, 'scores/' + args.task + '/' + aero)
#     os.rename('score.json', 'scores/' + args.task + '/score.json')
