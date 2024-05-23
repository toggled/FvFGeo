import logging
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import tqdm
import socket
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn, optim
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

from torch_geometric.data import DataLoader
from data import MeshAirfoilDataset, ccMeshAirfoilNSDataset #,ccMeshAirfoilDataset
from modelsNEWNoNut import *
from torch_geometric.nn.unpool import knn_interpolate #for residual

server = socket.gethostname()
if server == 'neptune':
    vtk_folder = '/ntuzfs/data/cfdgcnRR/Fine_vtk/'
    cc_folder = '/ntuzfs/data/cfdgcnRR/ccfine_ns/'
if server == 'police01':
    vtk_folder = '/home/jessica/Downloads/submission/cfd-gcn_geom/Fine_vtk/'
    cc_folder = '/home/jessica/Downloads/submission/cfd-gcn_geom/ccfine_ns/'
    
vtk_folder = '/home/jessica/Downloads/submission/cfd-gcn_geom/Fine_vtk/'
cc_folder = '/home/jessica/Downloads/submission/cfd-gcn_geom/ccfine_ns/'

parser = ArgumentParser()
parser.add_argument('--exp-name', '-e', default='',
                    help='Experiment name, defaults to model name.')
parser.add_argument('--version', type=int, default=None,
                    help='If specified log version doesnt exist, create it.'
                            ' If it exists, continue from where it stopped.')
parser.add_argument('--load-model', '-lm', default='', help='Load previously trained model.')

parser.add_argument('--model', '-m', default='mgn', # mgn/fvmgn
                    help='Which model to use.')
parser.add_argument('--max-epochs', '-me', type=int, default=250, #500,
                    help='Max number of epochs to train for.')
parser.add_argument('--optim', default='adam', help='Optimizer.')
parser.add_argument('--batch-size', '-bs', type=int, default=1)
parser.add_argument('--learning-rate', '-lr', dest='lr', type=float, default=1e-4) 
parser.add_argument('--hidden-size', '-hs', type=int, default=128)
parser.add_argument('--gpus', nargs='*', type=int, help='cuda(s) to use, e.g. --gpus 0 1 for using [0,1].')
parser.add_argument('--dataloader-workers', '-dw', type=int, default=32,
                    help='Number of Pytorch Dataloader workers to use.')
# parser.add_argument('--train-val-split', '-tvs', type=float, default=0.9,
#                     help='Percentage of training set to use for training.')
# parser.add_argument('--val-check-interval', '-vci', type=int, default=None,
#                     help='Run validation every N batches, '
#                          'defaults to once every epoch.')
# parser.add_argument('--early-stop-patience', '-esp', type=int, default=0,
#                     help='Patience before early stopping. '
#                          'Does not early stop by default.')
# parser.add_argument('--train-pct', type=float, default=1.0,
#                     help='Run on a reduced percentage of the training set,'
#                          ' defaults to running with full data.')
parser.add_argument('--verbose', type=int, default=1, choices=[0, 1],
                    help='Verbosity level. Defaults to 1, 0 for quiet.')

parser.add_argument('--no-log', action='store_true',
                    help='Dont save any logs or checkpoints.')

#ADDED:
parser.add_argument('--saf', type=int, default=1,
                    help='0, or 1 to use SAF in the input.')
parser.add_argument('--dsdf', type=int, default=1,
                    help='0, or 1 to use dSDF in the input.')
parser.add_argument('--FV','-FV', action='store_true',
                    help='Use FV attributes in the convolutions.')
parser.add_argument('--residual', action='store_true', #type=bool, default=False,
                    help='Add coarse mesh solution to network output before prediction.')
parser.add_argument('--data_type', default='scarce',
                    help='Which datasets: scarce, full, reynolds or aoa.')
#ADDED####

args = parser.parse_args()
args.nodename = os.uname().nodename
if args.exp_name == '':
    args.exp_name = args.model
args.distributed_backend = 'dp'

torch.multiprocessing.set_sharing_strategy('file_system') #ADDED for pl error with residual

def get_training_data():
    """ Loads training data as Dataloader """
    if (args.model == 'fvmgn') or  (args.model == 'fvmgnres'):
        train_data = ccMeshAirfoilNSDataset(root=cc_folder,data_type=args.data_type,mode='train')
        if (args.model == 'fvmgnres'):
            new_data = []
            for data in train_data: #[:10]:
                coarse_data = torch.load(data.coarse_path)
                coarse_x, coarse_y = coarse_data.x, train_data.coarse_preprocess(coarse_data.y[:,:2])
                coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)
                # coarse_batch = coarse_data.batch
                data.estimate = knn_interpolate(coarse_y, coarse_x[:, :2], data.pos.to(dtype=torch.float32), k=3)
                new_data.append(data)
            train_data = new_data
    else: # MGN
        train_data = MeshAirfoilDataset(root=vtk_folder,data_type=args.data_type,mode='train')
    train_loader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.dataloader_workers)
    # in_channels = train_data.get(0).x.shape[-1]#self.data[0].x.shape[-1]
    # out_channels = train_data.get(0).y.shape[-1] #self.data[0].y.shape[-1]
    in_channels = train_data[0].x.shape[-1]
    out_channels = train_data[0].y.shape[-1]
    print('out_channels: ',out_channels)

    return train_loader, in_channels, out_channels

def get_val_data():
    """ Loads validation data as Dataloader """
    if (args.model == 'fvmgn') or  (args.model == 'fvmgnres'):
        val_data = ccMeshAirfoilNSDataset(root=cc_folder, data_type=args.data_type,mode='val')
        if (args.model == 'fvmgnres'):
            new_data = []
            for data in val_data:
                coarse_data = torch.load(data.coarse_path)
                coarse_x, coarse_y = coarse_data.x, val_data.coarse_preprocess(coarse_data.y[:,:2])
                coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)#[:,:3]
                # coarse_batch = coarse_data.batch
                data.estimate = knn_interpolate(coarse_y, coarse_x[:, :2], data.pos.to(dtype=torch.float32), k=3)
                new_data.append(data)
            val_data = new_data
    else:
        val_data = MeshAirfoilDataset(root=vtk_folder, data_type=args.data_type,mode='val')
    
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.dataloader_workers)
    # if self.hparams.verbose:
    #     logging.info(f'Val data: {len(self.val_data)} examples, '
    #                  f'{len(val_loader)} batches.')
    return val_loader

def get_test_data():
    """ Loads training data as Dataloader """
    if (args.model == 'fvmgn') or  (args.model == 'fvmgnres'):
        test_data = ccMeshAirfoilNSDataset(root = cc_folder, data_type=args.data_type,mode='test')
        if (args.model == 'fvmgnres'):
            new_test_data = []
            for data in test_data:
                coarse_data = torch.load(data.coarse_path)
                coarse_x, coarse_y = coarse_data.x, test_data.coarse_preprocess(coarse_data.y[:,:2])
                coarse_x, coarse_y = coarse_x.to(dtype=torch.float32), coarse_y.to(dtype=torch.float32)#[:,:3]
                # coarse_batch = coarse_data.batch
                data.estimate = knn_interpolate(coarse_y, coarse_x[:, :2], data.pos.to(dtype=torch.float32), k=3)
                new_test_data.append(data)
            test_data = new_test_data
    else:
        test_data = MeshAirfoilDataset(root = vtk_folder, data_type=args.data_type,mode='test')
    test_loader = DataLoader(test_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.dataloader_workers)
    return test_loader

def get_model(in_channels, out_channels):
    """ Returns the appropriate model """
    hidden_channels = args.hidden_size
    if args.model == 'mgn':
        model = MGN(node_channels = in_channels,
                    edge_channels = 3,
                        hidden_channels=hidden_channels,
                        saf = args.saf, # Added 
                        dsdf = args.dsdf
                    )
        
    elif args.model == 'fvmgn':
        model = FVMGN(node_channels = in_channels, 
                        edge_channels = 3, 
                        hidden_channels = hidden_channels,
                        saf = args.saf, 
                        dsdf = args.dsdf, 
                        use_FV = True
                        )
    elif args.model == 'fvmgnres':
        model = FVMGN_residual(node_channels = in_channels, 
                        edge_channels = 3, 
                        hidden_channels = hidden_channels,
                        saf = args.saf, 
                        dsdf = args.dsdf, 
                        use_FV = True,
                        use_res = True,
                        )
    else:
        raise NotImplementedError
    return model 

class LightningWrapper(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.save_hyperparameters(hparams)  

        # self.step = None  # count test step because apparently Trainer doesnt
        self.criterion = nn.MSELoss()


    def forward(self, *args, **kwargs):
        # in lightning, forward defines the prediction/inference actions
        output = self.model(*args, **kwargs)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        decay_steps = 5000000
        initial_lr = self.hparams.lr
        final_lr = 1e-6

        # Define a learning rate scheduler using exponential decay
        def lr_lambda(step):
            if step < decay_steps:
                decay_factor = (final_lr / initial_lr) ** (1.0 / decay_steps)
                return decay_factor ** step
            else:
                return final_lr / initial_lr
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer] ,[scheduler]

    def training_step(self, batch, batch_idx):
        pred_fields = self.forward(batch)
        true_fields = batch.y
        loss = self.criterion(pred_fields, true_fields)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # you can run only the validation loop using trainer.validate(model)
        # loads the best checkpoint automatically by default if checkpointing was enabled during fitting.
        outputs = self.forward(batch)
        val_loss = self.criterion(outputs, batch.y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        test_loss = self.criterion(outputs, batch.y)
        # self.log("test_loss", test_loss)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True)
        torch.save(self.model, self.hparams.exp_name+'_final0'+str(0)) #ADDED####

if __name__ == '__main__':
    print(args, file=sys.stderr)
    train_loader,in_channels,out_channels = get_training_data()
    val_loader = get_val_data()
    model = get_model(in_channels=in_channels,out_channels=out_channels)
    pl_model = LightningWrapper(model,args)
    if not args.no_log:
        # logger = TensorBoardLogger(args.model+'_logs/', name=args.model) #tensorboard --logdir tb_logs/
        # csv_logger = CSVLogger(save_dir=args.model+"_csvlogs", name=args.model)
        logger = TensorBoardLogger(args.exp_name+'_logs/', name=args.model)
        csv_logger = CSVLogger(save_dir=args.exp_name+"_csvlogs", name=args.model)

    checkpoint_callback = ModelCheckpoint(dirpath=args.model+'_ckpts/',
                                            monitor='val_loss',
                                            save_top_k=10,
                                            save_weights_only=False,
                                            verbose=args.verbose)
        
    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                            logger=[logger,csv_logger],
                            max_epochs=args.max_epochs,
                            # gradient_clip_val=1,
                            # gradient_clip_algorithm="value",
                            accelerator='gpu', #ADDED: DDP
                            devices = args.gpus, #ADDED: DDP
                            #strategy = 'ddp', #ADDED: DDP
                            # precision=16 #ADDED: half pres
                            precision='16-mixed',
                            )

    model_parameters = filter(lambda p: p.requires_grad, pl_model.parameters()) #ADDED
    model_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('model_parameters: ',model_parameters) #ADDED####
    

    trainer.fit(pl_model,train_dataloaders =train_loader,val_dataloaders=val_loader)

    test_loader = get_test_data()
    trainer.test(ckpt_path='best', dataloaders = test_loader)

    # trainer = pl.Trainer(callbacks=[checkpoint_callback], resume_from_checkpoint='/path/to/your/checkpoint.ckpt')
    # trainer.fit(model, train_loader, val_loader)
    # torch.save(pl_model.model, args.exp_name+"_final"+str(1)) #ADDED####


# MGN (2282370)
# python mainNEW.py -e mgn --model mgn --batch-size 1 --saf 0 --dsdf 0 --hidden-size 128 --gpus 0

# MGN_geom (2283650)
# python mainNEW.py -e mgn_geom --model mgn --batch-size 1 --saf 1 --dsdf 1 --hidden-size 128 --gpus 1

# FVMGN
# python mainNEW.py -e fvmgn_geom --model fvmgn --batch-size 1 --saf 1 --dsdf 1 --hidden-size 128 -FV
# python mainNEW.py -e fvmgn_geom --model fvmgn --batch-size 1 --saf 1 --dsdf 1 --hidden-size 128 -FV --gpus 0

# FVMGN_residual (2302470)
# python mainNEW.py -e fvmgnres_geom --model fvmgnres --batch-size 1 --saf 1 --dsdf 1 --hidden-size 128 -FV --residual --gpus 1