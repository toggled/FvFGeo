import torch
import datasets as cdata
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models
import math
import random
import numpy as np
from helpers_merge import merge_mids, merge_mgs
import os


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._create_model() # self.model_class = models.Airfoil, self.dataset_class = cdata.MeshAirfoilDataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)
        self.writer = SummaryWriter(os.path.join(self.args.dump_dir, 'log'))
        self.pbar = tqdm(range(self.current_epoch, self.args.num_epochs), unit="iters")

        os.makedirs(self.args.dump_dir, exist_ok=True)
        for subdir in ['ckpts', 'log', 'test_RMSE']:
            dir = os.path.join(self.args.dump_dir, subdir)
            os.makedirs(dir, exist_ok=True)

    def _create_model(self):
        if self.args.case.startswith('af'):
            from modelsNEW import FVAirfoil
            self.model_class  = FVAirfoil
            self.dataset_class = cdata.airfrans
        else:
            raise NotImplementedError("A Case not wrapped yet")
        
        # self.model = self.model_class(pos_dim=self.args.space_dim, ld=self.args.hidden_dim, layer_num=self.args.multi_mesh_layer, mlp_hidden_layer=self.args.hidden_depth, MP_times=self.args.mp_time)
        if self.args.case == 'shape_FV_SAF_dSDF' or  self.args.case == 'af_FV_SAF_dSDF':
            print('FVF model')
            self.model = self.model_class(pos_dim=self.args.space_dim, ld=self.args.hidden_dim,\
                    layer_num=self.args.multi_mesh_layer, mlp_hidden_layer=self.args.hidden_depth,\
                    MP_times=self.args.mp_time,use_FV=True,use_FV2 = False, use_SAF=True,use_dSDF=True,use_res=False)
        elif self.args.case == 'shape_SAF_dSDF' or self.args.case == 'af_SAF_dSDF':
            self.model = self.model_class(pos_dim=self.args.space_dim, ld=self.args.hidden_dim,\
                    layer_num=self.args.multi_mesh_layer, mlp_hidden_layer=self.args.hidden_depth,\
                    MP_times=self.args.mp_time,use_FV=False,use_SAF=True,use_dSDF=True,use_res=False)
        elif self.args.case == 'shape' or self.args.case == 'af':
            self.model = self.model_class(pos_dim=self.args.space_dim, ld=self.args.hidden_dim,\
                    layer_num=self.args.multi_mesh_layer, mlp_hidden_layer=self.args.hidden_depth,\
                    MP_times=self.args.mp_time,use_FV=False,use_SAF=False,use_dSDF=False,use_res=False)
        else: # Models other than shape
            self.model = self.model_class(pos_dim=self.args.space_dim, ld=self.args.hidden_dim, layer_num=self.args.multi_mesh_layer, mlp_hidden_layer=self.args.hidden_depth, MP_times=self.args.mp_time)

        POST_FIX_1 = '_layernum_' + str(self.args.multi_mesh_layer)
        POST_FIX_2 = POST_FIX_1 + '_MPHIDDENLAYER_' + str(self.args.hidden_depth) + '_MPHIDDENTDIM_' + str(self.args.hidden_dim) + '_MPtime_' + str(self.args.mp_time) + '_NoiseLevel_' + str(
            self.args.noise_level)
        self.checkpt_name = self.args.case + POST_FIX_2 + '.pt'
        if self.args.restart_epoch >= 0:
            print('hi, restarting')
            self.model.load_state_dict(torch.load(os.path.join(self.args.dump_dir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            self.current_epoch = self.args.restart_epoch + 1
            self.args.lr *= self.args.gamma**(self.current_epoch)
            self.args.lr = max(self.args.lr, 1e-6)
            print('restarted lr is: ', self.args.lr)
        else:
            self.current_epoch = 0

        self.model = self.model.to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters in the model:", num_params)

    def _preproc_multi_infos(self, mdata, b_data):
        # process the multi-level mesh for batched data here
        # 1. if there is contact, merge multiple graphs into a big one by adding offsets (each layer)
        _, n, _ = mdata.in_feature.shape
        if mdata.has_contact:
            # a batch of m_s info
            m_id = b_data.m_idx  # constant across time sequence
            m_gs_list = b_data.m_g  # constant across time sequence
            m_cgs_list = b_data.m_cg  # different across tiem sequence
            # merge midx, mg, mcg across different by adding offset to each layer
            b = int(b_data.x.shape[0] // n)
            b_num_nodes = [n] * b
            merged_midx = merge_mids(m_id, b_num_nodes)
            merged_mg = merge_mgs(m_gs_list, m_id, b_num_nodes)
            merged_mcg = merge_mgs(m_cgs_list, m_id, b_num_nodes)
            # send to device
            m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in merged_mg]
            m_cgs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in merged_mcg]
            # combine edge sets
            m_g_cg = [[g, cg] for g, cg in zip(m_gs, m_cgs)]
            # make dim = 3 for consistency
            b_data.x = b_data.x.unsqueeze(0).to(self.device)
            b_data.y = b_data.y.unsqueeze(0).to(self.device)
            return merged_midx, m_g_cg, b_data
        else:
            # no contact, then share the graph between batches
            # only need to reshape input tensor
            m_ids = mdata.m_idx
            m_gs_list = mdata.m_g
            m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
            # reshape
            # print('B_DATA.U.SHAPE[0]: ',b_data.y.shape[0])
            # print('n = ',n)
            b = b_data.y.shape[0] // n
            b_data.x = b_data.x.reshape(b, n, -1).to(self.device)
            b_data.y = b_data.y.reshape(b, n, -1).to(self.device)
            # print(b_data.x.shape)
            # print(b_data.y.shape)
            # print('len(m_ids) = ',len(m_ids))
            # print('len(m_gs) = ',len(m_gs))
            # for i in range(len(m_ids)):
            #     print(" => ", i, " ", max(m_ids[i]))
            #     assert (b_data.x.shape[1]>max(m_ids[i]) )
            return m_ids, m_gs, b_data

    def _create_datset_offline(self, id, mode='train'):
        # print('_create_datset_offline: ',id,mode)
        if mode == 'train':
            prob = np.random.rand()
            add_noise = (prob < 0.667)
            mdata = self.dataset_class(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       noise_shuffle=add_noise,
                                       noise_level=self.args.noise_level,
                                       noise_gamma=self.args.noise_gamma,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh)
            # for b_data in mdata:
                # print('TESTING 1')
                # x = b_data.x; m_ids = mdata.m_idx
                # for i in range(len(m_ids)):
                #     assert (x.shape[1]>max(m_ids[i]) )
                
        else:
            mdata = self.dataset_class(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       noise_shuffle=False,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       mode=mode)
            # for b_data in mdata:
                # print('TESTING 2')
                # x = b_data.x; m_ids = mdata.m_idx
                # for i in range(len(m_ids)):
                #     assert (x.shape[1]>max(m_ids[i]) )
                    
        return mdata

    def run_epoch(self, epoch, mode='train'):
        # print('epoch start ------')
        if mode != 'train':
            self.model.eval()
        mean_loss_insts = 0
        count_insts = 0
        # if mode != 'train':
        #     self.args.consist_mesh = False 
        instance_len = self.args.n_train if mode == 'train' else (self.args.n_valid if mode == 'valid' else self.args.n_test)
        print('mode: ',mode,' args.consist_mesh = ',self.args.consist_mesh)
        # print(mode,' => len = ',instance_len)
        instance_list = list(range(instance_len))
        random.seed(1)
        if mode == 'train':
            random.shuffle(instance_list)
        # to avoid messy writer, store RMSE 1st in an array
        rmse_array = np.zeros(len(instance_list))
        for i in tqdm(range(len(instance_list)),desc='forward progress: '): # samples/airfoils
            id = instance_list[i]
            #print('id = ',id)
            mean_loss = 0
            count = 0
            if self.args.case.startswith('shape') or self.args.case.startswith('af'):
                if self.args.case == 'shape_FV_SAF_dSDF':
                    mdata = cdata.ShapeFV(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       mode = mode,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       in_normal_feature_list=['mesh_pos','saf','dsdf'])
                elif self.args.case == 'shape_SAF_dSDF':
                    mdata = cdata.Shape(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       mode = mode,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       in_normal_feature_list=['mesh_pos','saf','dsdf'])
                    # for b_data in mdata:
                    #     print('TESTING 3')
                    #     x = b_data.x; m_ids = mdata.m_idx
                    #     for i in range(len(m_ids)):
                    #         assert (x.shape[1]>max(m_ids[i]) )
                elif self.args.case == 'shape':
                    mdata = cdata.Shape(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       mode = mode,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh)
                elif self.args.case == 'af_FV_SAF_dSDF':
                    mdata = cdata.airfransFV(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       mode = mode,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       in_normal_feature_list=['mesh_pos','saf','dsdf'])
                elif self.args.case == 'af_SAF_dSDF':
                    mdata = cdata.airfrans(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       mode = mode,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       in_normal_feature_list=['mesh_pos','saf','dsdf'])
                elif self.args.case == 'af':
                    mdata = cdata.airfrans(self.args.data_dir,
                                       instance_id=id,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=1,
                                       mode = mode,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh)
                else:
                    raise ValueError("not implemented")
                pen_coeff= None
            else:
                mdata = self._create_datset_offline(id, mode=mode)    
                # print(mdata.L)
                # print('mdata = ', type(mdata))
                # print(len(mdata))
                pen_coeff = mdata.suggested_pen_coef().to(self.device)
                # print('pen_coeff: ',pen_coeff)
            for id_batch, b_data in enumerate(DataLoader(mdata, batch_size=1, shuffle=True)): # EDITED
            # for id_batch, b_data in enumerate(DataLoader(mdata, batch_size=self.args.batch, shuffle=True)):
                # print('id_batch = ',id_batch)
                # if id_batch == 0:
                #     print('b_data = ',b_data)
                # for i in range(len(mdata.m_idx)):
                #         assert (b_data.x.shape[1]>max(mdata.m_idx[i]) )
                m_ids, m_gs, b_data = self._preproc_multi_infos(mdata, b_data)
                for i in range(len(m_ids)):
                    assert (b_data.x.shape[1]>max(m_ids[i]) )
                # print('len(m_ids): ',len(m_ids))
                # print('len(m_gs): ',len(m_gs))
                # optimization
                self.optimizer.zero_grad()

                if self.args.case.startswith('shape') or self.args.case.startswith('af'):
                    # for i in range(len(m_ids)):
                    #     assert (b_data.x.shape[1]>max(m_ids[i]) )
                    # if self.args.case == 'shape_SAF_dSDF':
                        # print('bdata.x nan = ',torch.any(torch.isnan(b_data.x)))
                        # print('bdata.y nan = ',torch.any(torch.isnan(b_data.y)))
                    loss, _, non_zero_elements = self.model(m_ids, m_gs, b_data.x, b_data.y, data=b_data, pen_coeff=None)
                    # if self.args.case == 'shape_SAF_dSDF':
                        # print('isnan: ', torch.isnan(loss))
                else:
                    # for i in range(len(m_ids)):
                    #     assert (b_data.x.shape[1]>max(m_ids[i]) )
                    loss, _, non_zero_elements = self.model(m_ids, m_gs, b_data.x, b_data.y, pen_coeff)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                # stats
                mean_loss += loss.item() * non_zero_elements
                count += non_zero_elements
            # stats
            with torch.autograd.no_grad():
                mean_loss_insts += mean_loss
                count_insts += count
                mean_loss /= count
                rmse_array[id] = math.sqrt(mean_loss)
            # safety clean
            del mdata
        # write sequentially afterwards
        for i in range(len(instance_list)):
            self.writer.add_scalar(f'Inst RMSE/{mode}/Epoch: {epoch}', rmse_array[i], i)
        # stats
        mean_loss_insts /= count_insts
        if mode == 'train':
            # opt scheduler
            if self.optimizer.param_groups[0]['lr'] > 1e-6:
                self.scheduler.step()
        else:
            self.model.train()
        # print('epoch end ------')
        return mean_loss_insts

    def train(self):
        for epoch in self.pbar:
            mean_loss_train = self.run_epoch(epoch)
            with torch.autograd.no_grad():
                # train loss record
                self.pbar.set_description(f"Epoch: {epoch}, Traning RMSE: {math.sqrt(mean_loss_train)}")
                self.writer.add_scalar('RMSE/train', math.sqrt(mean_loss_train), epoch)
                self.writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], epoch)
                # dump ckpt
                ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', str(epoch) + "_" + self.checkpt_name)
                torch.save(self.model.state_dict(), ckpt_path)
                # valid loss record
                if self.args.n_valid > 0:
                    mean_loss_valid = self.run_epoch(epoch, mode='valid')
                    print('Validation RMSE = ',math.sqrt(mean_loss_valid))
                    self.writer.add_scalar('RMSE/valid', math.sqrt(mean_loss_valid), epoch)
        self.test()
    def test(self):
        epoch = self.args.restart_epoch
        with torch.autograd.no_grad():
            # reload the model
            # print('test on epoch, ', epoch)
            if self.args.n_test > 0:
                mean_loss_test = self.run_epoch(epoch, mode='test')
                self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_test), epoch)

        RMSE_cell = np.array([math.sqrt(mean_loss_test)])
        print('Test RMSE = ',RMSE_cell)
        dump_path = os.path.join(self.args.dump_dir, 'test_RMSE', 'epoch_' + str(epoch) + '.csv')
        np.savetxt(dump_path, RMSE_cell, delimiter=',')

    def rollout(self):
        instance_list = list(range(self.args.n_test))
        with torch.autograd.no_grad():
            for i in range(len(instance_list)):
                id = instance_list[i]
                # rollout for this instance, then record into a file
                mdata = self._create_datset_offline(id, mode='test')
                L = mdata.in_feature.shape[0]
                instance_rollout_error = np.zeros(L)
                for id_batch, b_data in enumerate(DataLoader(mdata, batch_size=1, shuffle=False)):
                    _, n, _ = mdata.in_feature.shape
                    m_ids = mdata.m_idx
                    m_gs_list = mdata.m_g
                    m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
                    if mdata.has_contact:
                        m_cgs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in mdata.m_cgs[id_batch]]
                        m_g_cg = [[g, cg] for g, cg in zip(m_gs, m_cgs)]
                        m_gs = m_g_cg
                    pen_coeff = mdata.suggested_pen_coef().to(self.device)
                    if id_batch == 0:
                        current_stat = b_data.x.reshape(1, n, -1).to(self.device)
                    b_data.y = b_data.y.reshape(1, n, -1).to(self.device)
                    loss, out, _ = self.model(m_ids, m_gs, current_stat, b_data.y, pen_coeff)
                    # record global error
                    instance_rollout_error[id_batch] = math.sqrt(loss.item())
                    # push forward state
                    current_stat = mdata._push_forward(out, current_stat)

                dir = os.path.join(self.args.dump_dir, 'rollout_RMSE_epoch_' + str(self.args.restart_epoch))
                os.makedirs(dir, exist_ok=True)
                dump_path = os.path.join(dir, str(id) + '.csv')
                np.savetxt(dump_path, instance_rollout_error, delimiter=',')
