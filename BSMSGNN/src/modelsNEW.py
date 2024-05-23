import torch
# from ops import GMP, MLP, ContactGMP
# from BSMS import BSGMP
from opsNEW import FVGMP, MLP, ContactGMP
from BSMSnew import FVBSGMP


class FVModelGeneral(torch.nn.Module):
    def __init__(self, pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian, MP_model, edge_set_num, has_contact, use_FV=True, use_SAF=True, use_dSDF=True, use_res=True, use_FV2=False):
        super(FVModelGeneral, self).__init__()
        self.FV = use_FV; self.saf = use_SAF; self.dsdf = use_dSDF; #self.res = use_res
        self.FV2 = use_FV2 #use FV only in first conv but not second.
        
        in_dim += 2 if use_SAF else 0; in_dim += 8 if use_dSDF else 0
        # in_dim += 3 if use_res else 0
        # print("in_dim = ",in_dim)
        if use_FV:
            self.encode = MLP(in_dim+1, ld, ld, mlp_hidden_layer, True)
            self.process = FVBSGMP(layer_num, ld, mlp_hidden_layer, pos_dim, lagrangian, MP_model,\
                                   edge_set_num, use_FV=use_FV, use_FV2=use_FV2)
            self.decode = MLP(ld+1, ld, out_dim, mlp_hidden_layer, False)
        else:
            self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
            self.process = FVBSGMP(layer_num, ld, mlp_hidden_layer, pos_dim, lagrangian, MP_model,\
                                   edge_set_num, use_FV=use_FV)
            self.decode = MLP(ld, ld, out_dim, mlp_hidden_layer, False)
            
        self.MP_times = MP_times
        self.pos_dim = pos_dim
        self.mse = torch.nn.MSELoss(reduction='none')

    def _get_nodal_latent_input(self, node_in):
        # NOTE implement in childs
        # NOTE we want to remove absolute position from input
        return node_in

    def _get_pos_type(self, node_in):
        # NOTE by defualt, we agree in feature ends with X,type
        return node_in[..., -(1 + self.pos_dim):-1].clone(), node_in[..., -1].clone()

    def _penalize(self, loss, pen_coeff):
        # NOTE implement in childs, pen_coeff shape should be [B(or 1),F] or [F]
        # loss in [B,N,F]
        if len(pen_coeff.shape) == 2:
            pen_coeff = pen_coeff.unsqueeze(1)
        return loss * pen_coeff

    # def _update_states(self, node_in, node_tar, node_type, out):
    #     # NOTE implement in childs
    #     return out

    def _pre(self, node_in, node_tar, node_type):
        # NOTE implement in childs
        return node_in

    def _mask(self, node_in, node_tar, node_type, out):
        # NOTE implement in childs
        mask = torch.ones_like(node_tar)
        return out, mask

    def _EMD(self, node_feature, m_ids, multi_gs, pos, data=None):
        # print('(bef0) node_feature: ',node_feature.shape)
        # for i in range(len(m_ids)):
        #         assert (node_feature.shape[1]>max(m_ids[i]) )
        node_feature = self._get_nodal_latent_input(node_feature)
        # print('(bef) node_feature: ',node_feature.shape)
        # print('node_feature <=> ',torch.any(torch.isnan(node_feature)))
        
        # if self.saf: #SV implementation
        #     node_feature = torch.cat((node_feature,data.saf),dim=2)
        # if self.dsdf: #DID implementation
        #     node_feature = torch.cat((node_feature,data.dsdf),dim=2)
        if self.FV: #FVF implementation
            data.node_FVattr = data.node_FVattr.to(node_feature.get_device())
            node_feature = torch.cat((node_feature,data.node_FVattr),dim=-1)
        
        # print('node_feature: ',node_feature.shape)
        # old_x_shape = node_feature.shape[1]
        # for i in range(len(m_ids)):
        #         assert (node_feature.shape[1]>max(m_ids[i]) )
        x = self.encode(node_feature)
        # print('x = ',torch.any(torch.isnan(x)))
        # assert (x.shape[1]==old_x_shape)
        for _ in range(self.MP_times):
            # for i in range(len(m_ids)):
            #     assert (x.shape[1]>max(m_ids[i]) )
            x = self.process(x, m_ids, multi_gs, pos, data=data)
        # print('process x = ',torch.any(torch.isnan(x)))
        if self.FV: #FVF implementation
            x = torch.cat((x,data.node_FVattr),dim=-1)
            
        x = self.decode(x)
        return x

    def forward(self, m_idx, m_gs, node_in, node_tar, pen_coeff=None, data=None):
        # get mat pos and type
        # print(node_in.shape)
        node_pos, node_type = self._get_pos_type(node_in)
        # print(node_pos.shape,' -- ',node_type.shape)
        # preprocess: set scripted bcs
        # node_in = self._pre(node_in, node_tar, node_type)
        # infer: encode->MP->decode->time integrate to update states
        # print('')
        for i in range(len(m_idx)):
                assert (node_in.shape[1]>max(m_idx[i]) )
        # print('-----')
        # print('node_in: ',torch.any(torch.isnan(node_in)))
        out = self._EMD(node_in, m_idx, m_gs, node_pos, data=data)
        # out = self._update_states(node_in, node_tar, node_type, out)
        # masking: e.g. 1st kind bc, scripted bc
        out, mask = self._mask(node_in, node_tar, node_type, out)
       
        # print('out: ',torch.any(torch.isnan(out)))
        # print('mask: ',torch.any(torch.isnan(mask)))
        # error cal
        loss = self.mse(out, node_tar)
       
        # print(torch.any(torch.isnan(loss)))
        if pen_coeff != None:
            loss = self._penalize(loss, pen_coeff)
        loss = (loss * mask).sum()
        # print(torch.any(torch.isnan(loss)))
        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements
        # print(torch.any(torch.isnan(mse_loss_val)))
        # print('------')

        return mse_loss_val, out, non_zero_elements

class FVAirfoil(FVModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, mlp_hidden_layer, MP_times,use_FV=True, use_FV2= False, use_SAF=True, use_dSDF=True, use_res=True):
        in_dim = 1   # type
        out_dim = pos_dim + 1  # vel,pressure
        MP_model = FVGMP
        super(FVAirfoil, self).__init__(pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian=False, MP_model=MP_model, edge_set_num=1, has_contact=False,\
                                      use_FV=use_FV, use_FV2 = use_FV2, use_SAF=use_SAF, use_dSDF=use_dSDF, use_res=use_res)

    # def _update_states(self, node_in, node_tar, node_type, out):
    #     # donot time integrate pressure
    #     out[..., :-1] = out[..., :-1] + node_in[..., :-(1 + self.pos_dim)]
    #     return out

    # def _pre(self, node_in, node_tar, node_type):
    #     # 0 => int node
    #     preset_node = (node_type != 0).bool().unsqueeze(-1)
    #     node_in[..., :self.pos_dim] = torch.where(preset_node, node_tar[..., :self.pos_dim], node_in[..., :self.pos_dim])
    #     return node_in

    def _mask(self, node_in, node_tar, node_type, out):
        # 0 => int node
        int_node = (node_type == 0).bool().unsqueeze(-1)
        mask = torch.where(int_node, torch.ones_like(node_tar), torch.zeros_like(node_tar))
        # print(int_node.shape)
        # print(out.shape)
        # print(node_tar.shape)
        out = torch.where(int_node, out, node_tar) # physics prior on bc
        return out, mask

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [vel,density,type] out of [vel,density,pos,type]
        # return torch.cat((node_in[..., :self.pos_dim + 1], node_in[..., -1:]), dim=-1)
        # at this point node_in = [pos,node_type,saf,dsdf]
        if self.saf and self.dsdf:
            return node_in[...,2:] # => [node_type,saf,dsdf]
        else:
            return node_in[...,2:3] # => [node_type]


