import torch
import os
import numpy as np
import functools
# import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import EdgeConv
from torch_scatter import scatter_sum
from samplers import samples_gen
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from model_tools import PointNet, GaussianFourierProjection, marginal_prob_std, diffusion_coeff


class Graph_SDE(nn.Module):

    def __init__(self, conf, input_dim):
        super(Graph_SDE, self).__init__()
        self.conf = conf
        self.input_dim = input_dim
        self.x_encoder = nn.Linear(self.input_dim, conf.feat_len)

        self.margin_fn = functools.partial(marginal_prob_std, conf=self.conf)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, conf=self.conf)

        self.mlp1 = nn.Sequential(
            nn.Linear(conf.feat_len * 2 + conf.feat_len * 2 + conf.feat_len * 2, conf.feat_len),
            nn.ReLU(True),
            nn.Linear(conf.feat_len, conf.feat_len),
        )
        self.conv1 = EdgeConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(conf.feat_len * 2 + conf.feat_len * 2 + conf.feat_len * 2, conf.feat_len),
            nn.ReLU(True),
            nn.Linear(conf.feat_len, conf.feat_len),
        )
        self.conv2 = EdgeConv(self.mlp2)
        self.mlp3 = nn.Sequential(
            nn.Linear(conf.feat_len * 2 + conf.feat_len * 2 + conf.feat_len * 2, conf.feat_len),
            nn.ReLU(True),
            nn.Linear(conf.feat_len, self.input_dim),
        )
        self.conv3 = EdgeConv(self.mlp3)

        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=conf.feat_len),
                                   nn.Linear(conf.feat_len, conf.feat_len))
        self.act = lambda x: x * torch.sigmoid(x)


    def forward(self, x_pose, t, emb_pcs):
        """x_pose includes x, edge_index and batch """
        t_embed = self.act(self.t_embed(t.squeeze(-1)))

        x = self.x_encoder(x_pose.x)
        x = torch.cat([x, t_embed, emb_pcs], dim=-1)
        x = torch.relu(self.conv1(x, x_pose.edge_index))
        x = torch.cat([x, t_embed, emb_pcs], dim=-1)
        x = torch.relu(self.conv2(x, x_pose.edge_index))
        x = torch.cat([x, t_embed, emb_pcs], dim=-1)
        x = self.conv3(x, x_pose.edge_index)
        x = x / (self.margin_fn(t) + 1e-7)
        return x

    def train_one_batch(self, x_pose, emb_pcs, eps=1e-5):
        batch_size = max(x_pose.batch) + 1
        random_t = torch.rand(batch_size, device=self.conf.device) * (1. - eps) + eps
        random_t = random_t.unsqueeze(-1)
        # [bs, 1] -> [num_nodes, 1]
        random_t = random_t[x_pose.batch]

        z = torch.randn_like(x_pose.x)
        std = self.margin_fn(random_t)
        x_pose.x = x_pose.x + z * std

        # Get score
        score = self.forward(x_pose=x_pose, t=random_t, emb_pcs=emb_pcs)

        # Get loss
        node_l2 = torch.sum((score * std + z) ** 2, dim=-1)
        batch_l2 = scatter_sum(node_l2, x_pose.batch.to(self.conf.device), dim=0)
        loss = torch.mean(batch_l2)
        return loss

    def sample_one_batch(self, emb_pcs, batch_code, knn_k, sampler, num_steps, x_init=None, eps=1e-3):
        if x_init == None:
            x_init = torch.randn(emb_pcs.size(0), self.input_dim, device=self.conf.device)
        batch_code_device = batch_code.to(self.conf.device)
        if knn_k > 100:
            knn_k_input = 100
        else:
            knn_k_input = knn_k
        edge_index = knn_graph(x=x_init, k=knn_k_input, batch=batch_code_device, loop=True) # Fully connected graph
        x = Data(x=x_init, edge_index=edge_index, batch=batch_code_device).to(self.conf.device)
        return samples_gen(self, x, emb_pcs, eps, sampler=sampler, num_steps=num_steps)
    
class AssembleModel(nn.Module):
    def __init__(self, conf, input_dim) -> None:
        super(AssembleModel, self).__init__()
        self.conf = conf
        self.input_dim = input_dim
        self.point_encoder = PointNet(conf.feat_len)
        self.sel_GNN = Graph_SDE(conf, 1) # Only need one dim

    def get_whole_feats(self, pcs):
        # push through PointNet
        return self.point_encoder(pcs)
    
    def forward(self, x_data, pcs):
        emb_pcs = self.get_whole_feats(pcs)
        loss_sel = self.sel_GNN.train_one_batch(x_data["data_sel"], emb_pcs)
        return {"loss_sel": loss_sel}
    
    def get_x_data_train(self, x_init, batch_code_device, knn_k):
        if self.conf.sel_first:
            raise NotImplementedError
        else:
            x_init_sel = x_init[:, -1:]
            if knn_k > 100:
                knn_k_input = 100
            else:
                knn_k_input = knn_k
            edge_sel = knn_graph(x=x_init_sel, k=knn_k_input, batch=batch_code_device, loop=True) # Fully connected graph
            data_sel = Data(x=x_init_sel, edge_index=edge_sel, batch=batch_code_device).to(self.conf.device)
            choose_part_pose = x_init[:, -1] > self.conf.sel_thre
            x_init_assemble = x_init[choose_part_pose, :-1]
            batch_assemble = batch_code_device[choose_part_pose]
            edge_assemble = knn_graph(x=x_init_assemble, k=knn_k_input, batch=batch_assemble, loop=True)
            data_assemble = Data(x=x_init_assemble, edge_index=edge_assemble, batch=batch_assemble).to(self.conf.device)
            return {"data_sel": data_sel, "data_assemble": data_assemble, "choose_part_pose": choose_part_pose}
    
    def sample_new(self, emb_pcs, batch_code, knn_k, sel_sample=None, assemble_sample=None, ass_x_init=None):
        # Get selection
        if sel_sample == None:
            sel_sample = self.sel_GNN.sample_one_batch(emb_pcs, batch_code, knn_k, self.conf.sel_sampler, num_steps=self.conf.sel_num_steps) # N x 1
        choose_part = sel_sample["pred_part_poses"].squeeze(-1) > self.conf.sel_thre
        if True in choose_part:                
            return {"sel_sample": sel_sample, "assemble_sample": assemble_sample, "choose_part": choose_part}
        else:
            return None
    

    def ar_sampling(self, input_pcs, batch_code, knn_k, max_sampling=15):
        """
        Auto-regressive sampling
        We sampling selection bool vector one by one. 
        Select parts -> remove the selected parts -> select parts from the rest parts -> ...
        if the number of parts is less than min_parts, we resample, until the iteration reach max_sampling.
        stop condition 1: we find all the groups
        stop condition 2: the iteration reach max_sampling
        """
        # create 2D list for selections
        batch_size = max(batch_code) + 1
        # Initialize the selection list
        sel_list = [[] for _ in range(batch_size)]
        # Initialize the all selection list
        all_sel_list = []
        # get emb_pcs
        emb_pcs = self.get_whole_feats(input_pcs)
        """
        iterative sampling. Note that we need to utilize the batch inference to accelerate the sampling process.
        Idea: at inference stage, we do batch sampling,
        after interation, we seperate the batch into a list of samples.
        we do operations in the list of samples.
        in the next iteration, we form a new batch from the list to do the next batch sampling.
        """
        # a bool vector to remember all the selected parts
        bool_sel_vec = torch.zeros(input_pcs.size(0), dtype=torch.bool)
        # initialize next emb_pcs, batch_code
        next_emb_pcs = emb_pcs.clone()
        next_batch_code = batch_code.clone()
        # iterative sampling
        for _ in range(max_sampling):
            # get batch data
            if next_emb_pcs.size(0) < 1: # if no parts left, break
                break
            batch_data = self.sample_new(next_emb_pcs, next_batch_code, knn_k)
            if batch_data == None:
                break
            else:
                # current_choose_part, keep the dimension the same with the original emb_pcs
                current_batch_sel_tensor = torch.zeros(input_pcs.size(0), dtype=torch.bool)
                current_batch_sel_tensor[(~bool_sel_vec)] = batch_data["choose_part"].clone().detach().cpu()
                # update all_sel_list
                all_sel_list.append(current_batch_sel_tensor)
                # update bool_sel_vec
                bool_sel_vec[current_batch_sel_tensor] = True
                # check bool_sel_vec by using min_parts
                # use not bool_sel_vec to get the next emb_pcs, batch_code
                next_emb_pcs = emb_pcs[~bool_sel_vec]
                next_batch_code = batch_code[~bool_sel_vec]
                # update sel_list
                for i in range(batch_size):
                    # checking, if the batch only contains False, we skip it
                    if torch.sum(current_batch_sel_tensor[batch_code == i]) == 0:
                        continue
                    else:
                        sel_list[i].append(current_batch_sel_tensor[batch_code == i])
        # convert sel_list to sel_tensor
        for i in range(batch_size):
            if len(sel_list[i]) == 0:
                sel_list[i] = None
            else:
                sel_list[i] = torch.stack(sel_list[i], dim=0)
        # convert all_sel_list to all_sel_tensor
        all_sel_tensor = torch.stack(all_sel_list, dim=0)
        return sel_list, all_sel_tensor
    
def eval_selection_batch(pred_sel_list, gt_sel_list, save_single_path=None, start_idx=None):
    """
    We use this function to obtain the FP, FN, TP, TN
    We use FP, FN, FP, TN to calculate the precision, recall, F1 score
    The sequence of the sel list is unknown, we need to calculate the best matching
    """
    # get the number of batch
    batch_size = len(pred_sel_list)
    # initialize the TP, FP, FN, TN
    tp = 0
    fp = 0
    fn = 0
    # tn = 0
    # initialize the list of batch macro_precision, macro_recall, macro_f1
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    # calculate the TP, FP, FN, TN
    for i in range(batch_size):
        # for pred and gt, the first dimension is the number of samples, we use best matching to calculate the TP, FP, FN, TN
        # get the pred number of samples
        # check if this sample is None
        if pred_sel_list[i] == None:
            continue
        pred_num_samples = pred_sel_list[i].size(0)
        # get the gt number of samples
        gt_num_samples = gt_sel_list[i].size(0)
        # initialize batch tp, fp, fn, tn
        batch_tp = 0
        batch_fp = 0
        batch_fn = 0
        # batch_tn = 0
        # calculate the best matching
        for j in range(gt_num_samples):
            # max_tp, max_fp, max_fn, max_tn
            max_tp = 0
            max_fp = 0
            max_fn = 0
            # max_tn = 0
            max_cur_score = 0
            gt_sel_origin = gt_sel_list[i][j]
            for k in range(pred_num_samples):
                # check if this sample is None
                pred_sel_origin = pred_sel_list[i][k]
                if pred_sel_origin == None:
                    continue
                # We only need the true part of both gt_sel and pred_sel
                need_parts = pred_sel_origin | gt_sel_origin
                gt_sel = gt_sel_origin[need_parts]
                pred_sel = pred_sel_origin[need_parts]
                # calculate the score
                # calculate the TP
                cur_tp = torch.sum(pred_sel & gt_sel)
                cur_fp = torch.sum(pred_sel & ~gt_sel)
                cur_fn = torch.sum(~pred_sel & gt_sel)
                total_num = pred_sel.size(0)
                cur_score = cur_tp / (total_num + 1e-8)
                if cur_score > max_cur_score:
                    # update max
                    max_cur_score = cur_score
                    max_tp = cur_tp
                    max_fp = cur_fp
                    max_fn = cur_fn
                    # max_tn = cur_tn
            # update tp, fp, fn, tn
            tp += max_tp
            fp += max_fp
            fn += max_fn
            # tn += max_tn
            # update batch_tp, batch_fp, batch_fn, batch_tn
            batch_tp += max_tp
            batch_fp += max_fp
            batch_fn += max_fn
            # batch_tn += max_tn
        # for each batch, calcuate the macro_precision, macro_recall, macro_f1, and append them
        # use batch value to calculate
        single_data_eval = {"tp": batch_tp, "fp": batch_fp, "fn": batch_fn}
        macro_eval_dict = calc_prf(single_data_eval)
        if save_single_path != None:
            # we merge two dicts
            save_dict = {**single_data_eval, **macro_eval_dict}
            np.save(os.path.join(save_single_path, str(start_idx + i)), save_dict)
        macro_precision_list.append(macro_eval_dict["precision"])
        macro_recall_list.append(macro_eval_dict["recall"])
        macro_f1_list.append(macro_eval_dict["f1_score"])
    return {"tp": tp, "fp": fp, "fn": fn, "macro_precision_list": macro_precision_list, 
            "macro_recall_list": macro_recall_list, "macro_f1_list": macro_f1_list, "batch_size": batch_size}

def jcard_sim(pred_sel_origin, gt_sel_origin):
    # We only need the true part of both gt_sel and pred_sel
    need_parts = pred_sel_origin | gt_sel_origin
    gt_sel = gt_sel_origin[need_parts]
    pred_sel = pred_sel_origin[need_parts]
    # calculate the score
    # calculate the TP
    cur_tp = torch.sum(pred_sel & gt_sel)
    total_num = pred_sel.size(0)
    cur_score = cur_tp / (total_num + 1e-8)
    return cur_score


# calculate precision, recall, F1 score
def calc_prf(eval_dict):
    """
    calculate the precision, recall, F1 score
    """
    tp = eval_dict["tp"]
    fp = eval_dict["fp"]
    fn = eval_dict["fn"]
    # calculate the precision
    precision = tp / (tp + fp + 1e-8)
    # calculate the recall
    recall = tp / (tp + fn + 1e-8)
    # calculate the F1 score
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def reform_gt_sel_list(batch_code, gt_sel_list):
    """
    gt_sel_list: N * total_num_parts
    We reform it as a batch_size list: N * batch_1; N * batch_2; ...; N * batch_batch_size
    """
    batch_size = max(batch_code) + 1
    # initialize the reform_gt_sel_list
    reform_gt_sel_list = []
    for i in range(batch_size):
        cur_batch_gt_sel_list = gt_sel_list[:, batch_code == i]
        after_remove_cur_batch_gt_sel_list = []
        for n in range(cur_batch_gt_sel_list.size(0)):
            # append only the current batch contains True
            if torch.sum(cur_batch_gt_sel_list[n]) > 0:
                after_remove_cur_batch_gt_sel_list.append(cur_batch_gt_sel_list[n])
        # stack the after_remove_cur_batch_gt_sel_list
        after_remove_cur_batch_gt_sel_list = torch.stack(after_remove_cur_batch_gt_sel_list, dim=0)
        reform_gt_sel_list.append(after_remove_cur_batch_gt_sel_list)
    return reform_gt_sel_list
