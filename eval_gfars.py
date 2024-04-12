import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import random
from mix_part_data import MixPartDataLoader, mix_collect_fn, random_mix_collect_fn
from mix_part_tools import utils
from mix_part_tools.assembly_tools import batch_assembly
from model_sel_SDE import AssembleModel, eval_selection_batch, calc_prf, reform_gt_sel_list
from torch.utils.tensorboard import SummaryWriter
from define_dict import DATASET_POSE_TYPE, INPUT_DIM
from torch_geometric.nn import DataParallel

def main(conf):
    gen_img_path = os.path.join(conf.base_dir, conf.exp_name, "gen_img")
    if not os.path.exists(gen_img_path):
        os.makedirs(gen_img_path)
    ref_img_path = os.path.join(conf.base_dir, conf.exp_name, "ref_img")
    if not os.path.exists(ref_img_path):
        os.makedirs(ref_img_path)
    print('-'*70)
    print('testing dataset size:')
    test_data_path = os.path.join(conf.data_dir, conf.category, "test") # You must indicate whether it is training dataset or testing dataset!
    test_set = MixPartDataLoader(conf, test_data_path)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=conf.val_batch_size, shuffle=False, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn, worker_init_fn=utils.worker_init_fn)
    
    # Create the model
    model_sel_SDE = AssembleModel(conf, INPUT_DIM[conf.pose_type])
    model_sel_SDE = model_sel_SDE.to(conf.device)
    model_sel_SDE.load_state_dict(torch.load(conf.model_path, map_location=conf.device))
    model_sel_SDE.eval()
    # initialize the TP, FP, FN, TN
    tp = 0
    fp = 0
    fn = 0
    # initialize the list of batch macro_precision, macro_recall, macro_f1
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    with torch.no_grad():
        save_idx = 0
        for _, val_batch in enumerate(test_loader):
            test_batch = val_batch["all_parts"].to(conf.device)
            test_batch_code = val_batch["batch_code"].to(conf.device)
            test_knn_k = max(val_batch["total_parts"])
            # perform auto-regressive sampling
            pred_sel_list, pred_all_sel_tensor = model_sel_SDE.ar_sampling(test_batch, test_batch_code, test_knn_k)
            gt_sel_tensor = val_batch[DATASET_POSE_TYPE["qua"]][:, :, -1]
            # use conf.sel_thre to get the final selection
            gt_sel_tensor = gt_sel_tensor > conf.sel_thre # 4 * N
            # get gt_sel_list
            gt_sel_list = reform_gt_sel_list(test_batch_code, gt_sel_tensor)
            # obtain tp, fp, fn, tn
            eval_dict = eval_selection_batch(pred_sel_list, gt_sel_list)
            # update tp, fp, fn, tn
            tp += eval_dict["tp"]
            fp += eval_dict["fp"]
            fn += eval_dict["fn"]
            # update macro_precision, macro_recall, macro_f1, directly add to the list
            macro_precision_list += eval_dict["macro_precision_list"]
            macro_recall_list += eval_dict["macro_recall_list"]
            macro_f1_list += eval_dict["macro_f1_list"]
            # record in tensorboard
            gt_poses = val_batch[DATASET_POSE_TYPE["qua"]].to(conf.device)[:, :, :-1].sum(dim=0) # become a 2D tensor
            # we repeat gt_poses at dim 0 for pred_all_sel_tensor.size(0) times
            # unsequeeze dim 0
            gt_poses = gt_poses.unsqueeze(0)
            gt_poses = gt_poses.repeat(pred_all_sel_tensor.size(0), 1, 1)
            # We concatenate gt_poses and pred_all_sel_tensor along dim -1
            gen_sel_poses = torch.cat([gt_poses, pred_all_sel_tensor.float().unsqueeze(-1).to(conf.device)], dim=-1)
            for gen_idx in range(gen_sel_poses.size(0)):
                batch_assembly(conf, test_batch, gen_sel_poses[gen_idx].to(conf.device), test_batch_code, pose_type="qua",
                                                  data_idx_start=save_idx, gen_idx=gen_idx, save_fn=gen_img_path)
            for gen_idx in range(val_batch[DATASET_POSE_TYPE["qua"]].size(0)):
                batch_assembly(conf, test_batch, val_batch[DATASET_POSE_TYPE["qua"]][gen_idx].to(conf.device), test_batch_code, pose_type="qua",
                               data_idx_start=save_idx, gen_idx=gen_idx, save_fn=ref_img_path)
            save_idx += conf.val_batch_size
    # calculate precision, recall, f1
    macro_precision = np.mean(macro_precision_list)
    macro_recall = np.mean(macro_recall_list)
    macro_f1 = np.mean(macro_f1_list)
    # use tp, fp, fn, tn to calculate micro_precision, micro_recall, micro_f1
    micro_precision = tp / (tp + fp + 1e-8)
    micro_recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    print("macro_precision: ", macro_precision)
    print("macro_recall: ", macro_recall)
    print("macro_f1: ", macro_f1)
    print("micro_precision: ", micro_precision)
    print("micro_recall: ", micro_recall)
    print("micro_f1: ", micro_f1)
    # save it as dict as npy file
    eval_result_dict = {"macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1,
                        "micro_precision": micro_precision, "micro_recall": micro_recall, "micro_f1": micro_f1}
    np.save(os.path.join(conf.base_dir, conf.exp_name, "eval_result_dict.npy"), eval_result_dict)
    # save it as txt file
    with open(os.path.join(conf.base_dir, conf.exp_name, "eval_result_dict.txt"), "w") as f:
        f.write(str(eval_result_dict))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    # experimental setting
    parser.add_argument('--base_dir', type=str, default='logs', help='model def file')
    parser.add_argument('--exp_name', type=str, default='my_exp_1', help='Please set your exp name, all the output will be saved in the folder with this exp_name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')

    # datasets parameters:
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--data_dir', type=str, default='./data_output', help='data directory')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sel_first', action='store_true', default=False, help='selection at the first place')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)

    # model path
    parser.add_argument('--model_path', type=str, default=None, help='The model path for validation.')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epoch_save', type=int, default=100)
    parser.add_argument('--val_every_epochs', type=int, default=100)
    parser.add_argument('--cont_train_start', type=int, default=0, help='If you want to continue training, please set this option greater than 0.')
    parser.add_argument('--cont_path', type=str, default=None, help='The model path for continue training.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--print_loss', type=int, default=100)

    # Validation options
    parser.add_argument('--val_batch_size', type=int, default=1)

    # Sampler options
    parser.add_argument('--sel_sampler', type=str, default='PC_origin', help='Sampler options: EM, PC and ODE')
    parser.add_argument('--assem_sampler', type=str, default='PC_origin', help='Sampler options: EM, PC and ODE')
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--sel_num_steps', type=int, default=500)
    parser.add_argument('--snr', type=float, default=0.16)
    parser.add_argument('--t0', type=float, default=1.0)
    parser.add_argument('--cor_steps', type=int, default=1)
    parser.add_argument('--cor_final_steps', type=int, default=1)
    parser.add_argument('--noise_decay_pow', type=int, default=1)
    
    # assembly parameters
    parser.add_argument('--pose_type', type=str, default='euler', help='poses type option')
    parser.add_argument('--euler_type', type=str, default='xyz', help='what is the euler type: e.g. xyz')
    parser.add_argument('--sel_thre', type=float, default=0.5) # an important hyper-parameter

    # rendering parameters
    parser.add_argument('--obj_png', type=str, default='png', help='Generation options: obj, png, both and no')
    parser.add_argument('--render_img_size', type=int, default=512)



    conf = parser.parse_args()

    # control randomness
    if conf.seed >= 0:
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)

    main(conf)