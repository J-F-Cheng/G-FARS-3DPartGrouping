import os
from argparse import ArgumentParser
import random
import numpy as np
import torch
from data_dynamic import PartNetPartDataset
from mix_part_tools import utils
from copy import deepcopy
from tqdm import tqdm

def proc_one_batch(conf, batch, data_features, noise_data=False):
    '''
    Input: one batch (For creation purposed batch size is 1)
    
    Output: The processed data which includes three attributes:
    1. point cloud of the parts
    2. the corresponding poses
    3. how many parts
    Return type: dict

    Notice: the first dimension is batch size (always be 1)
    '''
    input_part_pcs = torch.cat(batch[data_features.index('part_pcs')], dim=0)          # point cloud
    input_part_valids = torch.cat(batch[data_features.index('part_valids')], dim=0)    # B x P
    real_num_part = int(input_part_valids[0].sum().item())    # how many parts
    
    if noise_data:
        random_seq = torch.randperm(real_num_part)
        random_how_many = random.randint(1, real_num_part)
        random_part_sel = random_seq[:random_how_many]
        input_part_pcs = input_part_pcs[0, random_part_sel]
        return {"point_cloud": input_part_pcs, "poses": None, "euler_poses": None, "how_many": random_how_many}
    else:
        gt_part_poses = torch.cat(batch[data_features.index('part_poses')], dim=0)     # B x P x (3 + 4) corresponding poses
        
        # Eliminate the 0s in the arrays
        input_part_pcs = input_part_pcs[0, :real_num_part]
        gt_part_poses = gt_part_poses[0, :real_num_part]

        # Get euler poses
        euler_poses = utils.quaternion_to_euler_torch_data(deepcopy(gt_part_poses), conf.euler_type, "cpu")
        
        # Expand poses to eight dim (selection signal)
        sel_signal = torch.ones(gt_part_poses.size(0), 1)
        if conf.sel_first:
            gt_part_poses = torch.cat([deepcopy(sel_signal), gt_part_poses], dim=-1)
            euler_poses = torch.cat([deepcopy(sel_signal), euler_poses], dim=-1)
        else:
            gt_part_poses = torch.cat([gt_part_poses, deepcopy(sel_signal)], dim=-1)
            euler_poses = torch.cat([euler_poses, deepcopy(sel_signal)], dim=-1)

        return {"point_cloud": input_part_pcs, "poses": gt_part_poses, "euler_poses": euler_poses, "how_many": real_num_part}


def pose_padding(conf, pose, before, after):
    if before > 0 and after > 0:
        before_pad = torch.zeros(before, pose.size(-1))
        after_pad = torch.zeros(after, pose.size(-1))
        return torch.cat([before_pad, pose, after_pad], dim=0)
    elif before == 0 and after > 0:
        after_pad = torch.zeros(after, pose.size(-1))
        return torch.cat([pose, after_pad], dim=0)
    elif before > 0  and after == 0:
        before_pad = torch.zeros(before, pose.size(-1))
        return torch.cat([before_pad, pose], dim=0)
    else:
        print("before and after cannot be 0 at the same time! Besides, they cannot be smaller than 0")
        raise ValueError

def parts_fusion_save(conf, fusion_parts, how_many_fusion, max_data, save_path=None, shuffle=True):
    '''
    Input: a list which contains the dicts from proc_one_batch

    Output: includes two attributes:
    1. mixed parts
    2. N arrays of the corresponding poses

    No need to return, save to a path
    '''
    all_parts = []
    total_parts = 0    # We need to know the total parts in advance
    for one_fusion in fusion_parts:
        all_parts.append(one_fusion["point_cloud"])
        total_parts += one_fusion["how_many"]

    all_poses = []
    all_euler_poses = []
    data_valid = []
    
    for i in range(max_data): # Only create poses for real data
        if i < how_many_fusion: # For real data
            posi_shift = 0
            for j in range(i):
                posi_shift += fusion_parts[j]["how_many"]
            all_poses.append(pose_padding(conf, fusion_parts[i]["poses"], posi_shift, total_parts - posi_shift - fusion_parts[i]["how_many"]))
            all_euler_poses.append(pose_padding(conf, fusion_parts[i]["euler_poses"], posi_shift, total_parts - posi_shift - fusion_parts[i]["how_many"]))
            data_valid.append(torch.ones(total_parts).bool())
        else:
            # Add dummy data
            all_poses.append(torch.zeros(total_parts, 8))
            all_euler_poses.append(torch.zeros(total_parts, 7))
            data_valid.append(torch.zeros(total_parts).bool())

    return_dict = {"all_parts": torch.cat(all_parts, dim=0), "all_poses": torch.stack(all_poses, dim=0), 
                   "all_euler_poses": torch.stack(all_euler_poses, dim=0), "total_parts": total_parts, "data_valid": torch.stack(data_valid, dim=0)}

    if shuffle:
        rand_idx = torch.randperm(return_dict["total_parts"])
        return_dict["all_parts"] = return_dict["all_parts"][rand_idx]
        return_dict["all_poses"] = return_dict["all_poses"][:, rand_idx]
        return_dict["all_euler_poses"] = return_dict["all_euler_poses"][:, rand_idx]

    if conf.save_numpy:
        # Transform to numpy
        return_dict['all_parts'] = return_dict['all_parts'].numpy()
        return_dict['all_poses'] = return_dict['all_poses'].numpy()
        return_dict['all_euler_poses'] = return_dict['all_euler_poses'].numpy()

        # Save numpy dict
        np.save(save_path + '.npy', return_dict)
    else:
        # Save pt files
        torch.save(return_dict, save_path + '.pt')

def random_sel(prob_list):
    random_num = random.uniform(0.0, 1.0 - 1e-5)
    prob_add = 0
    for sel in range(len(prob_list)):
        prob_add += prob_list[sel]
        if random_num <= prob_add:
            return sel
    raise ValueError

def iter_batch(conf, train_test_loader, epochs, data_features, save_path, data_how_many, data_how_many_prob):
    noise_data = [0, 1, 2]
    noise_data_prob = [0.8, 0.15, 0.05]
    
    data_id = 0
    for _ in range(epochs):
        if conf.how_many_fusion < 2:
            how_many_fusion = data_how_many[random_sel(data_how_many_prob)]
            max_data = conf.max_data
        else:
            how_many_fusion = conf.how_many_fusion
            max_data = conf.how_many_fusion
        if conf.how_many_noise_data < 0:
            how_many_noise_data = noise_data[random_sel(noise_data_prob)]
        else:
            how_many_noise_data = conf.how_many_noise_data
        total_data = how_many_fusion + how_many_noise_data
        fusion_list = [] # The list for fusion
        for batch_ind, batch in enumerate(tqdm(train_test_loader), 0):
            if len(batch) == 0: # If you get an empty batch, then go next round
                continue
            if len(fusion_list) < how_many_fusion:
                fusion_list.append(proc_one_batch(conf, batch, data_features, False)) # Add data
            elif (len(fusion_list) >= how_many_fusion) and (len(fusion_list) < total_data):
                fusion_list.append(proc_one_batch(conf, batch, data_features, True)) # Add noise data
            if len(fusion_list) == total_data: # If the fusion list reaches the pre defined length, process and save them
                parts_fusion_save(conf, fusion_list, how_many_fusion, max_data, os.path.join(save_path, conf.category + '_data_' + str(data_id)))
                # reset random number, clean the fusion list, data id ++
                if conf.how_many_fusion < 2:
                    how_many_fusion = data_how_many[random_sel(data_how_many_prob)]
                else:
                    how_many_fusion = conf.how_many_fusion
                if conf.how_many_noise_data < 0:
                    how_many_noise_data = noise_data[random_sel(noise_data_prob)]
                else:
                    how_many_noise_data = conf.how_many_noise_data
                total_data = how_many_fusion + how_many_noise_data
                fusion_list = []
                data_id += 1


def main(conf):
    '''Create training dataset'''
    train_data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs']
    print("Creating training dataset...")
    if conf.category == "All":
        train_dataset_chair = PartNetPartDataset("Chair", conf.data_dir, "Chair.train.npy", train_data_features, \
            max_num_part=conf.max_num_part, level=conf.level)
        train_dataset_table = PartNetPartDataset("Table", conf.data_dir, "Table.train.npy", train_data_features, \
            max_num_part=conf.max_num_part, level=conf.level)
        train_dataset_lamp = PartNetPartDataset("Lamp", conf.data_dir, "Lamp.train.npy", train_data_features, \
            max_num_part=conf.max_num_part, level=conf.level)
        # Concatenate datasets
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_chair, train_dataset_table, train_dataset_lamp])
    else:
        train_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.train_data_fn, train_data_features, \
            max_num_part=conf.max_num_part, level=conf.level)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, \
            num_workers=conf.num_workers, drop_last=False, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)

    # Create save path
    train_save_path = os.path.join(conf.new_dataset_dir, conf.category, 'train')
    flag_create = utils.create_directory(train_save_path)
    if flag_create != True:
        print("Exit program.")
        exit()
    train_data_how_many = conf.data_how_many
    train_data_how_many_prob = conf.data_how_many_prob
    print("train_data_how_many: ", train_data_how_many)
    print("train_data_how_many_prob: ", train_data_how_many_prob)
    iter_batch(conf, train_dataloader, conf.train_data_epochs, train_data_features, train_save_path, train_data_how_many, train_data_how_many_prob)

    '''Create Testing dataset'''

    test_data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'contact_points', 'sym', 'pairs', 'match_ids']
    print("Creating testing dataset...")
    if conf.category == "All":
        test_dataset_chair = PartNetPartDataset("Chair", conf.data_dir, "Chair.val.npy", test_data_features, \
                                     max_num_part=conf.max_num_part, level=conf.level)
        test_dataset_table = PartNetPartDataset("Table", conf.data_dir, "Table.val.npy", test_data_features, \
                                     max_num_part=conf.max_num_part, level=conf.level)
        test_dataset_lamp = PartNetPartDataset("Lamp", conf.data_dir, "Lamp.val.npy", test_data_features, \
                                     max_num_part=conf.max_num_part, level=conf.level)
        # Concatenate datasets
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_chair, test_dataset_table, test_dataset_lamp])
    else:
        test_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, test_data_features, \
                                     max_num_part=20, level=conf.level)
                                     
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                                 pin_memory=True, \
                                                 num_workers=0, drop_last=False,
                                                 collate_fn=utils.collate_feats_with_none,
                                                 worker_init_fn=utils.worker_init_fn)

    test_save_path = os.path.join(conf.new_dataset_dir, conf.category, 'test')
    flag_create = utils.create_directory(test_save_path)
    if flag_create != True:
        print("Exit program.")
        exit()
    test_data_how_many = train_data_how_many
    test_data_how_many_prob = train_data_how_many_prob
    print("test_data_how_many: ", test_data_how_many)
    print("test_data_how_many_prob: ", test_data_how_many_prob)
    iter_batch(conf, test_dataloader, conf.test_data_epochs, test_data_features, test_save_path, test_data_how_many, test_data_how_many_prob)

    print("Done.")

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    # data creation parameters:
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--data_dir', type=str, default='../prep_data', help='data directory')
    parser.add_argument('--train_data_fn', type=str, help='training data file that indexs all data tuples')
    parser.add_argument('--val_data_fn', type=str, help='validation data file that indexs all data tuples')
    parser.add_argument('--how_many_fusion', type=int, default=3, help='You need to indicate how many data used to form one part repository.')
    parser.add_argument('--max_data', type=int, default=4, help='You need to indicate how many data used to form one part repository.')
    parser.add_argument('--how_many_noise_data', type=int, default=0, help='You need to indicate how many noise data used to form one part repository.')
    # we receive data_how_many and data_how_many_prob as a list
    parser.add_argument('--data_how_many', type=int, nargs='+', default=[2, 3], help='You need to indicate how many data used to form one part repository.')
    parser.add_argument('--data_how_many_prob', type=float, nargs='+', default=[0.7, 0.3], help='You need to indicate how many data used to form one part repository.')
    parser.add_argument('--train_data_epochs', type=int, default=2)
    parser.add_argument('--test_data_epochs', type=int, default=1)
    parser.add_argument('--new_dataset_dir', type=str, default='./data_output')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_num_part', type=int, default=20)
    parser.add_argument('--level',type=str,default='3',help='level of dataset')
    parser.add_argument('--sel_first', action='store_true', default=False, help='selection at the first place')
    parser.add_argument('--save_numpy', action='store_true', default=False, help='save numpy?')
    parser.add_argument('--euler_type', type=str, default='xyz', help='what is the euler type: e.g. xyz')



    conf = parser.parse_args()

    # control randomness
    if conf.seed >= 0:
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)

    main(conf)
