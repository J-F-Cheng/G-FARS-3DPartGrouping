from torch.utils.data import Dataset, DataLoader
import os
import torch

class MixPartDataLoader(Dataset):
    def __init__(self, conf, data_path):
        self.conf = conf
        self.category = conf.category
        self.data_path = data_path

    def __len__(self):
        file_names = os.listdir(self.data_path)
        return len(file_names)

    def __getitem__(self, idx):
        f_name = os.path.join(self.data_path, self.category + '_data_' + str(idx) + '.pt')
        return torch.load(f_name)

class MixPartDataLoader_for_del(Dataset):
    def __init__(self, conf, data_path):
        self.conf = conf
        self.category = conf.category
        self.data_path = data_path
        self.file_names = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return torch.load(self.file_names[idx])

def mix_collect_fn(batches):
    '''
    Input: a list of dicts which contains part repository and the corresponding poses

    Output: 1. the merged batches for the graph neural network
            2. merged batch code

    '''
    return_batches = {"all_parts": [], "all_poses": [], "all_euler_poses": [], "total_parts": [], "batch_code": [], "data_valid": []}

    batch_size = len(batches)

    for batch_idx in range(batch_size):
        return_batches["all_parts"].append(batches[batch_idx]["all_parts"])
        return_batches["all_poses"].append(batches[batch_idx]["all_poses"])
        return_batches["all_euler_poses"].append(batches[batch_idx]["all_euler_poses"])
        return_batches["total_parts"].append(batches[batch_idx]["total_parts"])
        return_batches["batch_code"].append(batch_idx * torch.ones(batches[batch_idx]["total_parts"], dtype=torch.int64))
        return_batches["data_valid"].append(batches[batch_idx]["data_valid"])

    return_batches["all_parts"] = torch.cat(return_batches["all_parts"], dim=0)
    return_batches["all_poses"] = torch.cat(return_batches["all_poses"], dim=1)
    return_batches["all_euler_poses"] = torch.cat(return_batches["all_euler_poses"], dim=1)
    return_batches["batch_code"] = torch.cat(return_batches["batch_code"], dim=0)
    return_batches["data_valid"] = torch.cat(return_batches["data_valid"], dim=1)

    return return_batches

def random_mix_collect_fn(batches):
    return_batches = {"all_parts": [], "all_poses": [], "all_euler_poses": [], "total_parts": [], "batch_code": []}

    batch_size = len(batches)

    for batch_idx in range(batch_size):
        return_batches["all_parts"].append(batches[batch_idx]["all_parts"])
        return_batches["all_poses"].append(batches[batch_idx]["all_poses"])
        return_batches["all_euler_poses"].append(batches[batch_idx]["all_euler_poses"])
        return_batches["total_parts"].append(batches[batch_idx]["total_parts"])
        return_batches["batch_code"].append(batch_idx * torch.ones(batches[batch_idx]["total_parts"], dtype=torch.int64))

    return_batches["all_parts"] = torch.cat(return_batches["all_parts"], dim=0)
    return_batches["all_poses"] = torch.cat(return_batches["all_poses"], dim=0)
    return_batches["all_euler_poses"] = torch.cat(return_batches["all_euler_poses"], dim=0)
    return_batches["batch_code"] = torch.cat(return_batches["batch_code"], dim=0)

    return return_batches
