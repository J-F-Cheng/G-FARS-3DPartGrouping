import os
import shutil
import torch
import numpy as np
from torch_geometric.nn import DataParallel
from .quaternion import euler_to_quaternion, qeuler

def create_directory(dir_path):
    if os.path.exists(dir_path):
        # If directory exists, ask the user if they want to delete the directory and its contents
        answer = input(f"The directory '{dir_path}' already exists. Do you want to delete the directory and its contents? [y/n]: ")
        if answer.lower() == "y":
            # Delete directory
            shutil.rmtree(dir_path)
        else:
            return False
    # Create directory
    os.makedirs(dir_path)
    return True


def save_network(network, dir):
    if isinstance(network, DataParallel):
        torch.save(network.module.state_dict(), dir)
    else:
        torch.save(network.state_dict(), dir)

def euler_to_quaternion_torch_data(e, order, device):
    """input: n * 6
        output: n * 7"""
    e_clone = e.clone()
    qua_data = torch.zeros(e_clone.size(0), 7)
    qua_data[:, :3] = e_clone[:, :3]
    qua_data[:, 3:] = torch.tensor(euler_to_quaternion(e_clone[:, 3:].cpu().numpy(), order), device=device)
    return qua_data.to(device)

def sel_euler_to_quaternion_torch_data(e, order, device, sel_first=False):
    '''
    Transform euler data with selection place to the quaternion data
    '''

def quaternion_to_euler_torch_data(qua, order, device):
    qua_clone = qua.clone()
    e_data = torch.zeros(qua_clone.size(0), 6)
    e_data[:, :3] = qua_clone[:, :3]
    e_data[:, 3:] = qeuler(qua_clone[:, 3:], order)
    return e_data.to(device)

def collate_feats_with_none(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


