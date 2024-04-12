import os
import torch
from argparse import ArgumentParser
from mix_part_data import MixPartDataLoader, mix_collect_fn
from mix_part_tools import utils
from torch.utils.data import DataLoader

def reform_dataset(data_loader: DataLoader, train_test):
    save_dir = os.path.join(conf.save_data_dir, conf.category, train_test)
    flag_create = utils.create_directory(save_dir)
    if flag_create != True:
        print("Exit program.")
        exit()
    save_idx = 0
    for _, batch in enumerate(data_loader):
        for gen_idx in range(batch["all_poses"].size(0)):
            data_valid = batch["data_valid"][gen_idx].cpu().bool()
            if not (True in data_valid): # If no data available
                continue
            save_data = dict()
            save_data["all_parts"] = batch["all_parts"].cpu()
            save_data["all_poses"] = batch["all_poses"][gen_idx].cpu()
            save_data["all_euler_poses"] = batch["all_euler_poses"][gen_idx].cpu()
            save_data["total_parts"] = batch["total_parts"][0]
            torch.save(save_data, os.path.join(save_dir, "{}_data_{}.pt".format(conf.category, save_idx)))
            save_idx += 1

def main(conf):
    print('-'*70)
    print('training dataset size:')
    train_data_path = os.path.join(conf.data_dir, conf.category, "train") # You must indicate whether it is training dataset or testing dataset!
    train_set = MixPartDataLoader(conf, train_data_path)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn, worker_init_fn=utils.worker_init_fn)

    print('-'*70)
    print('testing dataset size:')
    test_data_path = os.path.join(conf.data_dir, conf.category, "test") # You must indicate whether it is training dataset or testing dataset!
    test_set = MixPartDataLoader(conf, test_data_path)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn, worker_init_fn=utils.worker_init_fn)
    
    print("reform training dataset.")
    reform_dataset(train_loader, "train")
    if conf.reform_test:
        print("reform testing dataset.")
        reform_dataset(test_loader, "test")

    print("Done!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--data_dir', type=str, default='./dataset/MixPartRandom', help='data directory')
    parser.add_argument('--save_data_dir', type=str, default='./dataset/RandomMixPartRandom')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--reform_test', action='store_true', default=False)

    conf = parser.parse_args()
    main(conf)
