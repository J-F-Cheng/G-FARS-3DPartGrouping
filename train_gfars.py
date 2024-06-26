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
    # Create exp and log folder
    base_path = os.path.join(conf.base_dir, conf.exp_name)
    writer_log_path = os.path.join(base_path, 'log')
    if not os.path.exists(writer_log_path):
        os.makedirs(writer_log_path)
    # Use tensorboard
    writer = SummaryWriter(writer_log_path)
    print('experiment start.')
    print('-'*70)
    print('training dataset size:')
    train_data_path = os.path.join(conf.data_dir, conf.category, "train") # You must indicate whether it is training dataset or testing dataset!
    train_set = MixPartDataLoader(conf, train_data_path)
    print(len(train_set))
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=random_mix_collect_fn, worker_init_fn=utils.worker_init_fn)

    print('-'*70)
    print('testing dataset size:')
    test_data_path = os.path.join(conf.data_dir_test, conf.category, "test") # You must indicate whether it is training dataset or testing dataset!
    test_set = MixPartDataLoader(conf, test_data_path)
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=conf.val_batch_size, shuffle=True, pin_memory=True, 
                    num_workers=conf.num_workers, collate_fn=mix_collect_fn, worker_init_fn=utils.worker_init_fn)
    
    # Create the model
    model_sel_SDE = AssembleModel(conf, INPUT_DIM[conf.pose_type])
    if conf.cont_train_start > 0:
        print('Continue training, load model from path: ', conf.cont_path)
        model_sel_SDE.load_state_dict(torch.load(conf.cont_path, map_location=conf.device))
    elif conf.cont_train_start < 0:
        print('cont_train_start cannot smaller than 0!')
        raise ValueError
    model_sel_SDE = model_sel_SDE.to(conf.device)

    # Create optimizer
    network_opt = torch.optim.Adam(model_sel_SDE.parameters(), lr=conf.lr)

    # model save path
    model_save_path = os.path.join(base_path, 'model_save')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    
    # Selective training algorithm
    iter_num = 0
    for epoch in range(conf.cont_train_start, conf.epochs):
        model_sel_SDE.train()
        for _, batch in enumerate(train_loader):
            x_init = batch[DATASET_POSE_TYPE[conf.pose_type]].to(conf.device)
            batch_code_device = batch["batch_code"].to(conf.device)
            all_parts = batch["all_parts"].to(conf.device)
            knn_k = max(batch["total_parts"])
            x_data_dict = model_sel_SDE.get_x_data_train(x_init, batch_code_device, knn_k)
            losses = model_sel_SDE(x_data_dict, all_parts)
            loss_sel = losses["loss_sel"]
            loss = loss_sel
            network_opt.zero_grad()
            loss.backward()
            network_opt.step()
            if (iter_num + 1) % conf.print_loss == 0:
                print("epoch {}, loss_sel: {}".format(epoch, loss_sel.item()))
                writer.add_scalars('Training loss:', {"loss_sel": loss_sel.item(),}, iter_num)
            iter_num += 1
        # Save model
        if (epoch + 1) % conf.epoch_save == 0:
            utils.save_network(model_sel_SDE, os.path.join(model_save_path, 'model_epoch_{}.pth'.format(epoch)))
        # Validation
        if (epoch + 1) % conf.val_every_epochs == 0:
            # continue
            # Please fix the following code
            print('validation at epoch {}.'.format(epoch))
            model_sel_SDE.eval()
            _, val_batch = next(enumerate(test_loader))
            val_gen_multi = []
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
            # calculate precision, recall, f1
            eval_result_dict = calc_prf(eval_dict)
            print(eval_result_dict)
            # record in tensorboard
            writer.add_scalars('Validation result:', eval_result_dict, epoch)
            gt_poses = val_batch[DATASET_POSE_TYPE["qua"]].to(conf.device)[:, :, :-1].sum(dim=0) # become a 2D tensor
            # we repeat gt_poses at dim 0 for pred_all_sel_tensor.size(0) times
            # unsequeeze dim 0
            gt_poses = gt_poses.unsqueeze(0)
            gt_poses = gt_poses.repeat(pred_all_sel_tensor.size(0), 1, 1)
            # We concatenate gt_poses and pred_all_sel_tensor along dim -1
            gen_sel_poses = torch.cat([gt_poses, pred_all_sel_tensor.float().unsqueeze(-1).to(conf.device)], dim=-1)

            # Gen show
            gen_list = []
            for gen_idx in range(gen_sel_poses.size(0)):
                batch_assemblies = batch_assembly(conf, test_batch, gen_sel_poses[gen_idx].to(conf.device), test_batch_code, pose_type="qua")
                gen_list.append(torch.stack(batch_assemblies['render_imgs'], dim=0))
            gen_multi = torch.stack(gen_list, dim=0)
            # Show in tensorboard
            for gen_batch_idx in range(gen_multi.size(1)):
                writer.add_images('epoch_{}_gen'.format(epoch), gen_multi[:,gen_batch_idx,...], gen_batch_idx, dataformats='NHWC')

            # Reference show
            val_gen_multi = []
            for gen_idx in range(val_batch[DATASET_POSE_TYPE["qua"]].size(0)):
                batch_assemblies = batch_assembly(conf, test_batch, val_batch[DATASET_POSE_TYPE["qua"]][gen_idx].to(conf.device), test_batch_code, pose_type="qua")
                val_gen_multi.append(torch.stack(batch_assemblies['render_imgs'], dim=0))
            val_gen_multi = torch.stack(val_gen_multi, dim=0)
            # Show in tensorboard
            for gen_batch_idx in range(val_gen_multi.size(1)):
                writer.add_images('epoch_{}_ref'.format(epoch), val_gen_multi[:,gen_batch_idx,...], gen_batch_idx, dataformats='NHWC')
    # Save final model
    utils.save_network(model_sel_SDE, os.path.join(model_save_path, 'model_final.pth'))


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
    parser.add_argument('--data_dir_test', type=str, default='./data_output', help='data directory')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sel_first', action='store_true', default=False, help='selection at the first place')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    
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
    parser.add_argument('--val_how_many_gen', type=int, default=8)

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
