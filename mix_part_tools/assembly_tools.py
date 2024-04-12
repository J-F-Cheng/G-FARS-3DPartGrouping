'''
Author: J-F-Cheng
Assembly tools for mix part assembly task


'''

import os
import torch
from torch import Tensor
from .quaternion import qrot
from .point_cloud_render import point_cloud_render
from .utils import euler_to_quaternion_torch_data

# Part selection for one part repository
def part_sel(conf, input_part_pcs: Tensor, poses: Tensor, pose_type):
    '''
    T: total_part_num
    Input: 1. input parts T x 1000
            2. poses containing the selection T x 7 or T x 8 (Normally it is T x 8)
            3. selection threshold
            4*. batch code (no need since this function only used in final assembly, not learning process)
    Output: The selected parts and poses
    
    '''

    if conf.sel_first:
        choose_part_pose = poses[..., 0] > conf.sel_thre
        sel_input_part_pcs = input_part_pcs[choose_part_pose]
        sel_poses = poses[choose_part_pose]
        
        # from euler check. The output should be qua
        if pose_type=='euler':
            out_poses = euler_to_quaternion_torch_data(sel_poses[..., 1:], conf.euler_type, conf.device)
        elif pose_type=='qua' or pose_type=='euler_direct':
            out_poses = sel_poses[..., 1:]
        else:
            print('Not a recognised type!')
            raise RuntimeError

        # Return poses without selection place
        return {'sel_input_part_pcs': sel_input_part_pcs, 'sel_poses': out_poses, 'sel_vector': poses[..., 0]}
    else:
        choose_part_pose = poses[..., -1] > conf.sel_thre
        sel_input_part_pcs = input_part_pcs[choose_part_pose]
        sel_poses = poses[choose_part_pose]

        # from euler check. The output should be qua
        if pose_type=='euler':
            out_poses = euler_to_quaternion_torch_data(sel_poses[..., :-1], conf.euler_type, conf.device)
        elif pose_type=='qua' or pose_type=='euler_direct':
            out_poses = sel_poses[..., :-1]
        else:
            print('Not a recognised type!')
            raise RuntimeError

        # Return poses without selection place
        return {'sel_input_part_pcs': sel_input_part_pcs, 'sel_poses': out_poses, 'sel_vector': poses[..., -1]}

def transform_point_cloud_tait_bryan_xyz(point_cloud, transform_params):
    """
    Transforms a batch of point clouds using translation and Tait-Bryan angles in xyz order
    
    Parameters:
    - point_cloud: Tensor of shape (batch_size, N, 3) representing the point clouds
    - transform_params: Tensor of shape (batch_size, 6) representing the translation and Tait-Bryan angles in xyz order for each point cloud
    
    Returns:
    - transformed_point_cloud: Tensor of shape (batch_size, N, 3) representing the transformed point clouds
    """
    batch_size, num_points, _ = point_cloud.shape
    
    translation = transform_params[:, :3].unsqueeze(-2)
    angles = transform_params[:, 3:]
    
    c_x = torch.cos(angles[:, 0])
    c_y = torch.cos(angles[:, 1])
    c_z = torch.cos(angles[:, 2])
    s_x = torch.sin(angles[:, 0])
    s_y = torch.sin(angles[:, 1])
    s_z = torch.sin(angles[:, 2])
    
    rotation_matrix = torch.stack([torch.stack([c_y*c_z, s_x*s_y*c_z+c_x*s_z, c_x*s_y*c_z-s_x*s_z], -1),
                                   torch.stack([-c_y*s_z, c_x*c_z-s_x*s_y*s_z, s_x*c_z+c_x*s_y*s_z], -1),
                                   torch.stack([s_y, -s_x*c_y, c_x*c_y], -1)], -2)
    
    rotated_point_cloud = torch.matmul(point_cloud, rotation_matrix)
    transformed_point_cloud = rotated_point_cloud + translation
    
    return transformed_point_cloud

def assembly_parts(conf, sel_parts_poses, pose_type):
    '''
    Input: 1. selected input parts selected poses [dict] (selection place is not included)
    Output: assembled shape
    
    '''
    num_point = sel_parts_poses['sel_input_part_pcs'].shape[1]
    cur_poses = sel_parts_poses['sel_poses']
    cur_input_parts = sel_parts_poses['sel_input_part_pcs']
    # print("cur_poses.size(): ", cur_poses.size())

    if cur_poses.size(0) == 0: # If no parts selected
        return None
    
    if pose_type == "euler" or pose_type == "qua":
        return qrot(cur_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_parts) + \
                                        cur_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
    elif pose_type == "euler_direct":
        return transform_point_cloud_tait_bryan_xyz(cur_input_parts, cur_poses)

def pyg_batch_to_list(pyg_batch, batch_code, batch_size):
    '''
    This function may not be neccessary.
    Input: a pyg batch with batch code
    Output: a list for the batch
    
    '''
    batch_list = []
    for batch_idx in range(batch_size):
        batch_sel = batch_code == batch_idx # Select batch data
        batch_list.append(pyg_batch[batch_sel])

    return batch_list

def batch_assembly(conf, batch_parts, batch_poses, batch_code=None, pose_type='qua', data_idx_start=None, gen_idx=None, save_fn=None, save_tag=""):
    '''
    Input: 1. the batch data of part repositories

    Output: [list] batch assembled shapes

    '''
    if batch_code != None:
        batch_size = max(batch_code) + 1
        parts_list = pyg_batch_to_list(batch_parts, batch_code, batch_size)
        poses_list = pyg_batch_to_list(batch_poses, batch_code, batch_size)
    else:
        parts_list = batch_parts
        poses_list = batch_poses
    shapes = []
    render_imgs = []
    for batch_idx in range(batch_size):
        sel_parts_poses = part_sel(conf, parts_list[batch_idx], poses_list[batch_idx], pose_type=pose_type)
        if save_fn != None:
            save_path = os.path.join(save_fn, "{}_data_{}_{}{}.pt".format(conf.category, data_idx_start + batch_idx, gen_idx, save_tag))
            save_pose_dict = {"sel_poses": sel_parts_poses['sel_poses'], "sel_vector": sel_parts_poses["sel_vector"]}
            torch.save(save_pose_dict, save_path)
        shapes.append(assembly_parts(conf, sel_parts_poses, pose_type))
    
    # Rendering
    if conf.obj_png == "png" or conf.obj_png == "both":
        for batch_idx in range(batch_size):
            if save_fn != None:
                save_path = os.path.join(save_fn, "{}_data_{}_{}{}.png".format(conf.category, data_idx_start + batch_idx, gen_idx, save_tag))
                cur_shape = shapes[batch_idx]
                if cur_shape != None:
                    render_img = point_cloud_render(save_path, cur_shape, conf)
                else:
                    # Create a dummy image, only shown in tensorboard, not saved
                    render_img = torch.zeros(conf.render_img_size, conf.render_img_size, 3, device=conf.device)
            else:
                cur_shape = shapes[batch_idx]
                if cur_shape != None:
                    render_img = point_cloud_render(None, cur_shape, conf)
                else:
                    # Create a dummy image, only shown in tensorboard, not saved
                    render_img = torch.zeros(conf.render_img_size, conf.render_img_size, 3, device=conf.device)
            render_imgs.append(render_img)
    
    if conf.obj_png == "obj" or conf.obj_png == "both":
        NotImplementedError

    return {'shapes': shapes, 'render_imgs': render_imgs}

