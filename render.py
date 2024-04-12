import numpy as np
import torch
import os
from mix_part_tools.point_cloud_render import pre_render_process_stack
from mix_part_tools.quaternion import qrot
from mitsuba_render import mitsuba_render

def standardize_bbox(pcl, colors, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    pcl = pcl[pt_indices]
    ret_colors = colors[pt_indices]
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32)
    return result, ret_colors

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.017"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def rotate_axis(points, axis, angle):
    rad = np.radians(angle)
    cos = np.cos(rad)
    sin = np.sin(rad)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    elif axis == 'y':
        rotation_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    elif axis == 'z':
        rotation_matrix = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    else:
        raise ValueError('Invalid axis:', axis)
    return np.dot(points, rotation_matrix)

class Conf:
    def __init__(self) -> None:
        self.device = "cpu"

conf = Conf()


def render_pts(pts, pts_colors, point_save_path, mixed_idx, idx, del_xml):
    pts = np.array(pts.cpu())
    pts_colors = np.array(pts_colors.cpu())

    pts = rotate_axis(pts, "z", 90)
    pts = rotate_axis(pts, "x", 180)
    pts = rotate_axis(pts, "y", 90)

    xml_segments = [xml_head]

    pcl, pcl_colors = standardize_bbox(pts, pts_colors, pts.shape[0])
    for i in range(pcl.shape[0]):
        color = pcl_colors[i]
        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    render_xml = os.path.join(point_save_path, 'render_{}_{}.xml'.format(mixed_idx, idx))
    with open(render_xml, 'w') as f:
        f.write(xml_content)

    mitsuba_render(mixed_idx, idx, point_save_path)
    if del_xml:
        os.remove(render_xml)

def get_global_colors(test_num, gen_idx_list, main_part_set_path, main_pose_sel_file_path, parts_save_path, cate, sel_thresh=0.5, need_inputs=True, del_xml=True):
    # in order to assign the same color for the same parts, we first use gt_single_list to obtain the color list, then in the gen_single_list, we use the same color list
    part_set_path = os.path.join(main_part_set_path, "{}_data_{}.pt".format(cate, test_num))
    part_dict = torch.load(part_set_path, map_location="cpu")
    part_set = part_dict["all_parts"]
    print("part_set size:", part_set.size())
    pts_colors_list = torch.zeros_like(part_set)
    global_part_idx = 0
    for gen_idx in gen_idx_list:
        
        pose_sel_file_path = os.path.join(main_pose_sel_file_path, "{}_data_{}_{}.pt".format(cate, test_num, gen_idx))

        

        pose_sel = torch.load(pose_sel_file_path, map_location="cpu")

        sel_vector_bool = pose_sel["sel_vector"] > sel_thresh
        pose = pose_sel["sel_poses"]
        print("part pose size: ", pose.size())
        sel_parts = part_set[sel_vector_bool]
        num_point = sel_parts.shape[1]
        cur_poses = pose
        cur_input_parts = sel_parts
        sel_parts = qrot(cur_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_parts) + \
                                                cur_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

        _, pts_colors = pre_render_process_stack(sel_parts, conf)
        pts_colors_list[sel_vector_bool] = pts_colors
        if need_inputs:
            for idx in range(cur_input_parts.size(0)):
                render_pts(cur_input_parts[idx], pts_colors[idx], parts_save_path, test_num, global_part_idx, del_xml)
                global_part_idx += 1
    return pts_colors_list
        


def render_one(test_num, gen_idx, main_part_set_path, main_pose_sel_file_path, point_save_path, cate, global_colors, sel_thresh=0.5, del_xml=True):
    part_set_path = os.path.join(main_part_set_path, "{}_data_{}.pt".format(cate, test_num))
    pose_sel_file_path = os.path.join(main_pose_sel_file_path, "{}_data_{}_{}.pt".format(cate, test_num, gen_idx))

    part_dict = torch.load(part_set_path, map_location="cpu")
    part_set = part_dict["all_parts"]
    print("part_set size:", part_set.size())


    pose_sel = torch.load(pose_sel_file_path, map_location="cpu")

    sel_vector_bool = pose_sel["sel_vector"] > sel_thresh
    pose = pose_sel["sel_poses"]
    print("part pose size: ", pose.size())
    sel_parts = part_set[sel_vector_bool]
    num_point = sel_parts.shape[1]
    cur_poses = pose
    cur_input_parts = sel_parts
    sel_parts = qrot(cur_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_parts) + \
                                            cur_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

    pts, _ = pre_render_process_stack(sel_parts, conf)
    pts_colors = global_colors[sel_vector_bool]
    # reshape to (num * 1000) * 3
    pts_colors = pts_colors.reshape(-1, 3)
    print("pts_colors: ", pts_colors.size())
    pts = np.array(pts.cpu())
    pts_colors = np.array(pts_colors.cpu())

    pts = rotate_axis(pts, "z", 90)
    pts = rotate_axis(pts, "x", 180)
    pts = rotate_axis(pts, "y", 90)

    xml_segments = [xml_head]

    pcl, pcl_colors = standardize_bbox(pts, pts_colors, pts.shape[0])
    for i in range(pcl.shape[0]):
        color = pcl_colors[i]
        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    render_xml = os.path.join(point_save_path, 'render_{}_{}.xml'.format(test_num, gen_idx))
    with open(render_xml, 'w') as f:
        f.write(xml_content)

    mitsuba_render(test_num, gen_idx, point_save_path)
    if del_xml:
        os.remove(render_xml)

##########################  PLEASE CHANGE HERE  #######################################
test_list = [0,1] # test mix set
gen_all_idx_list = [[0,1],[0,1]] # the index of the generated shapes in the mix set
gt_all_idx_list = [[0,1],[0,1]] # the index of the gt shapes in the mix set

cate = "Chair" # the category of the mix set
test_path = "path/to/eval_logs"

need_gt_render = True # whether to render the gt shapes
need_inputs = True # whether to render the input parts
#######################################################################################

dataset_path = "dataset/MixPartRandom/{}/test".format(cate)
gen_main_pose_sel_file_path = "{}/gen_img".format(test_path)
gt_main_pose_sel_file_path = "{}/ref_img".format(test_path)
gen_point_save_path = "{}/pointflow_render/{}/gen".format(test_path, cate)
gt_point_save_path = "{}/pointflow_render/{}/ref".format(test_path, cate)
parts_save_path = "{}/pointflow_render/{}/parts".format(test_path, cate)

if not os.path.exists(gen_point_save_path):
    os.makedirs(gen_point_save_path)

if not os.path.exists(gt_point_save_path):
    os.makedirs(gt_point_save_path)

if not os.path.exists(parts_save_path):
    os.makedirs(parts_save_path)



for global_idx in range(len(test_list)):
    test_number = test_list[global_idx]
    global_colors = get_global_colors(test_number, gt_all_idx_list[global_idx], dataset_path, gt_main_pose_sel_file_path, parts_save_path, cate, need_inputs=need_inputs, del_xml=True)
    # render gen
    for gen_idx in gen_all_idx_list[global_idx]:
        try:
            render_one(test_number, gen_idx, dataset_path, gen_main_pose_sel_file_path, gen_point_save_path, cate, global_colors)
        except:
            print("Error: ", test_number, gen_idx)
    if need_gt_render:
        # render gt
        for gt_idx in gt_all_idx_list[global_idx]:
            render_one(test_number, gt_idx, dataset_path, gt_main_pose_sel_file_path, gt_point_save_path, cate, global_colors)
