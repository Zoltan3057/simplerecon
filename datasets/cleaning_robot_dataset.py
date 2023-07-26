import functools
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from utils.geometry_utils import rotx
from utils.generic_utils import read_image_file
import PIL.Image as pil
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

class CleaningRobotDataset(GenericMVSDataset):
    """ 
    Reads a CleaningRobot scan folder.
    
    self.capture_metadata is a dictionary indexed with a scan's id and is 
    populated with a scan's frame information when a frame is loaded for the 
    first time from that scan.

    This class does not load depth, instead returns dummy data.

    Inherits from GenericMVSDataset and implements missing methods.
    """
    def __init__(
            self, 
            dataset_path,
            split,
            mv_tuple_file_suffix,
            include_full_res_depth=False,
            limit_to_time_id=None,
            num_images_in_tuple=None,
            color_transform=transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            tuple_info_file_location=None,
            image_height=384,
            image_width=512,
            high_res_image_width=640,
            high_res_image_height=480,
            image_depth_ratio=2,
            shuffle_tuple=False,
            include_full_depth_K=False,
            include_high_res_color=False,
            pass_frame_id=False,
            skip_frames=None,
            skip_to_frame=None,
            verbose_init=True,
            native_depth_width=256,
            native_depth_height=192,
        ):
        super().__init__(dataset_path=dataset_path,
                split=split, mv_tuple_file_suffix=mv_tuple_file_suffix, 
                include_full_res_depth=include_full_res_depth, 
                limit_to_time_id=limit_to_time_id,
                num_images_in_tuple=num_images_in_tuple, 
                color_transform=color_transform, 
                tuple_info_file_location=tuple_info_file_location, 
                image_height=image_height, image_width=image_width, 
                high_res_image_width=high_res_image_width, 
                high_res_image_height=high_res_image_height, 
                image_depth_ratio=image_depth_ratio, shuffle_tuple=shuffle_tuple, 
                include_full_depth_K=include_full_depth_K, 
                include_high_res_color=include_high_res_color, 
                pass_frame_id=pass_frame_id, skip_frames=skip_frames, 
                skip_to_frame=skip_to_frame, verbose_init=verbose_init,
                native_depth_width=native_depth_width,
                native_depth_height=native_depth_height,
            )

        self.capture_metadata = {}
        self.image_resampling_mode=pil.BICUBIC
        self.load_metadata_flag = False
        self.load_capture_metadata(time_id)


    def get_frame_id_string(self, frame_id):
        """ Returns an id string for this frame_id that's unique to this frame
            within the scan.

            This string is what this dataset uses as a reference to store files 
            on disk.
        """
        return frame_id

    def get_valid_frame_ids(self, split, time_id, store_computed=True):
        """ 根据pose的txt文件读取frame_id.
        """
        return self.valid_ids
        
    def load_pose(self, time_id, frame_id):
        """ Loads a frame's pose file.

            Args: 
                time_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                world_T_cam (numpy array): matrix for transforming from the 
                    camera to the world (pose).
                cam_T_world (numpy array): matrix for transforming from the 
                    world to the camera (extrinsics).

        """

        frame_pose = self.capture_metadata[time_id][int(frame_id)]
        quat = frame_pose[:4]
        translation = frame_pose[4:]
        rotation_matrix = R.from_quat(quat).as_matrix()
        world_T_body = np.identity(4)
        world_T_body[:3, :3] = rotation_matrix
        world_T_body[:3, 3] = translation

        T_cb_GLOBAL = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0.06],
            [1, 0, 0, -0.16583],
            [0, 0, 0, 1]
        ])

        world_T_cam = T_cb_GLOBAL * world_T_body.inverse();
        cam_T_world = np.linalg.inv(world_T_cam)
        return world_T_cam, cam_T_world

    def load_intrinsics(self, time_id, frame_id, flip=None):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.

            Args: 
                time_id: the scan this file belongs to.
                frame_id: id for the frame. Not needed for ScanNet as images 
                share intrinsics across a scene.
                flip: unused

            Returns:
                output_dict: A dict with
                    - K_s{i}_b44 (intrinsics) and invK_s{i}_b44 
                    (backprojection) where i in [0,1,2,3,4]. i=0 provides
                    intrinsics at the scale for depth_b1hw. 
                    - K_full_depth_b44 and invK_full_depth_b44 provides 
                    intrinsics for the maximum available depth resolution.
                    Only provided when include_full_res_depth is true. 
            
        """
        output_dict = {}
        intrinsic_txt_path = os.path.join(self.dataset_path, time_id, "camera_intrinsics.json")
        with open(intrinsic_txt_path) as f:
            intrinsic_data = json.load(f)

        image_width = 800
        image_height = 600
        
        fx = intrinsic_data['intrinsic_800_600']['fx']
        fy = intrinsic_data['intrinsic_800_600']['fy']
        cx = intrinsic_data['intrinsic_800_600']['cx']
        cy = intrinsic_data['intrinsic_800_600']['cy']

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(cx)
        K[1, 2] = float(cy)

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:   
            full_K = K.clone()

            full_K[0] *= (self.native_depth_width/image_width) 
            full_K[1] *= (self.native_depth_height/image_height) 

            output_dict[f"K_full_depth_b44"] = full_K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(full_K)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= (self.depth_width/image_width) 
        K[1] *= (self.depth_height/image_height)

        # Get the intrinsics of all the scales
        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_capture_metadata(self, time_id):
        """读取 txt pose 文件
        """
        if self.load_metadata_flag:
            return
        # 前四个是四元素，后面三个是位置
        posetxt_path = os.path.join(self.dataset_path, time_id, "camera_pose_indexed.txt")
        posetxt_data = np.loadtxt(posetxt_path, delimiter=',')
        valid_ids = []
        valid_ids_pose = []
        metadata = dict()
        for line in posetxt_data:
            values = line.split(',')
            valid_ids.append(int(values[0]))
            valid_ids_pose.append([float(value) for value in values[1:]])
            metadata[int(values[0])] = [float(value) for value in values[1:]]
        self.capture_metadata[time_id] = metadata
        self.valid_ids = np.array(valid_ids)
        self.load_metadata_flag = True

    def get_cached_depth_filepath(self, time_id, frame_id):
        return None

    def get_cached_confidence_filepath(self, time_id, frame_id):
        return None

    def get_full_res_depth_filepath(self, time_id, frame_id):
        return None

    def get_full_res_confidence_filepath(self, time_id, frame_id):
        return None

    def load_full_res_depth_and_mask(self, time_id, frame_id):
        return None

    def get_color_filepath(self, time_id, frame_id):
        """ returns the filepath for a frame's color file at the dataset's 
            configured RGB resolution.

            Args: 
                time_id: timestamp.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the size 
                required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        scene_path = os.path.join(self.dataset_path, time_id)        
        # instead return the default image
        return os.path.join(scene_path, f"{frame_id}.jpg")