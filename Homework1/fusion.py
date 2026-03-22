# Copyright (c) 2018 Andy Zeng

import numpy as np
from skimage import measure
import trimesh

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        # so vol_bnds / voxel_size will tell use the number of voxels along each dimension, 
        # remember use integer 

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self.vol_bnds = vol_bnds
        self.voxel_size = float(voxel_size)

        # this is why we called it TSDF , 5*voxel_size is the bounds 
        self.trunc_margin = 5 * self.voxel_size  # truncation on SDF 


        #######################    Task 2    #######################
        # TODO: build voxel grid coordinates and initiailze volumn attributes
        # Initialize voxel volume
        # first make out the X,Y,Z of the tsdf_vol
        X = self.vol_bnds[0,1] - self.vol_bnds[0,0]
        Y = self.vol_bnds[1,1] - self.vol_bnds[1,0]
        Z = self.vol_bnds[2,1] - self.vol_bnds[2,0]


        # note that we should use ceil to make sure voxels can cover the whole (X,Y,Z) space.
        self.num_X = int(np.ceil(X / self.voxel_size))
        self.num_Y = int(np.ceil(Y / self.voxel_size))
        self.num_Z = int(np.ceil(Z / self.voxel_size))

        self.tsdf_vol = np.zeros((self.num_X, self.num_Y, self.num_Z))
        # for computing the cumulative moving average of weights per voxel
        self.weight_vol = np.zeros((self.num_X, self.num_Y, self.num_Z))
        # Set voxel grid coordinates
        self.vox_coords = np.zeros((self.num_X, self.num_Y, self.num_Z, 3)) # (self.num_X, self.num_Y, self.num_Z, 3)
        for i in range(self.num_X):
            for j in range(self.num_Y):
                for k in range(self.num_Z):
                    # save the world coordinates of the start point of the voxel grid
                    self.vox_coords[i,j,k] = [self.vol_bnds[0,0] + i*self.voxel_size, self.vol_bnds[1,0] + j*self.voxel_size, self.vol_bnds[2,0] + k*self.voxel_size]
        ############################################################

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
            depth_im (ndarray): A depth image of shape (H, W).
            cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign for the current observation. 
        """

        #######################    Task 2    #######################
        # TODO: Convert voxel grid coordinates to pixel coordinates
        # TODO: Eliminate pixels outside depth images
        # TODO: Sample depth values

        # we already have the voxel grid coordinates, which is the world coordinates
        # we should convert that to camera coordinates, and then to pixel coordinates
        # so that we can sample the depth values from the depth image, and compute the TSDF values for each voxel grid.
        
        H,W = depth_im.shape

        # loop through the voxel grid , convert to camera coordinates
        for i in range(self.num_X):
            for j in range(self.num_Y):
                for k in range(self.num_Z):
                    world_pt = np.append(self.vox_coords[i,j,k], 1) # (x,y,z,1)
                    cam_pt = cam_pose @ world_pt # (4,4) @ (4,1) = (4,1)
                    cam_pt = cam_pt[:3] # (x,y,z)
                    # then convert to pixel coordinates
                    pixel_pt = cam_intr @ cam_pt # (3,3) @ (3,1) = (3,1)
                    pixel_pt = pixel_pt / pixel_pt[2] # (u,v,1)
                    u, v = pixel_pt[:2].astype(int) # (u,v)
                    if u < 0 or u >= W or v < 0 or v >= H:
                        continue # outside depth image , just skip it 
                    depth = depth_im[v, u]


        ############################################################
        
        #######################    Task 3    #######################
        # TODO: Compute TSDF for current frame
        ############################################################
                    tsdf = min(1, (depth - cam_pt[2]) / self.trunc_margin) 

        #######################    Task 4    #######################
        # TODO: Integrate TSDF into voxel volume
                    w_old = self.weight_vol[i,j,k]
                    w_new = self.weight_vol[i,j,k] + obs_weight
                    self.tsdf_vol[i,j,k] = (self.tsdf_vol[i,j,k] * w_old + tsdf * obs_weight) / w_new
                    self.weight_vol[i,j,k] = w_new
        ############################################################


def cam_to_world(depth_im, cam_intr, cam_pose, export_pc=False):
    """Get 3D point cloud from depth image and camera pose
    
    Args:
        depth_im (ndarray): Depth image of shape (H, W).
        cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
        export_pc (bool): Whether to export pointcloud to a PLY file.
        
    Returns:
        world_pts (ndarray): The 3D point cloud of shape (N, 3).
    """
    # we first convert depth image to camera coordinates, then transform to world coordinates using cam_pose
    # then we can export the point cloud to a PLY file using trimesh if export_pc is True
    
    #######################    Task 1    #######################
    # TODO: Convert depth image to world coordinates

    # first convert depth image to camera coordinates
    # we can see that the depth image is a .depth.png file, which we can read it as a ndarray 
    # of shape (H, W), each point is a depth value, meters.

    # here we will use the fx,fy,cx,cy from the camera intrinsics matrix 
    """
    K = [fx   0   cx]
        [0   fy   cy]
        [0    0    1]

    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    z = depth_im[v, u]
    """

    # here I do some matrix operations and found some magic things about the transformation 
    # from camera coordinates to world and from world to camera coordinates
    # whose T is something different from each other, which I think is important.
    # that is from world to camera coordinates, the T is -Rt,
    # but from camera to world coordinates, the T is just the t, (t is the position of the camera in the world coordinates)

    H,W = depth_im.shape
    camera_pts = np.zeros((H, W, 3))
    cx = cam_intr[0,2]
    cy = cam_intr[1,2]
    fx = cam_intr[0,0]
    fy = cam_intr[1,1]

    for i in range(H):
        for j in range(W):
            z = depth_im[i,j]
            if z == 0: # invalid depth
                continue # so that the point will be (0,0,0) in camera coordinates ?
            x = z * (j - cx) / fx
            y = z * (i - cy) / fy
            camera_pts[i,j] = [x,y,z] 
    # now we have the point cloud in camera coordinates, and we can transform it to world coordinates
    # we have the camera pose, is it the transformation from camera to world or world to camera ?
    world_pts = np.zeros((H, W, 3)) 

    # precompute the inverse of the camera pose
    cam_pose_inv = np.linalg.inv(cam_pose)
    for i in range(H):
        for j in range(W):
            # notice that the camera pose is a 4x4 matrix
            # so we should fisrt convert the (x,y,z) to (x,y,z,1)
            pt = np.append(camera_pts[i,j], 1) # (x,y,z,1)
            world_pt = cam_pose_inv @ pt # (4,4) @ (4,1) = (4,1)
            world_pts[i,j] = world_pt[:3] # (x,y,z) 
    # now we have the point cloud in world coordinates, but we return a (N,3) shape
    world_pts = world_pts.reshape(-1, 3) # (H*W, 3)

    ############################################################
    
    if export_pc:
        # export the point cloud to a PLY file so that we can visualize it using MeshLab
        pointcloud = trimesh.PointCloud(world_pts)
        pointcloud.export("pointcloud.ply")
    
    return world_pts

class TSDFVolume1:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self.vol_bnds = vol_bnds
        self.voxel_size = float(voxel_size)
        self.trunc_margin = 3 * self.voxel_size

        # 计算体素数量
        self.num_X = int(np.ceil((vol_bnds[0,1] - vol_bnds[0,0]) / self.voxel_size))
        self.num_Y = int(np.ceil((vol_bnds[1,1] - vol_bnds[1,0]) / self.voxel_size))
        self.num_Z = int(np.ceil((vol_bnds[2,1] - vol_bnds[2,0]) / self.voxel_size))

        # 初始化TSDF和权重
        self.tsdf_vol = np.zeros((self.num_X, self.num_Y, self.num_Z), dtype=np.float32)
        self.weight_vol = np.zeros((self.num_X, self.num_Y, self.num_Z), dtype=np.float32)
        self.color_vol = np.zeros((self.num_X, self.num_Y, self.num_Z, 3), dtype=np.float32)
        

        x = self.vol_bnds[0, 0] + np.arange(self.num_X, dtype=np.float32) * self.voxel_size
        y = self.vol_bnds[1, 0] + np.arange(self.num_Y, dtype=np.float32) * self.voxel_size
        z = self.vol_bnds[2, 0] + np.arange(self.num_Z, dtype=np.float32) * self.voxel_size
        
        # 生成网格坐标 (使用ix_生成广播索引)
        self.vox_coords = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(np.float32)
        # self.vox_coords形状: (num_X, num_Y, num_Z, 3)
        self.vox_coords_flat = self.vox_coords.reshape(-1, 3)
        self.vox_coords_homo = np.hstack(
            [self.vox_coords_flat, np.ones((self.vox_coords_flat.shape[0], 1), dtype=np.float32)]
        )
        ############################################################

    def integrate(self, depth_im, color_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume."""
        H, W = depth_im.shape

        world_to_cam = np.linalg.inv(cam_pose).astype(np.float32)
        #world_to_cam = cam_pose.astype(np.float32)
        cam_pts = (world_to_cam @ self.vox_coords_homo.T).T[:, :3]

        cam_z = cam_pts[:, 2]
        valid_z = cam_z > 1e-6
        if not np.any(valid_z):
            return

        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]

        valid_idx = np.nonzero(valid_z)[0]
        cam_pts_valid = cam_pts[valid_idx]
        z_valid = cam_z[valid_idx]

        u = np.round((cam_pts_valid[:, 0] * fx) / z_valid + cx).astype(np.int32)
        v = np.round((cam_pts_valid[:, 1] * fy) / z_valid + cy).astype(np.int32)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            return

        valid_idx = valid_idx[in_img]
        u = u[in_img]
        v = v[in_img]
        cam_z_valid = z_valid[in_img]

        depth_vals = depth_im[v, u].astype(np.float32)
        depth_ok = depth_vals > 0
        if not np.any(depth_ok):
            return

        valid_idx = valid_idx[depth_ok]
        u = u[depth_ok]
        v = v[depth_ok]
        cam_z_valid = cam_z_valid[depth_ok]
        depth_vals = depth_vals[depth_ok]

        # 这里的tsdf，在readme中给出的定义是min(1, (depth - cam_z) / trunc_margin), 但是问题1是
        # 没有限制-1的范围， 第二是当大于1的时候完全没必要取1，直接不算了。
        # 结果却发现还是公式好。

        sdf = depth_vals - cam_z_valid
        valid_tsdf = (sdf >= -self.trunc_margin)
        if not np.any(valid_tsdf):
            return

        idx = valid_idx[valid_tsdf]
        sdf = sdf[valid_tsdf]
        tsdf_vals = np.minimum(1.0, sdf / self.trunc_margin).astype(np.float32)
        #tsdf_vals = np.clip(sdf / self.trunc_margin, -1.0, 1.0).astype(np.float32)

        tsdf_flat = self.tsdf_vol.reshape(-1)
        weight_flat = self.weight_vol.reshape(-1)
        color_flat = self.color_vol.reshape(-1, 3)

        w_old = weight_flat[idx]
        w_new = w_old + obs_weight
        tsdf_flat[idx] = (tsdf_flat[idx] * w_old + tsdf_vals * obs_weight) / w_new
        weight_flat[idx] = w_new

        if color_im is not None:
            near_surface = np.abs(sdf) <= self.trunc_margin
            if np.any(near_surface):
                cidx = idx[near_surface]
                cu = u[valid_tsdf][near_surface]
                cv = v[valid_tsdf][near_surface]

                sampled_rgb = color_im[cv, cu][:, ::-1].astype(np.float32)
                w_old_c = weight_flat[cidx] - obs_weight
                w_new_c = weight_flat[cidx]
                color_flat[cidx] = (
                    color_flat[cidx] * w_old_c[:, None] + sampled_rgb * obs_weight
                ) / w_new_c[:, None]

def cam_to_world(depth_im, cam_intr, cam_pose, export_pc=False):
    """Get 3D point cloud from depth image and camera pose (camera->world)."""
    H, W = depth_im.shape

    valid_v, valid_u = np.nonzero(depth_im > 0)
    if valid_u.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_im[valid_v, valid_u].astype(np.float32)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    x = (valid_u.astype(np.float32) - cx) * z / fx
    y = (valid_v.astype(np.float32) - cy) * z / fy

    cam_pts_homo = np.stack([x, y, z, np.ones_like(z, dtype=np.float32)], axis=1)
    #cam_pose_inv = np.linalg.inv(cam_pose).astype(np.float32)
    #world_pts = (cam_pose_inv @ cam_pts_homo.T).T[:, :3]
    world_pts = (cam_pose.astype(np.float32) @ cam_pts_homo.T).T[:, :3]

    if export_pc:
        trimesh.PointCloud(world_pts).export("pointcloud_2.ply")

    return world_pts