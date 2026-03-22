"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import os
import cv2
import numpy as np
from tqdm import tqdm 
from skimage import measure
import trimesh

import fusion

if __name__ == "__main__":

    print("Estimating voxel volume bounds...")
    n_imgs = 1000    # the total nums of images that we will do fusion
    
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')  # we use one camera so the intrinsics matrix is just one 
    
    # vol_bnds[:, 0]: minimum bounds of the voxel volume along x, y, z
    # vol_bnds[:, 1]: maximum bounds of the voxel volume along x, y, z
    vol_bnds = np.array([[np.inf, -np.inf],
                         [np.inf, -np.inf],
                         [np.inf, -np.inf]], dtype=np.float32)
    
    for i in tqdm(range(n_imgs)):
        
        # Read depth image 
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i), -1).astype(float)
        
        # depth is saved in 16-bit PNG in millimeters
        # here convert it to meters by dividing by 1000
        depth_im /= 1000.  
        # set invalid depth to 0 (specific to 7-scenes dataset)
        # ? 65.535 meters ? 
        # because we use 16-bit to save depth, the maximum value is 2^16 - 1 65535
        # 65535 / 1000 = 65.535 meters, so we set the max depth value to 0
        depth_im[depth_im == 65.535] = 0  

        # Read camera pose, a 4x4 rigid transformation matrix
        # that transforms points from the camera coordinate system to the world coordinate system 

        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  
        
        #######################    Task 1    #######################
        #  Convert depth image to world coordinates
        view_frust_pts = fusion.cam_to_world(
            depth_im, cam_intr, cam_pose,
            export_pc=(i == 0)  # export pointcloud only for the first frame
        )

        # TODO: Update voxel volume bounds `vol_bnds`
        # notice that we now have the point cloud (view_frust_pts) and we should update the vol_bnds based the maximum and minimum value of the point clouds 

        # np.min(view_frust_pts, axis=0) means find the min value through the first dimension N
        # so that we will get a (x_min,y_min,z_min) for the current view clouds.

        # np.minimum used to compare two arrays and return the element-wise minimum value.
        # np.maximum same ...

        if view_frust_pts.shape[0] == 0:
            continue
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.min(view_frust_pts, axis=0))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.max(view_frust_pts, axis=0))

    if not np.all(np.isfinite(vol_bnds)):
        raise RuntimeError("Failed to estimate valid volume bounds from depth frames.")

    print("Volume bounds:", vol_bnds)
    print("Volume extent (m):", (vol_bnds[:, 1] - vol_bnds[:, 0]))

    # directly visualize the cloud of all the points we have seen 
    # after test, it turns out that the point cloud is too large and issue with the visualization.
    # all_points = np.concatenate(all_points, axis=0)
    # all_points_cloud = trimesh.PointCloud(all_points)
    # all_points_cloud.export("all_points_cloud.ply")

    # Initialize TSDF voxel volume
    print("Initializing voxel volume...")

    # create the TSDFVolume object with the given volume bounds and voxel size (2cm)
    tsdf_vol = fusion.TSDFVolume1(vol_bnds, voxel_size=0.01)

    # Loop through images and fuse them together
    t0_elapse = time.time()

    for i in tqdm(range(n_imgs)):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

        color_im = cv2.imread("data/frame-%06d.color.jpg"%(i))  # we can also read the color image if we want to do color fusion in the future

        # Integrate observation into voxel volume
        tsdf_vol.integrate(depth_im, color_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    #######################    Task 4    #######################
    # TODO: Extract mesh from voxel volume, save and visualize it
    ############################################################

    # we can use marching cubes to extract the mesh from the TSDF voxel volume, and then save it as a PLY 
    # use package to do marching cubes 
    tsdf_for_mc = tsdf_vol.tsdf_vol.copy()
    observed_mask = tsdf_vol.weight_vol > 0
    tsdf_for_mc[~observed_mask] = 1.0
    verts, faces, normals, _ = measure.marching_cubes(
        tsdf_for_mc,
        level=0,
        mask=observed_mask
    )
    # because our marching cubes make the points cloud based the volume grid, which is not in the world coordinates
    verts_world = verts * tsdf_vol.voxel_size + tsdf_vol.vol_bnds[:, 0]
    # this position may not be integer
    vert_idx = np.round(verts).astype(np.int32)
    vert_idx[:, 0] = np.clip(vert_idx[:, 0], 0, tsdf_vol.num_X - 1)
    vert_idx[:, 1] = np.clip(vert_idx[:, 1], 0, tsdf_vol.num_Y - 1)
    vert_idx[:, 2] = np.clip(vert_idx[:, 2], 0, tsdf_vol.num_Z - 1)

    vertex_colors = tsdf_vol.color_vol[
        vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]
    ].astype(np.uint8)

    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        vertex_normals=normals,
        vertex_colors=vertex_colors,
        process=False
    )
    mesh.export("tsdf_mesh_2.ply")

