import ultralytics
from PIL import Image
import argparse
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import open3d as o3d
import pyrealsense2 as rs

def set_cam():
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx,
                                                                 intr.ppy)

    return align, pipeline, pinhole_camera_intrinsic


def get_seg(model: ultralytics.YOLO, img: Image.Image, depth: Image.Image, num: int, threshold: float, isDebug=False):
    result = model.predict(source=img, retina_masks=True)
    # result = model(img, conf=threshold)
    mask = result[0].masks.cpu().numpy().data
    img_mask = F.to_pil_image(mask[0])
    seg_index = np.where(np.asarray(img_mask) == 0)
    # processed_img = np.asarray(copy.deepcopy(img))
    processed_img = copy.deepcopy(img)
    # processed_depth = np.asarray(copy.deepcopy(depth))
    processed_depth = copy.deepcopy(depth)
    # print(seg_index)
    for i in range(len(seg_index[0])):
        processed_img[seg_index[0][i], seg_index[1][i]] = 0
        processed_depth[seg_index[0][i], seg_index[1][i]] = 0
    if isDebug:
        plt.imshow(np.asarray(processed_img))
        plt.show()
        plt.imshow(np.asarray(processed_depth))
        plt.show()
    return np.asarray(processed_img), np.asarray(processed_depth)


def get_pcd(color_image, depth_image, camera_intrinsics):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.clear()
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(color_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pointcloud += pcd
    return pointcloud


def get_bBox(pcd):
    filtered_pcd = pcd.remove_radius_outlier(15, 0.05)[0]
    filtered_pcd = filtered_pcd.remove_statistical_outlier(15, 0.01)[0]
    # filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5))
    # planes = filtered_pcd.detect_planar_patches(search_param = o3d.geometry.KDTreeSearchParamKNN(knn=5))
    geometries = []
    # for obox in planes:
    # mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
    # geometries.append(obox)
    # geometries.append(mesh)
    geometries.append(pcd)
    geometries.append(filtered_pcd.get_minimal_oriented_bounding_box())
    # geometries.append(tmp)
    geometries[1].color = [255, 0, 0]
    # o3d.visualization.draw_geometries_with_editing(geometries)
    return geometries


def get_center_and_angle(bBox):
    # ATTENTION!! Coordinates IS (x, z, y)
    np_points = np.asarray(bBox.get_box_points())
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(np_points[:, 0], np_points[:, 1], np_points[:, 2])
    ax.set_aspect('equal', adjustable='box')
    # plt.show()
    points = list(map(list, np.asarray(bBox.get_box_points())))
    points.sort(key=lambda point: sum([cor ** 2 for cor in point]))  # sort distance
    front_points = points[:4]
    front_points.sort(key=lambda point: point[1])  # sort z
    front_base_line = front_points[:2]
    front_base_line.sort(key=lambda point: point[0])  # sort x
    theta = np.arctan2(front_base_line[1][2] - front_base_line[0][2], front_base_line[1][0] - front_base_line[0][0])
    # print(points)
    pos = [(front_base_line[1][0] + front_base_line[0][0]) / 2, (front_base_line[1][2] + front_base_line[0][2]) / 2]
    return pos, theta


def main():
    align, pipeline, pinhole_camera_intrinsic = set_cam()
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()

    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    model = ultralytics.YOLO("best_7.pt")
    color, depth = get_seg(model, color_image, depth_image, 1, 0.25)
    point_cloud = get_pcd(color, depth.astype(np.float32), pinhole_camera_intrinsic)
    filtered_pcd, bBox = get_bBox(point_cloud)
    pos, angle = get_center_and_angle(bBox)
    print(pos, "\n", angle * 180, "in degree")


if __name__ == '__main__':
    main()