import ultralytics
from PIL import Image
import argparse
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import open3d as o3d
import os
from timeit import default_timer as timer

def get_seg(model:ultralytics.YOLO, img:Image.Image, depth:Image.Image, num:int, threshold:float, isDebug = False):
    result = model.predict(source = img, retina_masks = True, device="cpu")
    # result = model(img, conf=threshold)
    mask = result[0].masks.cpu().numpy().data
    img_mask = F.to_pil_image(mask[0])
    seg_index = np.where(np.asarray(img_mask) == 0)
    processed_img = np.asarray(copy.deepcopy(img))
    processed_img = copy.deepcopy(processed_img)
    processed_depth = np.asarray(copy.deepcopy(depth))
    processed_depth = copy.deepcopy(processed_depth)
    # print(seg_index)
    for i in range(len(seg_index[0])):
        processed_img[seg_index[0][i], seg_index[1][i]] = 0
        processed_depth[seg_index[0][i], seg_index[1][i]] = 0
    if isDebug:
        plt.axis("off")
        plt.imshow(np.asarray(processed_img))
        plt.show()
        plt.axis("off")
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
    # filtered_pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamKNN(knn=5))
    # planes = filtered_pcd.detect_planar_patches(search_param = o3d.geometry.KDTreeSearchParamKNN(knn=5))
    geometries = []
    # for obox in planes:
        # mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        # geometries.append(obox)
        # geometries.append(mesh)
    geometries.append(pcd)
    geometries.append(filtered_pcd.get_minimal_oriented_bounding_box(robust=True))
    # geometries.append(tmp)
    geometries[1].color=[255, 0, 0]
    # o3d.visualization.draw_geometries(geometries)
    return geometries


def get_center_and_angle(bBox):
    # ATTENTION!! Coordinates IS (x, z, y)
    np_points = np.asarray(bBox.get_box_points())
    # fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    # ax.scatter(np_points[:,0],np_points[:,1], np_points[:,2])
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
    points = list(map(list, np.asarray(bBox.get_box_points())))
    points.sort(key=lambda point: sum([cor ** 2 for cor in point]))#sort distance
    front_points = points[:4]
    front_points.sort(key=lambda point: point[1])#sort z
    front_base_line = front_points[:2]
    front_base_line.sort(key=lambda point : point[0])#sort x
    theta = np.arctan2(front_base_line[1][2] - front_base_line[0][2], front_base_line[1][0] - front_base_line[0][0])
    # print(points)
    pos = [(front_base_line[1][0] + front_base_line[0][0]) / 2, (front_base_line[1][2] + front_base_line[0][2]) / 2]
    return pos, theta

def get_num_lst(folder_path):
    num_lst = []
    for _, _, files in os.walk(folder_path):
        for name in files:
            num_lst.append(name[4:-4])
    return num_lst

def get_one(path, filename, model, intrinsic, out):
    print(filename)
    start = timer()
    input_image = Image.open(path + "/images/rgb_" + filename + ".png")
    # input_pcd = o3d.io.read_point_cloud(args.data_path.rstrip('/\\') + "/point_clouds/point_cloud" + filename + ".pcd")
    input_depth = Image.open(path + "/depths/depth_" + filename + ".png")
    # print(np.asarray(input_pcd.points))
    
    # o3d.visualization.draw_geometries([input_pcd])
    
    color, depth = get_seg(model, input_image, input_depth, 1, 0.25)
    
    
    # print(type(depth[1,1]))
    point_cloud = get_pcd(color, depth.astype(np.float32), intrinsic)
    filtered_pcd, bBox = get_bBox(point_cloud)
    # get_norm(input_pcd)
    # o3d.visualization.draw_geometries([point_cloud])
    # o3d.io.write_point_cloud('./p1.pcd', point_cloud)
    # tpcd = o3d.t.io.read_point_cloud("./p1.pcd")
    # o3d.t.io.write_point_cloud("./p1.ply", tpcd, write_ascii=True, compressed=False)
    # print(bBox)
    pos, angle = get_center_and_angle(bBox)
    end = timer()
    out.write("{0:.6f},{1:.6f},{2:.6f},{3:.8f}\n".format(pos[0], pos[1], angle * 180 / np.pi, end - start))
    # print(pos, "\n", angle * 180 / np.pi, "in degree")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', default="best6.pt", help="Segmentation Models")
    parser.add_argument('data_path', nargs='?', default="data/", help="Path to where the datas, including pcd, image and depth, are")
    parser.add_argument("camera", nargs='?', default="Camera.txt", help="Txt file where the camera intrinsics stores")
    parser.add_argument('out_dir', nargs='?', default="output/", help="Path to a folder to store processed pcd and images")
    # sample command: python seg_cut.py best_7.pt ./data/data_001 ./Camera.txt
    # What should be noticed is "Camera.txt" should only contain 1 row and each number in it represents width, height, fx, fy, ppx, ppy respectively
    args = parser.parse_args()
    model = ultralytics.YOLO(args.model_path)
    file_camera = open(args.camera, "r")
    width, height, fx, fy, ppx, ppy = map(eval, file_camera.readline().split())
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)
    path = args.data_path.rstrip('/\\')
    num_lst = get_num_lst(path + "/images/")
    out_file = open(path + "_time_result.csv", "w")
    # print(num_lst)
    for num in num_lst[10:]:
        get_one(path, num, model, pinhole_camera_intrinsic, out_file)
    out_file.close()

if __name__ == '__main__':
    main()
