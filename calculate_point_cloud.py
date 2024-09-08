import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
#Converting Depth to distance
def depth_to_distance(depth_value,depth_scale):
  return -1.0/(depth_value*depth_scale)

model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

capture = cv2.VideoCapture(0)
ret, img = capture.read()
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('Photos/photo.jpg', image)
plt.imshow(image)
plt.show()
input_batch = transform(image).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
output_norm = cv2.normalize(output, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
output_norm = (output_norm * 255).astype(np.uint8)
output_norm = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)
cv2.imshow('Photo', output_norm)
cv2.imwrite('midas/midas0000250.png', output_norm)
color_raw = o3d.io.read_image('Photos/photo.jpg')
depth_raw = o3d.io.read_image("midas/midas0000250.png")
print(depth_raw)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)

# Camera intrinsic parameters built into Open3D for Prime Sense
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# data = np.load('CalibrationMatrix_college_cpt.npz')
# camera_intrinsic = data['CameraMatrix']
# camera_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
#                                                          fx=camera_intrinsic[0][0], fy=camera_intrinsic[1][1], cx=camera_intrinsic[0][2], cy=camera_intrinsic[1][2])

# Create the point cloud from images and camera intrinsic parameters
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# Flip it, otherwise the point cloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
