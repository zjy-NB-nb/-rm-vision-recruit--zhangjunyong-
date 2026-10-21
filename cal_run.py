import cv2
import numpy as np
import os

# ========== 这里改你的棋盘参数 ==========
# 9列 6行内角点（常用A4打印棋盘格）
chess_w = 9
chess_h = 6
# 棋盘每个小方格边长，单位毫米！！
# 你打印A4棋盘，一般是 20mm，看你打印文件标注
square_size = 20

# 构造棋盘世界坐标
objp = np.zeros((chess_w * chess_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)
objp = objp * square_size

obj_points = []
img_points = []

img_folder = "calib_imgs"
img_names = os.listdir(img_folder)

for fname in img_names:
    img_path = os.path.join(img_folder, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片：{fname}，跳过")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 寻找棋盘内角点
    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h))
    if ret:
        print(f"成功识别角点：{fname}")
        obj_points.append(objp)
        # 亚像素细化角点，提高精度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        # 预览角点
        cv2.drawChessboardCorners(img, (chess_w, chess_h), corners2, ret)
        cv2.imshow("corners", img)
        cv2.waitKey(400)
    else:
        print(f"图片 {fname} 找不到全部角点，直接丢弃这张图")

cv2.destroyAllWindows()

# 执行相机标定
if len(obj_points) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print("="*40)
    print("相机内参矩阵 mtx：")
    print(mtx)
    print("\n畸变系数 dist：")
    print(dist)
    print(f"\n重投影误差：{ret:.4f}")
    print("="*40)
else:
    print("没有任何图片识别到棋盘角点，请检查图片！")
