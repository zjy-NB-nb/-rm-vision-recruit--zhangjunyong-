import cv2
import numpy as np

# 棋盘内角点：横向9个，纵向6个
pattern_size = (9, 6)
# 生成棋盘世界坐标
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)

obj_points = []  # 3D世界点
img_points = []  # 图像2D像素点

# USB摄像头编号1
cap = cv2.VideoCapture(1)
print("====标定操作说明====")
print("s 键：保存当前棋盘照片 | q键：结束采集，开始计算标定参数")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 查找棋盘内角点
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        # 画出角点
        cv2.drawChessboardCorners(frame, pattern_size, corners, found)
    cv2.imshow("标定采集窗口", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"✅ 已采集第 {len(img_points)} 张标定图片")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 必须采集大于10张图片才可以标定
if len(img_points) > 10:
    h, w = gray.shape
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print("\n=================标定结果=================")
    print("内参矩阵 K：")
    print(K)
    print("\n畸变系数 dist：")
    print(dist)

    # ==========新增：计算重投影误差==========
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2)/len(img_points_proj)
        total_error += error
    mean_error = total_error / len(obj_points)
    print(f"\n📌 平均重投影误差：{mean_error:.4f} 像素")
    print("✅ 把K和dist复制保存下来，后面PnP测距代码要用！")
else:
    print(f"❌ 图片数量不够，只采集了{len(img_points)}张，至少10张！")
