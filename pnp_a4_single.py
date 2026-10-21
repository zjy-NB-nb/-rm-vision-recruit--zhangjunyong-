import cv2
import numpy as np

# 相机内参、畸变系数
mtx = np.array([
[1.18973000e+03, 0.00000000e+00, 7.66608359e+02],
[0.00000000e+00, 1.19082670e+03, 5.09063266e+02],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist = np.array([[0.0188816, -0.04617239, -0.00232417, 0.00146018, 0.10082969]])

# A4纸世界坐标(单位mm)，顺序：左下、右下、右上、左上
obj_pts = np.array([
    [0,    0,    0],
    [210,  0,    0],
    [210,  297,  0],
    [0,    297,  0],
], dtype=np.float32)

# 图片路径
img_path = r"C:\Users\67172\Desktop\calib_imgs\test.jpg"
img = cv2.imread(img_path)

if img is None:
    print("❌ 图片读取失败，请检查路径和文件名！")
    exit()

print("✅ 新版本代码已启动（自动筛选面积最大四边形）")
print("✅ 图片尺寸：", img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ========== 自适应二值化（抗光照不均匀，比固定阈值好用） ==========
thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    2
)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

candidates = []
draw_img = img.copy()

# 遍历所有轮廓，收集符合条件的四边形
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 20000:
        continue
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        candidates.append([area, approx])
        cv2.drawContours(draw_img, [approx], -1, (255,0,0), 2) #蓝色画出所有候选四边形

found = False
dist_m = 0

# 挑选面积最大的四边形作为A4
if len(candidates) > 0:
    # 按面积从大到小排序
    candidates.sort(reverse=True, key=lambda x:x[0])
    best_area, best_approx = candidates[0]
    found = True

    img_pts = best_approx.reshape(4, 2).astype(np.float32)
    ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)

    dist_mm = np.linalg.norm(tvec)
    dist_m = dist_mm / 1000

    # 绿色绘制A4框
    cv2.drawContours(draw_img, [best_approx], -1, (0, 255, 0), 3)
    cv2.putText(draw_img, f"Distance: {dist_m:.2f} m", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("result", draw_img)
cv2.imshow("binary_thresh", thresh)

print(f"找到候选四边形数量：{len(candidates)}")
print(f"检测结果 found = {found}")
if found:
    print(f"✅ A4纸距离 = {dist_m:.3f} m")

cv2.waitKey(0)
cv2.destroyAllWindows()