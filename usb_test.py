import cv2
# 0是摄像头编号，如果打不开换成1
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 打不开摄像头，把代码里的0改成1再试")
else:
    print("✅ 摄像头打开成功")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("USB摄像头画面", frame)
    # 按键盘 q 关闭窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
