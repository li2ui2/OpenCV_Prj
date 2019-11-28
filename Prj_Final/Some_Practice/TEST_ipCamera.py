import cv2

video = "http://admin:admin@192.168.199.135:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址
# 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
capture = cv2.VideoCapture(0)

# 建个窗口并命名
cv2.namedWindow("camera", 1)
num = 0
# 用于循环显示图片，达到显示视频的效果
while True:
    ret, frame = capture.read()

    # 在frame上显示test字符
    image1 = cv2.putText(frame, 'test', (50, 100),
                         cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),
                         thickness=2, lineType=2)

    cv2.imshow('camera', frame)

    # 不加waitkey（） 则会图片显示后窗口直接关掉
    key = cv2.waitKey(3)
    if key == 27:
        # esc键退出
        print("esc break...")
        break

    if key == ord(' '):
        # 保存一张图像
        num = num + 1
        filename = "frames_%s.jpg" % num
        cv2.imwrite(filename, frame)
