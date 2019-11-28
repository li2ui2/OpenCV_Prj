# 来自opencv\sources\samples/python2/opt_flow.py
import cv2 as cv
import numpy as np

es = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
cap = cv.VideoCapture(r"E:\WorkSpace\PycharmProject\OpenCV_Prj\Prj2\vtest.avi")
frame1 = cap.read()[1]
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# 视频文件输出参数设置
out_fps = 12.0  # 输出文件的帧率
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
sizes = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out1 = cv.VideoWriter(r'E:\WorkSpace\PycharmProject\OpenCV_Prj\Prj2\v1.avi', fourcc, out_fps, sizes)
out2 = cv.VideoWriter(r'E:\WorkSpace\PycharmProject\OpenCV_Prj\Prj2\v2.avi', fourcc, out_fps, sizes)

while True:
    (ret, frame2) = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    draw = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    draw = cv.morphologyEx(draw, cv.MORPH_OPEN, kernel)
    draw = cv.threshold(draw, 25, 255, cv.THRESH_BINARY)[1]

    image, contours, hierarchy = cv.findContours(draw.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv.imshow('frame2', bgr)

    cv.imshow('draw', draw)
    cv.imshow('frame1', frame2)
    out1.write(bgr)
    out2.write(frame2)

    k = cv.waitKey(200) & 0xff
    if k == 27 or k == ord('q'):
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

out1.release()
out2.release()
cap.release()
cv.destroyAllWindows()