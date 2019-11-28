import cv2 as cv
#cv.imread函数中的cv.IMREAD_GRAYSCALE参数显示灰度图
src = cv.imread(r"E:\WorkSpace\PycharmProject\OpenCV_Prj\Practice\liwei.JPG")
cv.namedWindow("input",cv.WINDOW_AUTOSIZE)  #自适应决定窗口大小
cv.imshow("input",src)
cv.waitKey(0)  #使窗口一直存在，直到按键
cv.destroyAllWindows()