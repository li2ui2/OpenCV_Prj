import cv2 as cv
import numpy as np

#鼠标响应事件
def onMouse(event, x, y, flags, param):
    #鼠标左键点击事件
    if event == cv.EVENT_LBUTTONDOWN:
        if param['beginPoint'] is None:
            param['beginPoint'] = (x, y)
            param['got_begin_point'] = True

    #鼠标滑动事件
    if event == cv.EVENT_MOUSEMOVE:
        if param['got_begin_point']:
            param['endPoint'] = (x, y)
    #鼠标左键释放事件
    if event == cv.EVENT_LBUTTONUP:
        param['got_line'] = True
        param['endPoint'] = (x, y)
        #打印所画红线在图像中坐标点的位置
        print('The beginPoint is %s, and the endPoint is %s.' % (param['beginPoint'],param['endPoint']))

def main():
    """
    打开摄像头，开始录制。实现功能如下：
    1.录制视频，并保存所录制的视频，实现视频回放功能
    2.显示视频录制的镜像画面
    3.键盘读取功能：按空格键开始录制，再按空格键录视频停止；
                    按键'q',停止录制。
    4.鼠标读取功能：按住鼠标左键，可以再视频中画红色线
    """
    cap = cv.VideoCapture(0)    #读取摄像头
    if not cap.isOpened():
        print('capture device failed to open!')

    #存储视频
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('E:\\WorkSpace\\PycharmProject\\OpenCV_Prj\\Prj1\\output.avi', fourcc, 20.0, (640, 480))

    #定义鼠标点击动作的状态
    mouse_params = {'beginPoint': None, 'endPoint': None,
                    'got_begin_point': False, 'got_line': False}

    title = 'Video Record'
    Flip_title = 'Flip Video Record'
    Line_title = 'DrawLine_Result'
    while(True):
        ret, frame = cap.read()
        #显示原始画面
        cv.imshow(title, frame)
        #显示镜像画面
        gray = cv.cvtColor(frame, 0)
        flip = cv.flip(gray, 1)
        cv.imshow(Flip_title, flip)
        #存储视频文件
        out.write(frame)

        #鼠标读取
        cv.setMouseCallback(Flip_title, onMouse, mouse_params)
        if mouse_params['got_line']:
            cv.namedWindow(Line_title)
            cv.moveWindow(Line_title, 100, 100)
            im_draw = np.copy(flip)
            cv.line(im_draw, mouse_params['beginPoint'], mouse_params['endPoint'], (0, 0, 255), 2)
            cv.imshow(Line_title, im_draw)
            mouse_params['beginPoint'] = None
            mouse_params['got_line'] = False

        #键盘读取
        key = cv.waitKey(1)
        # 点击视频窗口，按space键暂停与播放
        if key & 0xFF == ord(' '):
            cv.waitKey(0)
            if key & 0xFF == ord(' '):
                cv.imshow(Flip_title, flip)
        # 点击视频窗口，按q键退出
        if key & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
