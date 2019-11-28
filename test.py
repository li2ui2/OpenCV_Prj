#人脸识别测试
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
import facenet
from sklearn.externals import joblib
import temp_face_api as tfi
from PIL import Image,ImageFont,ImageDraw

model_dir='./20170512-110547'
compare_emb_list = []
liwei_images_dir = 'E:\\WorkSpace\\PycharmProject\\face_recognition_system\\Data\\img_library'
compare_emb, compare_num, img_path_set = tfi.images_emb(liwei_images_dir)

with tf.Graph().as_default():
    with tf.Session() as sess:  
        # 加载模型
        facenet.load_model(model_dir)

        print('建立facenet embedding模型')
        # 返回给定名称的tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
        model = joblib.load('./models/svm_classifier43.model')

        # 开启ip摄像头
        #video = "http://admin:admin@192.168.199.135:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址
        # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
        capture = cv2.VideoCapture(0)
        cv2.namedWindow("camera", 1)
        c = 0
        num = 0
        frame_interval = 3  # frame intervals
        while True:
            ret, frame = capture.read()
            timeF = frame_interval
            # print(shape(frame))
            detect_face = []

            if (c % timeF == 0):
                find_results = []
                # cv2.imshow("camera",frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if gray.ndim == 2:
                    img = tfi.to_rgb(gray)
                det, crop_image, j_ = tfi.load_and_align_data(img, 160, 44, 1.0)
                if j_:
                    feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    for xx in range(len(emb)):
                        print(type(emb[xx, :]), emb[xx, :].shape)
                        detect_face.append(emb[xx, :])
                    detect_face = np.array(detect_face)
                    detect_face = detect_face.reshape(-1, 128)
                    #print('facenet embedding模型建立完毕')

                    predict = model.predict(detect_face)
                    print(predict)
                    result = []
                    dist_list = []
                    min_dist_index = 0
                    for i in range(len(predict)):
                        if predict[i] == 0:
                            result.append('Fansheng')
                        elif predict[i] == 1:
                            result.append('Liwei')
                        elif predict[i] == 2:
                            result.append('others')
                        elif predict[i] == 3:
                            result.append('s1')
                    #print('实时检测到的人脸框的标签预测完毕')

                    #对比库中图片，寻找最相似图片对应的标签
                    for j in range(compare_num):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], compare_emb[j, :]))))
                        dist_list.append(dist)
                    min_dist_index = dist_list.index(min(dist_list))
                    print(min(dist_list))
                    #print('寻找库中最相似图片完毕')

                    # 绘制矩形框并标注文字
                    max_area = 0
                    max_area_point = []
                    #找最大矩形框
                    count = 0
                    for rec_position in range(len(det)):
                        count = rec_position
                        a = abs(det[rec_position, 0]-det[rec_position, 2])
                        b = abs(det[rec_position, 1]-det[rec_position, 3])
                        s = a*b
                        if s > max_area:
                            max_area = s
                            max_area_point = [det[rec_position, 0],det[rec_position, 1],det[rec_position, 2],det[rec_position, 3]]
                    #绘制矩形框
                    cv2.rectangle(frame, (max_area_point[0], max_area_point[1]),
                                  (max_area_point[2], max_area_point[3]), (0, 255, 0), 2, 8, 0)
                    try:
                        #在左上角输出对应信息
                        cv2.putText(
                            frame,
                            result[count],
                            (max_area_point[0], max_area_point[1]),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            0.8,
                            (0, 0, 255),
                            thickness=2,
                            lineType=2)
                    except:
                        print('未检测到这个人')

                    # 缩放显示库中最相似图片
                    resemble_most_img = Image.open(img_path_set[min_dist_index])
                    resemble_most_img.thumbnail((200, 200))
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image.paste(resemble_most_img,(0,0))
                    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                cv2.imshow('camera', frame)
                # cv2.waitKey(0)
            c += 1
            key = cv2.waitKey(3)
            if key == 27:
                # esc键退出
                print("esc break...")
                break
            if key == ord('s'):
                # 按s键，保存一张图像
                num = num + 1
                filename = "frames_%s.jpg" % num
                cv2.imwrite(filename, frame)
            # if key == ord(' '):
            #     cv2.imshow('camera', frame)
        capture.release()
        cv2.destroyWindow("camera")

