# 训练分类器模型
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from os.path import join as pjoin
from sklearn.model_selection import train_test_split  # 制作训练数据和测试数据
from sklearn import metrics
from sklearn.externals import joblib  # 保存模型
from scipy import misc  #图像处理
import tensorflow as tf
import numpy as np
import os
import facenet
import align.detect_face

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

print('创建mtcnn网络，并加载参数')
with tf.Graph().as_default():
    #限制GPU使用率
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0) #使用率为百分百
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

def load_and_align_data(image, image_size, margin, gpu_memory_fraction):
    """
    主要执行resize操作
    :param image:
    :param image_size:
    :param margin:
    :param gpu_memory_fraction:
    :return: det,crop_image,1
    """
    # 读取图片 
    img = image
    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]

    # 检测出人脸框和5个特征点
    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 factor不清楚）
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 如果检测出图片中不存在人脸 则直接返回，return 0（表示不存在人脸，跳过此图）
    if len(bounding_boxes) < 1:
        return 0,0,0
    else:
        crop=[]
        det=bounding_boxes

        det[:,0]=np.maximum(det[:,0], 0)  #取第一列大于0的数据
        det[:,1]=np.maximum(det[:,1], 0)
        det[:,2]=np.minimum(det[:,2], img_size[1])
        det[:,3]=np.minimum(det[:,3], img_size[0])

        # det[:,0]=np.maximum(det[:,0]-margin/2, 0)
        # det[:,1]=np.maximum(det[:,1]-margin/2, 0)
        # det[:,2]=np.minimum(det[:,2]+margin/2, img_size[1])
        # det[:,3]=np.minimum(det[:,3]+margin/2, img_size[0])

        det=det.astype(int)

        for i in range(len(bounding_boxes)):
            temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
            aligned=misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
        crop_image=np.stack(crop)
            
        return det,crop_image,1

    # np.squeeze() 降维，指定第几维，如果那个维度不是1  则无法降维
    # det = np.squeeze(bounding_boxes[0,0:4])
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def load_data(data_dir):
    """
    返回一个data字典，该字典类型 key对应人物分类，
    value为读取的一个人的所有图片 类型为 ndarray
    :param data_dir: 对应项目中的train_dir路径，同一个人的所有图片放在同一个文件夹中
    :return: data
    """

    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)       
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]         

        # 存储每一类人的文件夹内所有图片
        data[guy] = curr_pics      
    return data

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

# facenet模型存放路径
model_dir='./20170512-110547'
#在新的计算图中做运算
with tf.Graph().as_default():
    with tf.Session() as sess:
        # 加载facenet模型
        facenet.load_model(model_dir)

        # 返回给定名称的tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # 从训练数据文件夹中加载图片并剪裁，最后embding，data为dict
        data=load_data('./Data/train_dirs/')
        print('image loaded')
        # keys列表存储图片文件夹类别（几个人）
        keys=[]
        for key in data:
            keys.append(key)
            print('folder:{},image numbers：{}'.format(key,len(data[key])))

        train_x=[]
        train_y=[]
        #name_flags = {}
        # 使用mtcnn模型获取每张图中face的数量以及位置，并将得到的embedding数据存储
        #for index in range(len(keys)):
            #name_flags[keys[index]] = index  #名字对应标签
        for index in range(len(keys)):
            for x in data[keys[index]]:
                det,images_me,i = load_and_align_data(x, 160, 44, 1.0)
                if i:
                    feed_dict = {images_placeholder: images_me, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    print(type(emb))
                    for xx in range(len(emb)):
                        print(type(emb[xx, :]), emb[xx, :].shape)
                        train_x.append(emb[xx, :])
                        train_y.append(index)
            print(len(train_x))

        print('所有人像转换完成，样本数为：{}'.format(len(train_x)))

          
train_x=np.array(train_x)
print(train_x.shape)
train_x=train_x.reshape(-1,128)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)

#分别得到train_x和train_y的训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

def knn_classifier(train_x, train_y):
    """
    KNN Classifier，K最近邻分类器。核心思想是如果一个样本在特征空间中的
    k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
    :param train_x:training data，待训练的数据
    :param train_y:target value，待训练数据一一对应的标签
    :return: model
    """
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

def svm_classifier(train_x, train_y):
    from sklearn import svm
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model
# classifiers = knn_classifier
#
# model = classifiers(X_train,y_train)
# predict = model.predict(X_test)
# SCORE_ = model.predict_proba(X_test)
# print(SCORE_)
# # 计算预测标签的精度
# accuracy = metrics.accuracy_score(y_test, predict)
# print ('accuracy: %.2f%%' % (100 * accuracy)  )
classifiers = svm_classifier

model = classifiers(X_train,y_train)

predict = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predict)
print ('accuracy: %.6f%%' % (100 * accuracy))
#save model
joblib.dump(model, './models/svm_classifier42.model')

# model = joblib.load('./models/svm_classifier42.model')
# predict = model.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, predict)
# print ('accuracy: %.2f%%' % (100 * accuracy)  )
