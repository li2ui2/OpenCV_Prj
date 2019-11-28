from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import facenet
import align.detect_face
from os.path import join as pjoin
import copy

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img
def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        # 存储每一类人的文件夹内所有图片
        data[guy] = curr_pics
    return data
# 读取单张图片
def load_and_align_data(image, image_size, margin, gpu_memory_fraction):
    # 读取图片
    img = image
    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        return 0, 0, 0
    else:
        crop = []
        det = bounding_boxes

        det[:, 0] = np.maximum(det[:, 0], 0)
        det[:, 1] = np.maximum(det[:, 1], 0)
        det[:, 2] = np.minimum(det[:, 2], img_size[1])
        det[:, 3] = np.minimum(det[:, 3], img_size[0])

        det = det.astype(int)

        for i in range(len(bounding_boxes)):
            temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
            aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
        crop_image = np.stack(crop)

        return det, crop_image, 1
# 读取文件夹中的所有图片
def load_and_align_data2(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        # img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

        # 根据cropped位置对原图resize，并对新得的aligned进行白化预处理
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def images_emb(images_dir):
    """
    :param images_dir:
    :return: compare_emb：返回某个文件夹中每张图片中人脸框的所对应的向量，
             compare_num：人脸框的数量,
            img_path_set：文件夹中的图片路径列表
    """
    model_dir = './20170512-110547'
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            img_path_set = []
            for index in os.listdir(images_dir):
                single_image = os.path.join(images_dir, index)
                print(single_image)
                print('loading...... :', index)
                img_path_set.append(single_image)

            images = load_and_align_data2(img_path_set, 160, 44, 1.0)
            image = []
            nrof_images = 0
            for i in range(len(images)):
                prewhitened = facenet.prewhiten(images[i])
                image.append(prewhitened)
                nrof_images = nrof_images + 1

            print('开始训练得到的文件夹中每张图片对应的向量')
            images_final =np.stack(image)
            feed_dict = { images_placeholder: images_final, phase_train_placeholder:False }
            compare_emb = sess.run(embeddings, feed_dict=feed_dict)
            compare_num = len(compare_emb)
    return compare_emb, compare_num, img_path_set