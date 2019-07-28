# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 下午3:10
# @Author  : Aries
# @Site    :
# @File    : Cifar10_data.py
# @Software: PyCharm
'''
下载Cifar-10进行训练
'''
import os
import tensorflow as tf

num_class = 10

'''
设定用于训练和评估的样本数
'''
num_examples_pre_epoch_for_train = 50000

num_examples_pre_epoch_for_eval = 10000

'''
定义用于返回Cifar-10数据
'''


class CIFAR10Record(object):
    pass


'''
定义读取Cifar-10数据的函数
'''


def read_cifar10(file_queue):
    result = CIFAR10Record()
    '''
    一幅图标签的长度
    '''
    label_bytes = 1  # 如果是Cifar-100数据集,则此处为2
    '''
    一幅图 高度 & 宽度 & 深度
    '''
    result.height = 32
    result.width = 32
    result.depth = 3  # RGB 3通道

    image_bytes = result.height * result.width * result.depth

    '''
    每个样本都包含一个label和image数据
    '''
    record_bytes = label_bytes + image_bytes
    '''
    创建一个文件读取类,并调用该类的read()函数从文件队中读取文件
    FixedLengthRecordReader类用于读取固定长度字节数信息(针对bin文件)
    
    构造函数原型
    def __init__(self,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               hop_bytes=None,
               name=None,
               encoding=None):
    '''
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)

    '''
    得到的value就是record_bytes长度包含多个label数据和image数据的字符串
    decode_raw()函数可以将字符串解析成图像对应的像素数组
    '''
    record_bytes = tf.decode_raw(value, tf.uint8)

    '''
    将得到的record_bytes数组中的第一个元素类型转换为int32类型
    strided_slice()函数用于对input截取[begin,end]区间的数据
    '''
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    '''
    转化图片数据,除了label之后剩下的便是图片数据
    
    Returns:
    A `Tensor` the same type as `input`.
    '''
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes])
                             , [result.depth, result.height, result.width])
    '''
    将 [depth height width]  转换为 [height,width,depth]的格式
    并将其赋值在result
    '''
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


'''
inputs()函数调用了read_cifar10()函数,可以选择是否对读入的数据进行数据增强处理
'''


def input(data_dir, batch_size, distorted):
    # 使用os的join()函数进行拼接
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    # 创建一个文件队列,并调用read_cifar10()函数读取队列中的文件
    '''
    将文件names传入文件队列创建函数
    '''
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)  # type: CIFAR10Record

    # 使用cast将图片转化为float32的格式
    reshape_image = tf.cast(read_input.uint8image, tf.float32)
    num_examples_pre_epoch = num_examples_pre_epoch_for_train

    ''' 
    对图像进行增强处理
    ----
    所谓增强处理 
    '''
    if distorted != None:
        '''
        step1:将[32,32,3]的图片裁剪成[24,24,3]大小
        '''
        cropped_image = tf.random_crop(reshape_image, [24, 24, 3])
        '''
        step2:随机左右翻转图片
        '''
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        '''
        step3:调整亮度
        def random_brightness(image, max_delta, seed=None):
        ---------------------------------------------------------------------------
         Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
        interval `[-max_delta, max_delta)`.
        '''
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)

        '''
        step4:调整对比度
        '''
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)

        '''
        标准化图片,注意不是归一化
        ----
        per_image_standardization(image)对图片中每一个像素减去平均值并除以像素方差
        '''
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        '''
        设置图片数据以及label的形状
        '''
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes. '
              % min_queue_examples)

        '''
        使用shuffe_batch()函数随机产生一个batch的image和label
        -------------------------------------------------
        def shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                  num_threads=1, seed=None, enqueue_many=False, shapes=None,
                  allow_smaller_final_batch=False, shared_name=None, name=None):
        ------------------------------------------------------------------------
        capacity: An integer. The maximum number of elements in the queue.
        min_after_dequeue: Minimum number elements in the queue after a
        dequeue, used to ensure a level of mixing of elements.
        '''
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size
                                                            , min_after_dequeue=min_queue_examples)
        return images_train, tf.reshape(labels_train, [batch_size])
    else:
        # todo 不对函数进行增强处理的逻辑
        '''
        step1:就是将图片裁剪成  24 * 24的图片
        step2:在此基础上对图片进行标准化处理
        '''
        resized_image = tf.random_crop(reshape_image, [24, 24, 3])
        float_image = tf.image.per_image_standardization(resized_image)

        '''
        设置图片数据以及label的形状
        '''
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes. '
              % min_queue_examples)

        '''
        使用shuffe_batch()函数随机产生一个batch的image和label
        -------------------------------------------------
        def shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                  num_threads=1, seed=None, enqueue_many=False, shapes=None,
                  allow_smaller_final_batch=False, shared_name=None, name=None):
        '''
        images_test, labels_test = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                          num_threads=16,
                                                          capacity=min_queue_examples + 3 * batch_size
                                                          , min_after_dequeue=min_queue_examples)
        return images_test, tf.reshape(labels_test, [batch_size])
