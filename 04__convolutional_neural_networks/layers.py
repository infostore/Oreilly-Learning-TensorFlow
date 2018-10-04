import tensorflow as tf


def weight_variable(shape):
    """
    완전 연결 계층이나 합성곱 계층의 가중치 지정
    :param shape:
    :return:
    """
    # 절단정규분포 : 평균으로부터 표준편차를 기준으로 크거나 작은 값들을 제거한 것
    # 일반적으로 사용되는 방식
    # 학습에 무작위 값을 사용하면 학습된 특징 간의 대칭성을 무너뜨려 모델이 다양하고 풍부한 표현을 학습할 수 있게 된다.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    완전 연결 계층이나 합성곱 계층의 편향값 정의
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    합성곱 정의
    :param x: 입력 이미지 또는 이전 합성곱 계층들에서 얻어진 특징 맵
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    최대값 풀링을 통해 높이와 넓이 차원을 각각 절반으로 줄여 전체적으로 특징 맵의 크기를 1/4로 줄인다.

    합성곱 계층 다음에 풀링(특징 맵 내에서 어떤 지역적 집계 함수를 사용해 데이터의 크기를 줄이는 것)
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b
