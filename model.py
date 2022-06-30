from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model

def conv_block(inputs, num_filters):
    """
    :param inputs: форма изображения
    :param num_filters: Количество каналов во входном изображении
    :return: CONV слой произведёт набор выходных нейронов, которые будут связаны с локальной областью входного исходного изображения
    """
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    """
    :param inputs: форма изображения
    :param num_filters: Количество каналов во входном изображении
    :return: x - (Conv2d) слой и p - (MaxPooling2d) Тензор ранга 4, представляющий максимальные объединенные значения.
    """
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """
    :param inputs: Conv2D слой
    :param skip_features: пропускной слой conv
    :param num_filters: Количество каналов во входном изображении
    :return: Тензор ранга 4, представляющий активацию (conv2DTranspose (входные данные, ядро) + смещение).
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    """
    :param input_shape: форма изображения
    :return: построенную модель сегментации
    """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bridge """
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    """ Model """
    model = Model(inputs, outputs)
    return model