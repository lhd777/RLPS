from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

def baseModel(data_shape):
    input_data = Input(shape=data_shape)

    conv2d_1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_data)
    mpool2d_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2d_1)  # (49,49)

    conv2d_2 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(mpool2d_1)
    mpool2d_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2d_2)  # (24,24)

    conv2d_3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(mpool2d_2)

    conv2d_4 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv2d_3)
    mpool2d_3 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2d_4)  # (12,12)

    conv2d_5 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(mpool2d_3)

    conv2d_6 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv2d_5)

    # output
    flatten = Flatten()(conv2d_6)
    drop1 = Dropout(0.6)(flatten)
    dense1 = Dense(128, activation='relu')(drop1)
    drop2 = Dropout(0.6)(dense1)
    output = Dense(7, activation='softmax')(drop2)

    model = Model(inputs=input_data, outputs=output)
    return model