# Define Residual Unit
def residual_unit(X, filter_num_first, filter_num_final, stride=1, short_cut=True):
    ShortCut = layers.Conv2D(filter_num_final, (1, 1), padding='same', strides=stride)(X)

    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filter_num_first, (1, 1), padding='same')(X)

    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filter_num_first, (3, 3), padding='same', strides=stride)(X)

    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filter_num_final, (1, 1), padding='same')(X)

    if short_cut == True:
        X = layers.Add()([X, ShortCut])

    return X