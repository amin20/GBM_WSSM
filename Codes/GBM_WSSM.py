# Import Required Modules


from keras.regularizers import l2
from keras.layers import Reshape, Activation, BatchNormalization, Conv2D, Dropout, Concatenate, Conv2DTranspose, regularizers

##############################################################################

def RELU_BN(
        x
        ): 
    
    return BatchNormalization(axis=-1)(Activation('relu')(x))                                    # Batch Normalization after Activation

def CONV(
        x, 
        nf, 
        sz, 
        wd, 
        p, 
        stride=1
        ):
    
    x = Conv2D(nf, (sz, sz), strides=(stride, stride), padding='same', kernel_initializer='he_uniform',
               kernel_regularizer=l2(wd))(x)
    return Dropout(p)(x) if p else x


def CONV_RELU_BN(
        x, 
        nf, 
        sz=3, 
        wd=0, 
        p=0, 
        stride=1
        ):
    
    return CONV(RELU_BN(x), nf, sz, wd=wd, p=p, stride=stride)


def DENSE_BLOCK(
        n, 
        x, 
        growth_rate, 
        p, 
        wd
        ):
    
    added = []
    for i in range(n):
        b = CONV_RELU_BN(x, growth_rate, p=p, wd=wd)
        x = Concatenate(axis=-1)([x, b])
        added.append(b)
    return x, added


def Transition_DOWN(
        x, 
        p, 
        wd
        ):
    

    return CONV_RELU_BN(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)


def Down_Path(
        x, 
        nb_layers, 
        growth_rate, 
        p, 
        wd
        ):
    
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = DENSE_BLOCK(n, x, growth_rate, p, wd)


        skips.append(x)
        x = Transition_DOWN(x, p=p, wd=wd)
    return skips, added


def Transition_UP(
        added, 
        wd=0
        ):
    
    x = Concatenate(axis=-1)(added)
    _, r, c, ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform',
                 kernel_regularizer = regularizers.l2(wd))(x)


def Up_Path(added, 
            skips, 
            nb_layers, 
            growth_rate, 
            p, 
            wd
            ):
    

    for i, n in enumerate(nb_layers):
        x = Transition_UP(added, wd)

        # concatenate the skip connections
        x = Concatenate(axis=-1)([x, skips[i]])
        x, added = DENSE_BLOCK(n, x, growth_rate, p, wd)
    return x


def reverse(a): return list(reversed(a))



def GBM_WSSM_Model(nb_classes, 
                    img_input, 
                    nb_dense_block=6,
                    growth_rate=16, 
                    nb_filter=48, 
                    nb_layers_per_block=[4, 5, 7, 10, 12, 15], 
                    p=0.2, 
                    wd=1e-4
                    ):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = CONV(img_input, nb_filter, 3, wd, 0)
    skips, added = Down_Path(x, nb_layers, growth_rate, p, wd)
    x = Up_Path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = CONV(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
