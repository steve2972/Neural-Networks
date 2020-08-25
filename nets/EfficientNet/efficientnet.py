import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import copy

def EfficientNet(width_coefficient, depth_coefficient, default_size, depth_divisor=8,
                    dropout_rate=0.2,activation='swish', blocks_args='default', 
                    model_name='efficientnet', input_shape=None, classes=1000,
                    drop_connect_rate=0.2,):
    def round_filters(filters, divisor=depth_divisor):
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor/2) //divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
    def round_repeats(repeats):
        # Rounds number of repeats based on depth_multiplier
        return int(math.ceil(depth_coefficient * repeats))

    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS

    if input_shape == None:
        input_shape = (224, 224, 3)

    # Build stem
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(name='stem_pad')(img_input)
    x = layers.Conv2D(32, 3, strides=2, padding='valid', use_bias=False,
                        name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build Conv Blocks

    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name='block{}{}_'.format(i+1, chr(j+97)),
                **args
            )
    #Build Top
    x = layers.Conv2D(
        round_filters(1280),
        1,
        padding='same',
        use_bias=False,
        name='top_conv'
    )(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = layers.Dense(
        classes,
        activation='softmax',
        name='predictions'
    )(x)

    model = keras.Model(img_input, x, name=model_name)
    return model



def block(inputs, activation='swish', drop_rate=0., filters_in=32, filters_out=16,
            kernel_size=3, strides=1, expand_ratio=1, se_ratio=0., id_skip=True,
            name=''):

    # An inverted residual block

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False, name=name+'expand_conv')(inputs)
        x = layers.BatchNormalization(name=name+'expand_bn')(x)
        x = layers.Activation(activation, name=name+'expand_activation')(x)

    else:
        x = inputs
    
    # Depthwise Convolution
    if strides==2:
        x = layers.ZeroPadding2D()(x)
        conv_pad='valid'
    else:
        conv_pad='valid'

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad,
                                use_bias=False, name=name+'dwconv')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name+'se_squeeze')(x)
        se = layers.Reshape((1,1,filters), name=name+'reshape')(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding='same',
            activation=activation,
            name=name+'se_reduce'
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding='same',
            activation='sigmoid',
            name=name+'se_expand'
        )(se)
        x = layers.multiply([x, se], name=name+'se_excite')
    
    # Output phase
    x = layers.Conv2D(filters_out, 1, padding='same', use_bias=False, name=name+'projection')(x)
    if id_skip and strides==1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name+'drop')(x)
        x = layers.add([x, inputs], name=name+'add')
    return x

def EfficientNetB0(input_shape=None,
                   classes=1000,
                   **kwargs):
  return EfficientNet(
      1.0,
      1.0,
      224,
      model_name='efficientnetb0',
      input_shape=input_shape,
      classes=classes,
      **kwargs)

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]