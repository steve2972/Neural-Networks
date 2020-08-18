import tensorflow as tf
from tensorflow import nn
from tensorflow import keras
from tensorflow.keras import layers
from math import ceil

class Swish(keras.Model):
    def __init__(self):
        super(Swish, self).__init__()
    
    def call(self, x):
        return keras.activations.swish(x)

def _add_conv(out, in_channels, channels, kernel=1, stride=1, pad=0,
                num_group=1, active=True, relu6=False):
    
    out.append(layers.Conv2D(channels, kernel_size=kernel, strides=stride, use_bias=False, groups=num_group))
    out.append(layers.BatchNormalization())
    if active:
        out.append(layers.ReLU(max_value=6) if relu6 else layers.ReLU())

def _add_conv_swish(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(layers.Conv2D(channels, kernel_size=kernel, strides=stride, use_bias=False))
    out.append(layers.BatchNormalization())
    out.append(Swish())

class SE(keras.Model):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = layers.AveragePooling2D()
        self.fc = keras.Sequential(
            [
                layers.Conv2D(channels // se_ratio, kernel_size=1),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(channels, kernel_size=1, activation='sigmoid')
            ]
        )
    def call(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class LinearBottleneck(keras.Model):
    def __init__(self, in_channels, channels, t, stride, 
                use_se=True, se_ratio=12, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)

        # Use shortcut if stride = 1 and output is bigger than the input
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            # Expansion factor is greater than 1
            dw_channels = in_channels * t
            _add_conv_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels
        
        _add_conv(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels,
                  active=False)
        
        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))
        
        out.append(layers.ReLU(max_value=6))
        _add_conv(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        self.out = keras.Sequential(out)
    
    def call(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x
        
        return out

class ReXNetV1(keras.Model):
    def __init__(self, input_shape=(32,32,3),input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=1000,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 bn_momentum=0.9):
        super(ReXNetV1, self).__init__()

        _layers = [1,2,2,3,3,5]
        strides = [1,2,2,2,1,2]

        # Multiply the layer size by the depth multiplier
        _layers = [ceil(element * depth_mult) for element in _layers]
        strides = sum([[element] + [1] * (_layers[idx] - 1) for idx, element in enumerate(strides)], [])
        # Expansion size is 1 for only the first layer, 6 for everything else
        ts = [1] * _layers[0] + [6] * sum(_layers[1:])
        self.depth = sum(_layers[:]) * 3

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult <1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        features.append(layers.Conv2D(int(round(stem_channel * width_mult)), kernel_size=3, strides=2, use_bias=False, input_shape=input_shape))
        features.append(layers.BatchNormalization())
        features.append(Swish())

        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))
            
        if use_se:
            use_ses = [False] * (_layers[0] + _layers[1]) + [True] * sum(_layers[2:])
        else:
            use_ses = [False] * sum(_layers[:])

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_se=se, se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        _add_conv_swish(features, c, pen_channels)

        features.append(layers.AveragePooling2D())
        self.features = keras.Sequential(features)
        self.output = keras.Sequential(
            layers.Dropout(dropout_ratio),
            layers.Conv2D(classes, 1, use_bias=True)
        )
    
    def call(self, x):
        x = self.features(x)
        x = self.output(x)
        return tf.squeeze(x)

model = ReXNetV1(classes=10)

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)