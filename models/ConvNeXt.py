from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, Input, BatchNormalization, Activation, \
    add, concatenate, LayerNormalization


true, false = True, False


class SpatialDownSample2D(Layer):

    def __init__(self, filters, kernel=2, strides=(2, 2), before: bool = true):
        super(SpatialDownSample2D, self).__init__()

        self.conv = Conv2D(filters, kernel_size=kernel, strides=strides, padding="same")
        self.ln = LayerNormalization(epsilon=1e-6)

        self.before = before

    def call(self, inputs):
        if self.before:
            x = self.ln(inputs)
            x = self.conv(x)
            return x

        x = self.conv(inputs)
        x = self.ln(x)
        return x


class ConvNeXtBlock(Layer):

    def __init__(self, filters):
        super(ConvNeXtBlock, self).__init__()

        f1, f2 = filters

        self.dw = Conv2D(f1, kernel_size=7, padding="same", groups=f1)
        self.ln = LayerNormalization(epsilon=1e-6)
        self.conv1 = Conv2D(f2, kernel_size=1)
        self.conv2 = Conv2D(f1, kernel_size=1)

        self.activation = Activation("gelu")

    def call(self, inputs):
        x = self.dw(inputs)
        x = self.ln(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = add([x, inputs])
        return x


class ConvNeXt(Model):

    def __init__(self, filters, blocks, classes):
        super(ConvNeXt, self).__init__()

        self.stem = SpatialDownSample2D(filters[0], 4, (4, 4), before=false)

        self.spatial_ds = [self.stem]
        self.cblocks = []

        for i in range(len(blocks)):
            blk = [ConvNeXtBlock([filters[i], 4 * filters[i]]) for j in range(blocks[i])]
            self.cblocks.append(blk)

        for i in range(len(filters) - 1):
            self.spatial_ds.append(SpatialDownSample2D(filters[i + 1]))

        self.gap = GlobalAveragePooling2D()
        self.out_ln = LayerNormalization(epsilon=1e-6)

        self.clf = Dense(classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        for i in range(4):
            x = self.spatial_ds[i](x)
            for b in self.cblocks[i]:
                x = b(x)
        x = self.gap(x)
        x = self.out_ln(x)
        x = self.clf(x)
        return x
