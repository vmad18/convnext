from tensorflow import keras
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from models import ConvNeXt
from utils.Training import training_loop, encode_data
import numpy as np


true, false = True, False

size = {
    "ConvNeXt-T": [(96, 192, 384, 768), (3, 3, 9, 3)],
    "ConvNeXt-S": [(96, 192, 384, 768), (3, 3, 27, 3)],
    "ConvNeXt-B": [(128, 256, 512, 1024), (3, 3, 27, 3)],
    "ConvNeXt-L": [(192, 384, 768, 1536), (3, 3, 27, 3)],
    "ConvNeXt-XL": [(256, 512, 1024, 2048), (3, 3, 27, 3)]
}


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    params = size["ConvNext-B"]

    convnext = ConvNeXt(params[0], params[1], 10)
    loss = CategoricalCrossentropy(from_logits=true)
    opt = Adam(learning_rate=1e-4)
    epochs = 85
    batch_size = 64

    # train ConvNeXt model
    training_loop(
        convnext,
        loss,
        opt,
        np.asarray(x_train, dtype="float32"),
        y_train,
        epochs,
        batch_size)

    print(convnext.evaluate(np.asarray(x_test), encode_data(y_test)))


if __name__ == '__main__':
    main()
