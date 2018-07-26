import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

def load_dataset(batch_size, data_shape=(3, 256, 256)):
    training_set = mx.image.ImageDetIter(
        batch_size, data_shape,
        path_imgrec = "data/train.rec",
        path_imgidx = "data/train.idx",
        shuffle = True,
        rand_crop = 1,
        min_object_covered = 0.95,
        max_attempts = 200
    )
    validating_set = mx.image.ImageDetIter(
        batch_size, data_shape,
        path_imgrec = "data/val.rec",
        shuffle = False
    )
    return training_set, validating_set

def color_normalize(x):
    mean = mx.nd.array([[[0.485]], [[0.456]], [[0.406]]], x.context)
    std = mx.nd.array([[[0.229]], [[0.224]], [[0.225]]], x.context)
    return ((x / 255) - mean) / std


if __name__ == "__main__":
    training_set, _ = load_dataset(4)
    batch = training_set.next()
    print(color_normalize(batch.data[0]))
    img = batch.data[0][0].asnumpy()
    img = img.transpose((1, 2, 0)).astype(np.uint8)
    plt.imshow(img)
    for label in batch.label[0][0].asnumpy():
        if label[0] < 0:
            continue
        x_min = int(label[1] * img.shape[0])
        y_min = int(label[2] * img.shape[1])
        x_max = int(label[3] * img.shape[0])
        y_max = int(label[4] * img.shape[1])
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill = False,
            edgecolor = "red",
            linewidth = 3
        )
        plt.gca().add_patch(rect)
    plt.show()
