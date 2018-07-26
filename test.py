import math
import random
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import load_dataset, color_normalize, cook_label
from toy_ssd import FeatureExtractor, ToySSD

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def cook_image(img, size, norm=True):
    img = mx.image.resize_short(img, min(size))
    img, _ = mx.image.center_crop(img, size)
    if norm:
        return mx.image.color_normalize(
            img.astype("float32") / 255,
            mean = mx.nd.array([0.485, 0.456, 0.406]),
            std = mx.nd.array([0.229, 0.224, 0.225])
        )
    else:
        return img

parser = argparse.ArgumentParser(description="Start a ai_challenger_caption tester.")
parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)

print("Loading model...", flush=True)
features = FeatureExtractor(ctx=context)
model = ToySSD(1)
model.load_parameters("model/toy_ssd.params", ctx=context)

for path in args.images:
    print(path)
    raw_img = load_image(path)
    image = cook_image(raw_img, (256, 256))
    image = image.T.expand_dims(0).as_in_context(context)
    source = features(image)
    anchors, cls_preds, box_preds = model(source)
    print("anchors:", anchors)
    print("class predictions:", cls_preds)
    print("box delta predictions:", box_preds)
    cls_probs = mx.nd.softmax(cls_preds).transpose(axes=(0, 2, 1))
    output = mx.nd.contrib.MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=True, clip=False)
    print(output)
    show_img = cook_image(raw_img, (256, 256), norm=False)
    plt.imshow(show_img.asnumpy())
    pens = {}
    for det in output[0]:
        cid = det[0].asscalar()
        if cid < 0:
            continue
        score = det[1].asscalar()
        if score < 0.5:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [show_img.shape[0], show_img.shape[1]] * 2
        x_min, y_min, x_max, y_max = [int(p * s) for p, s in zip(det[2:6].asnumpy().tolist(), scales)]
        rect = plt.Rectangle((y_min, x_min), x_max - x_min, y_max - y_min, fill=False, edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        plt.gca().text(y_min, x_min-2, '{:.3f}'.format(score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=6, color='white')
    plt.show()
