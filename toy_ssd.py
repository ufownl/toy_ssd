import mxnet as mx

class FeatureExtractor:
    def __init__(self, ctx=mx.cpu()):
        net = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=True, ctx=ctx)
        self._features = mx.gluon.nn.HybridSequential()
        with self._features.name_scope():
            for block in net.features[:11]:
                self._features.add(block)
        self._features.hybridize()

    def __call__(self, inputs):
        return self._features(inputs)


def _class_predictor(num_anchors, num_classes):
    return mx.gluon.nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

def _box_predictor(num_anchors):
    return mx.gluon.nn.Conv2D(num_anchors * 4, 3, padding=1)

def _downsample(num_filters):
    block = mx.gluon.nn.HybridSequential()
    for _ in range(2):
        block.add(mx.gluon.nn.Conv2D(num_filters, 3, padding=1))
        block.add(mx.gluon.nn.BatchNorm(in_channels=num_filters))
        block.add(mx.gluon.nn.Activation("relu"))
    block.add(mx.gluon.nn.MaxPool2D(2))
    block.hybridize()
    return block

def _flatten_prediction(pred):
    return mx.nd.flatten(pred.transpose(axes=(0, 2, 3, 1)))

def _concat_predictions(preds):
    return mx.nd.concat(*preds, dim=1)


class ToySSD(mx.gluon.nn.Block):
    def __init__(self, num_classes, downsample_filters=512, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        self._anchor_sizes = [[0.2, 0.272], [0.37, 0.447]]
        self._anchor_ratio = [1, 2, 0.5]
        self._num_classes = num_classes
        with self.name_scope():
            self._cls_preds = mx.gluon.nn.Sequential()
            self._box_preds = mx.gluon.nn.Sequential()
            self._downsamples = mx.gluon.nn.Sequential()
            for i in range(len(self._anchor_sizes)):
                num_anchors = len(self._anchor_sizes[i]) + len(self._anchor_ratio) - 1
                self._cls_preds.add(_class_predictor(num_anchors, num_classes))
                self._box_preds.add(_box_predictor(num_anchors))
                if i < len(self._anchor_sizes) - 1:
                    self._downsamples.add(_downsample(downsample_filters))

    def forward(self, features):
        anchors = []
        cls_preds = []
        box_preds = []
        for i in range(len(self._anchor_sizes)):
            anchors.append(mx.nd.contrib.MultiBoxPrior(
                features,
                sizes = self._anchor_sizes[i],
                ratios = self._anchor_ratio
            ))
            cls_preds.append(_flatten_prediction(self._cls_preds[i](features)))
            box_preds.append(_flatten_prediction(self._box_preds[i](features)))
            if i < len(self._anchor_sizes) - 1:
                features = self._downsamples[i](features)
        return (
            _concat_predictions(anchors),
            _concat_predictions(cls_preds).reshape((0, -1, self._num_classes + 1)),
            _concat_predictions(box_preds)
        )


def targets(anchors, cls_preds, labels):
    res = mx.nd.contrib.MultiBoxTarget(anchors, labels, cls_preds.transpose(axes=(0, 2, 1)))
    box_target = res[0]
    box_mask = res[1]
    cls_target = res[2]
    return cls_target, box_target, box_mask


class FocalLoss(mx.gluon.loss.Loss):
    def __init__(self, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, y, label):
        y = F.softmax(y)
        y = F.pick(y, label)
        loss = -self._alpha * ((1 - y) ** self._gamma) * F.log(y)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SmoothL1Loss(mx.gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, y, label, mask):
        loss = F.smooth_l1((y - label) * mask, scalar=1.0)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


if __name__ == "__main__":
    features = FeatureExtractor(ctx=mx.cpu())
    model = ToySSD(1)
    model.initialize(mx.init.Xavier(), ctx=mx.cpu())
    anchors, cls_preds, box_preds = model(features(mx.nd.zeros((4, 3, 256, 256), ctx=mx.cpu())))
    print(anchors, cls_preds, box_preds)
    cls_target, box_target, box_mask = targets(anchors, cls_preds, mx.nd.zeros((4, 1, 5), ctx=mx.cpu()))
    print(cls_target, box_target, box_mask)
    cls_loss = FocalLoss()
    print(cls_loss(cls_preds, cls_target))
    box_loss = SmoothL1Loss()
    print(box_loss(box_preds, box_target, box_mask))
