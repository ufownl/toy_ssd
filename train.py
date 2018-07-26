import os
import time
import argparse
import mxnet as mx
from dataset import load_dataset, color_normalize, cook_label
from toy_ssd import FeatureExtractor, ToySSD, targets, FocalLoss, SmoothL1Loss

def train(batch_size, context, sgd=False):
    print("Loading dataset...", flush=True)
    training_set, validating_set = load_dataset(batch_size)

    features = FeatureExtractor(ctx=context)
    model = ToySSD(1)
    cls_loss = FocalLoss()
    box_loss = SmoothL1Loss()

    if os.path.isfile("model/toy_ssd.ckpt"):
        with open("model/toy_ssd.ckpt", "r") as f:
            ckpt_lines = f.readlines()
        ckpt_argv = ckpt_lines[-1].split()
        epoch = int(ckpt_argv[0])
        best_L = float(ckpt_argv[1])
        learning_rate = float(ckpt_argv[2])
        epochs_no_progress = int(ckpt_argv[3])
        model.load_parameters("model/toy_ssd.params", ctx=context)
    else:
        epoch = 0
        best_L = float("Inf")
        epochs_no_progress = 0
        learning_rate = 0.0005
        model.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate)
    if sgd:
        print("Optimizer: SGD")
        trainer = mx.gluon.Trainer(model.collect_params(), "SGD",
                                   {"learning_rate": learning_rate, "momentum": 0.5, "clip_gradient": 5.0})
    else:
        print("Optimizer: Adam")
        trainer = mx.gluon.Trainer(model.collect_params(), "Adam",
                                   {"learning_rate": learning_rate, "clip_gradient": 5.0})
    print("Training...", flush=True)
    while learning_rate >= 1e-8:
        ts = time.time()

        training_L = 0.0
        training_batch = 0
        training_set.reset()
        for batch in training_set:
            training_batch += 1
            x = color_normalize(batch.data[0].as_in_context(context))
            label = cook_label(batch.label[0].as_in_context(context))
            source = features(x)
            with mx.autograd.record():
                anchors, cls_preds, box_preds = model(source)
                cls_target, box_target, box_mask = targets(anchors, cls_preds, label)
                L = cls_loss(cls_preds, cls_target) + box_loss(box_preds, box_target, box_mask)
                L.backward()
            trainer.step(batch_size)
            batch_L = mx.nd.mean(L).asscalar()
            if batch_L != batch_L:
                raise ValueError()
            training_L += batch_L
            print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" %
                (epoch, training_batch, batch_L, training_L / training_batch, time.time() - ts), flush=True)

        validating_L = 0.0
        validating_batch = 0
        validating_set.reset()
        for batch in validating_set:
            validating_batch += 1
            x = color_normalize(batch.data[0].as_in_context(context))
            label = cook_label(batch.label[0].as_in_context(context))
            source = features(x)
            anchors, cls_preds, box_preds = model(source)
            cls_target, box_target, box_mask = targets(anchors, cls_preds, label)
            L = cls_loss(cls_preds, cls_target) + box_loss(box_preds, box_target, box_mask)
            batch_L = mx.nd.mean(L).asscalar()
            if batch_L != batch_L:
                raise ValueError()
            validating_L += batch_L

        epoch += 1

        avg_L = training_L / training_batch
        print("[Epoch %d]  learning_rate %.10f  training_loss %.10f  validating_loss %.10f  epochs_no_progress %d  duration %.2fs" % (
            epoch, learning_rate, training_L / training_batch, validating_L / validating_batch, epochs_no_progress, time.time() - ts
        ), flush=True)

        if avg_L < best_L:
            best_L = avg_L
            epochs_no_progress = 0
            model.save_parameters("model/toy_ssd.params")
            with open("model/toy_ssd.ckpt", "a") as f:
                f.write("%d %.10f %.10f %d\n" % (epoch, best_L, learning_rate, epochs_no_progress))
        elif epochs_no_progress < 2:
            epochs_no_progress += 1
        else:
            epochs_no_progress = 0
            learning_rate *= 0.5
            trainer.set_learning_rate(learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a toy_ssd trainer.")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    while True:
        try:
            train(
                batch_size = 256,
                context = context,
                sgd = args.sgd
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
