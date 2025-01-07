import sys
import json
import random
import torch
import torch.nn as nn

from thop import profile


def count_parameter(model, verbose=True):
    count = 0
    for name, parameter in model.named_parameters():
        size = 1
        for s in parameter.size():
            size *= s
        if verbose:
            print(f"{name}: {parameter.size()} -> {size}")
        count += size

    if verbose:
        print(f"Num of parameter = {count}\n")
    return count


def profile_model(model, input_size):
    model = model.cuda()
    model.eval()

    input = torch.randn(input_size).cuda()

    flops, params = profile(model, inputs=(input,))

    return flops, params


class StandardOutputDuplicator:
    """
    Helper class for duplicating the standard output to multiple streams

    A typical usage is to duplicate the standard output to both a file and the console

    Example:
    logFile = open("Log.log" mode = "w")
    sys.stdout = StandardOutputDuplicator(logFile)

    or

    sys.stdout = StandardOutputDuplicator(logFile1, logFile2, ...)

    Note that, only duplicate the output in the main process in multiprocessing scenario
    """

    OriginalSystemStandardOutput = None

    def __init__(self, *streams):
        if StandardOutputDuplicator.OriginalSystemStandardOutput is None:
            StandardOutputDuplicator.OriginalSystemStandardOutput = sys.stdout

        self.Streams = streams + (
            StandardOutputDuplicator.OriginalSystemStandardOutput,
        )

    def write(self, data):
        for stream in self.Streams:
            stream.write(data)

    def flush(self):
        pass


def write_json_result(result, path):
    with open(path, mode="w") as file:
        json.dump(result, file, indent=4)


def get_intersection_union(prediction, target, threshold=0.5):
    batch_size = prediction.size(0)

    prediction = prediction.view(batch_size, -1)
    target = target.view(batch_size, -1)

    intersection = (prediction > threshold).float() * target
    union = prediction + target

    return intersection, union


def dice_score(prediction, target, threshold=0.5, smooth=1e-6):
    intersection, union = get_intersection_union(prediction, target, threshold)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice


def iou_score(prediction, target, threshold=0.5, smooth=1e-6):
    intersection, union = get_intersection_union(prediction, target, threshold)

    iou = (intersection + smooth) / (union - intersection + smooth)

    return iou


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        dice = dice_score(prediction, target, self.smooth)
        loss = 1.0 - dice
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

    def all_reduce(self):
        device = torch.device("cuda")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(
            total, torch.distributed.ReduceOp.SUM, async_op=False
        )
        self.sum, self.count = total.tolist()
        self.average = self.sum / self.count
