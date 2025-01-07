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


def dice_score(prediction, target, smooth=1e-6):
    prediction = prediction.view(-1)
    target = target.view(-1)

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        dice = dice_score(prediction, target, self.smooth)
        loss = 1.0 - dice
        return loss


class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def inter_fd(self, f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = nn.functional.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = nn.functional.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        idx_s = random.sample(range(s_C), min(s_C, t_C))
        idx_t = random.sample(range(t_C), min(s_C, t_C))

        inter_fd_loss = nn.functional.mse_loss(
            f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach()
        )
        return inter_fd_loss

    def intra_fd(self, f_s):
        sorted_s, indices_s = torch.sort(
            nn.functional.normalize(f_s, p=2, dim=(2, 3)).mean([0, 2, 3]),
            dim=0,
            descending=True,
        )
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = nn.functional.mse_loss(
            f_s[:, 0 : f_s.shape[1] // 2, :, :],
            f_s[:, f_s.shape[1] // 2 : f_s.shape[1], :, :],
        )
        return intra_fd_loss

    def forward(self, feature, feature_decoder, final_up):
        f1 = feature[0][-1]  #
        f2 = feature[1][-1]
        f3 = feature[2][-1]
        f4 = feature[3][-1]  # lower feature

        f1_0 = feature[0][0]  #
        f2_0 = feature[1][0]
        f3_0 = feature[2][0]
        f4_0 = feature[3][0]  # lower feature

        f1_d = feature_decoder[0][-1]  # 14 x 14
        f2_d = feature_decoder[1][-1]  # 28 x 28
        f3_d = feature_decoder[2][-1]  # 56 x 56

        f1_d_0 = feature_decoder[0][0]  # 14 x 14
        f2_d_0 = feature_decoder[1][0]  # 28 x 28
        f3_d_0 = feature_decoder[2][0]  # 56 x 56

        final_layer = final_up

        loss = (
            self.intra_fd(f1)
            + self.intra_fd(f2)
            + self.intra_fd(f3)
            + self.intra_fd(f4)
        ) / 4
        loss += (
            self.intra_fd(f1_0)
            + self.intra_fd(f2_0)
            + self.intra_fd(f3_0)
            + self.intra_fd(f4_0)
        ) / 4
        loss += (
            self.intra_fd(f1_d_0) + self.intra_fd(f2_d_0) + self.intra_fd(f3_d_0)
        ) / 3
        loss += (self.intra_fd(f1_d) + self.intra_fd(f2_d) + self.intra_fd(f3_d)) / 3

        loss += (
            self.inter_fd(f1_d, final_layer)
            + self.inter_fd(f2_d, final_layer)
            + self.inter_fd(f3_d, final_layer)
            + self.inter_fd(f1, final_layer)
            + self.inter_fd(f2, final_layer)
            + self.inter_fd(f3, final_layer)
            + self.inter_fd(f4, final_layer)
        ) / 7

        loss += (
            self.inter_fd(f1_d_0, final_layer)
            + self.inter_fd(f2_d_0, final_layer)
            + self.inter_fd(f3_d_0, final_layer)
            + self.inter_fd(f1_0, final_layer)
            + self.inter_fd(f2_0, final_layer)
            + self.inter_fd(f3_0, final_layer)
            + self.inter_fd(f4_0, final_layer)
        ) / 7

        return loss
