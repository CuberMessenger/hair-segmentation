import sys
import torch

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
