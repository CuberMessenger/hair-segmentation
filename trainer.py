import tqdm
import torch
import torch.nn as nn

from utilities import *


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, label):
        loss = 0
        loss += 0.5 * self.dice_loss(prediction, label)
        loss += 0.5 * self.bce_loss(prediction, label)
        return loss


def ddp_training_step(config, model, helper, epoch, rank, verbose=True):
    batch_time = AverageMeter("batch-time")
    data_time = AverageMeter("data-time")
    losses = AverageMeter("losses")
    dices = AverageMeter("dices")
    ious = AverageMeter("ious")

    train_loader = helper.get_train_loader(config)
    loss_function = helper.get_loss_function()
    optimizer = helper.get_optimizer()
    scheduler = helper.get_scheduler()

    model.train()




#####################################################################

def evaluate_model(config, model, accelerator, helper, verbose=True):
    test_loader = helper.get_test_loader(config)
    model, test_loader = accelerator.prepare(model, test_loader)

    loss_function = LossFunction()

    losses = []
    dices = []

    model.eval()
    with torch.no_grad():
        for batch_image, batch_label in tqdm.tqdm(test_loader):
            batch_prediction = model(batch_image)

            loss = loss_function(batch_prediction, batch_label)
            losses.append(loss)

            dice = dice_score(batch_prediction, batch_label)
            dices.append(dice)

    losses = torch.Tensor(losses)
    dices = torch.Tensor(dices)

    loss = losses.mean().item()
    dice = dices.mean().item()

    if verbose:
        print(f"loss: {loss:.3e}")
        print(f"dice: {dice:.3f}")

    return loss, dice


def train_model(config, model, accelerator, helper, verbose=True):
    test_loss, test_dice = evaluate_model(
        config, model, accelerator, helper, verbose=False
    )
    print(
        f"Epoch [ 0/{config['epoch']}]: test loss = {test_loss:.3e}, test dice = {test_dice:.3f}"
    )

    train_loader = helper.get_train_loader(config)
    test_loader = helper.get_test_loader(config)

    loss_function = LossFunction()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model, train_loader, test_loader, optimizer, scheduler = accelerator.prepare(
        model, train_loader, test_loader, optimizer, scheduler
    )

    for epoch in range(config["epoch"]):
        model.train()

        losses = []
        dices = []

        for _, batch_data in enumerate(tqdm.tqdm(train_loader)):
            batch_image, batch_label = batch_data
            optimizer.zero_grad()

            batch_prediction = model(batch_image)
            loss = loss_function(batch_prediction, batch_label)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.detach())
            dices.append(dice_score(batch_prediction, batch_label).detach())
        scheduler.step()

        losses = torch.Tensor(losses)
        dices = torch.Tensor(dices)

        train_loss = losses.mean().item()
        train_dice = dices.mean().item()

        test_loss, test_dice = evaluate_model(
            config, model, accelerator, helper, verbose=False
        )

        if verbose:
            to_print = (
                f"Epoch [{' ' if epoch < 9 else ''}{epoch + 1}/{config['epoch']}]: "
            )
            to_print += f"train loss = {train_loss:.3e}, "
            to_print += f"train dice = {train_dice:.3f}, "
            to_print += f"test loss = {test_loss:.3e}, "
            to_print += f"test dice = {test_dice:.3f}"
            print(to_print)
