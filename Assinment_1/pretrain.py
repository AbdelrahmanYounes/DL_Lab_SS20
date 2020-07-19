import os
import numpy as np
import argparse
import torch
from pprint import pprint
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import check_dir, accuracy, get_logger
from models.pretraining_backbone import ResNet18Backbone
from utils.weights import load_from_weights
from torch.utils.tensorboard import SummaryWriter

global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--weights-init', type=str,
                        default="random")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bs', type=int, default=24, help='batch_size')
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)

    # build model and load weights

    model = ResNet18Backbone(pretrained=False).cuda()
    #model = load_from_weights(model, "pretrain_weights_init.pth", logger)
    checkpoint = torch.load("pretrain_weights_init.pth")
    model.load_state_dict(checkpoint['model'])
    #model.cuda()
    # raise NotImplementedError("TODO: load weight initialization")

    # load dataset
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "train"), transform=train_transform)
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)

    # TODO: loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # raise NotImplementedError("TODO: loss function")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    writer = SummaryWriter("./tensorboard")
    for epoch in range(100):
        print("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validate(val_loader, model, criterion)
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("val_loss:", val_loss)
        print("val_acc:", val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        torch.save(model.state_dict(), os.path.join("./trained_models", "model"+str(epoch)+".pth"))
        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join("./trained_models", "best.pth"))
            # raise NotImplementedError("TODO: save model if a new best validation error was reached")


# train one epoch over the whole training dataset. You can change the method's signature.
def train(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()* inputs.size(0)
    mean_train_loss = running_loss / total
    mean_train_accuracy = correct/ total

    return mean_train_loss, mean_train_accuracy
    # raise NotImplementedError("TODO: training routine")


# validation function. you can change the method's signature.
def validate(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()* inputs.size(0)
    mean_val_loss = running_loss / total
    mean_val_accuracy = correct / total

    # raise NotImplementedError("TODO: validation routine")
    return mean_val_loss, mean_val_accuracy


if __name__ == '__main__':
    #torch.cuda.empty_cache()
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
