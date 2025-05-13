import torch
import os
import shutil
import torch.nn as nn
from model import SimpleCNN
from dataset import CatDogDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, RandomAffine
from sklearn.metrics import classification_report, accuracy_score
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = ArgumentParser(description='CNN images classification model')
    parser.add_argument('--root', type=str, default='./data_test', help='Root')
    parser.add_argument("--epochs", '-e', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--train-model', "-t", type=str, default='train_model')
    parser.add_argument('--logging', "-l", type=str, default='logging')
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=5
        ),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])

    transform_test = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])

    if not os.path.isdir(args.train_model):
        os.mkdir(args.train_model)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    ## Load data
    train_data = CatDogDataset(root=args.root, train=True, transform=transform_train)
    test_data = CatDogDataset(root=args.root, train=False, transform=transform_test)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    summary_writer = SummaryWriter(args.logging)

    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize the model
    model = SimpleCNN(num_class=2).to(device)

    # Initialize the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ##if load model and keep trainning
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_accuracy = checkpoint['best_accuracy']
    else:
        start_epoch = 0
        best_accuracy = 0

    iters = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        process_bar = tqdm(train_loader)
        for iter, (images, labels) in enumerate(process_bar):
            # forward
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            process_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss: {}".format(epoch+1, args.epochs, iter+1, iters, loss))
            summary_writer.add_scalar('Train/Loss', loss, epoch*10+iter)

            # backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_pres, all_labels = [], []
        for iter, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                indices = torch.argmax(output, dim=1)
                all_pres.extend(indices)
                loss = criterion(output, labels)
        all_labels = [label.item() for label in all_labels]
        all_pres = [pre.item() for pre in all_pres]
        accuracy = accuracy_score(all_labels, all_pres)

        print('Epoch {}: Accuracy {}'.format(epoch+1, accuracy))
        summary_writer.add_scalar('Val/Accuracy', accuracy, epoch)

        checkpoint = {
            "epoch": epoch+1,
            "best_accuracy": best_accuracy,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() # save lr
        }
        torch.save(checkpoint, '{}/classify_cnn_last_simple.pt'.format(args.train_model))
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            checkpoint = {
                "epoch": epoch + 1,
                "best_accuracy": best_accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()  # save lr
            }
            torch.save(checkpoint, '{}/classify_cnn_best_simple.pt'.format(args.train_model))
            print(classification_report(all_labels, all_pres))
        # print(classification_report(all_labels, all_pres))
        # exit(0)

        ## tensorboard --logdir #folder