from argparse import ArgumentParser
import torch
from model import SimpleCNN, DeepCNN
from tranfer_learning import ResNetTransferModel
import cv2
import os
import numpy as np
import torch.nn as nn
from glob import glob

def get_args():
    parser = ArgumentParser(description='CNN Batch Inference')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--checkpoint', '-c', type=str, default='train_model/classify_cnn_best_simple.pt')
    parser.add_argument('--image-dir', '-imdir', type=str, required=True, help='Path to test image folder')
    args = parser.parse_args()
    return args


def load_and_preprocess_image(path, image_size):
    ori_image = cv2.imread(path)
    if ori_image is None:
        raise ValueError(f"Cannot read image {path}")
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image


def get_label_from_path(path):
    if 'cat' in path.lower():
        return 0
    elif 'dog' in path.lower():
        return 1
    else:
        raise ValueError(f"Cannot determine label from path: {path}")


if __name__ == '__main__':
    categories = ['cats', 'dogs']
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    # model = SimpleCNN(num_class=2).to(device)
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint["model"])
    model = ResNetTransferModel(num_class=2).to(device)
    state_dict = torch.load('train_model/resnet18_transfer_best.pt', map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.backbone.load_state_dict(state_dict)

    model.eval()

    # Load images
    image_paths = glob(os.path.join(args.image_dir, '*/*.jpg'))
    correct = 0
    total = 0

    softmax = nn.Softmax(dim=1)

    for path in image_paths:
        try:
            image = load_and_preprocess_image(path, args.image_size)
            label = get_label_from_path(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        # Inference
        image_tensor = torch.tensor(image).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probs = softmax(output)
            pred = torch.argmax(probs, dim=1).item()

        if pred == label:
            correct += 1
        total += 1

        print(
            f"[{os.path.basename(path)}] - True: {categories[label]}, Pred: {categories[pred]} (Conf: {probs[0][pred] * 100:.2f}%)")

    acc = correct / total * 100 if total > 0 else 0
    print(f"\nAccuracy: {acc:.2f}% on {total} test images.")


