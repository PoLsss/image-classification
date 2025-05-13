from argparse import ArgumentParser
import torch
from model import SimpleCNN, DeepCNN
from tranfer_learning import ResNetTransferModel
import cv2
import numpy as np
import torch.nn as nn


def get_args():
    parser = ArgumentParser(description='CNN inference')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--checkpoint', '-c', type=str, default='train_model/resnet18_transfer.pt')
    parser.add_argument('--image-path', '-im', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    categories = ['cats', 'dogs']
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetTransferModel(num_class=2).to(device)
    # model.load_checkpoint('./train_model/resnet18_transfer_best.pt', device=device)
    state_dict = torch.load('train_model/resnet18_transfer_best.pt', map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.backbone.load_state_dict(state_dict)

    model.eval()

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image.astype(np.float64)
    image = np.transpose(image, (2, 0, 1))
    image = image/255.0
    # 4 chiều để đưa vào mô hình
    image = image[None,:,:,:]   # [1,3,512,512]
    image = torch.from_numpy(image).to(device).float()

    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        # print(output)
        probs = softmax(output)
        # print(probs)
    max_idx = torch.argmax(probs)
    print(max_idx)
    predict_class = categories[max_idx]
    print(predict_class)
    # print("Prediction {}, rate {}".format(predict_class, probs[0, max_idx].item()))
    cv2.imshow("{}: {:.3f}".format(predict_class, probs[0, max_idx]*100), ori_image)
    cv2.waitKey(0)
