import argparse
from networks import *
import math
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from os.path import join
from PIL import Image, ImageOps
import time

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return torch.nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)

def save_images(images, name, path, nrow=1):
    img = images.cpu()
    im = img.data.numpy().astype(np.float32)
    im = im.transpose(0,2,3,1)
    imsave(im, [nrow, int(math.ceil(im.shape[0]/float(nrow)))], os.path.join(path, name) )

def imsave(images, size, path):
    img = merge(images, size)
    return cv2.imwrite(path, img)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def test(args):
    input_transform = transforms.Compose([
                                          transforms.ToTensor()
                                          ])
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    #--------------build models--------------------------
    srnet = NetSR(args.scale, num_layers_res=args.layers)
    weights = torch.load(args.model)
    print(weights.keys())
    pretrained_dict = weights["model"].state_dict()
    model_dict = srnet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    srnet.load_state_dict(model_dict)
    wavelet_rec = WaveletTransform(scale=args.scale, dec=False)
    if args.gpu > 0:
        srnet = srnet.cuda()
        wavelet_rec = wavelet_rec.cuda()
    file_path = join(args.workspace, args.path)
    file_path = join(file_path, args.image)
    img = Image.open(file_path)
    print(img.size)
    input = input_transform(img)
    input = input.view(1, 3, img.size[1], img.size[0])
    start_time = time.time()
    srnet.eval()
    if args.gpu > 0:
        input = input.cuda()
    wavelets = forward_parallel(srnet, input, args.gpu)
    prediction = wavelet_rec(wavelets)
    end_time = time.time()-start_time
    info = "===> time: {:4.4f}".format(end_time)
    print(info)
    print("saving file: " + args.image)
    file_path = os.path.join(args.workspace, args.out)
    print("saving file: " + file_path + "/" + args.image)
    save_images(prediction, args.image, path=file_path, nrow=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace", default="./", help="input image file path")
    parser.add_argument("-p", "--path", default="images/demo/", help="input image file path")
    parser.add_argument("-i", "--image", help="input image file path")
    parser.add_argument("-o", "--out", default="test_results", help="output image file path")
    parser.add_argument("-l", "--layers", default=2, type=int, help="number of the layers in residual block")
    parser.add_argument("-s", "--scale", default=2, type=int, help="image scale")
    parser.add_argument("-m", "--model", help="input model file path")
    parser.add_argument("-g", "--gpu", default=1, type=int, help="GPU device index, -1 is CPU")
    args = parser.parse_args()
    test(args)
