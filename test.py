from dataset import *
from networks import *
import math
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import os
import time

class option():
    def __init__(self):
        self.testfiles = 'images/demo.list'
        self.testsize = 10
        self.testroot = 'images/demo'
        self.output_height = 128
        self.output_width = None
        self.upscale = 2
        self.scale_back = False
        self.test_batchSize = 1
        self.workers = 2
        self.cuda = True
        self.ngpu = 1
        self.outf = 'test_results/'
        self.nrow = 1
        self.num_layers_res = 2

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return torch.nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)

def save_images(images, name, path, nrow=10):
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

opt = option()

if not os.path.isdir(opt.outf):
    os.mkdir(opt.outf)

mag = int(math.pow(2, opt.upscale))
if opt.scale_back:
    is_scale_back = True
else:
    is_scale_back = False

#--------------build models--------------------------
#srnet = srnet['model']
#srnet.eval()
srnet = NetSR(opt.upscale, num_layers_res=opt.num_layers_res)
weights = torch.load('model/x2/sr_model_epoch_90_iter_0.pth')
print(weights.keys())
pretrained_dict = weights["model"].state_dict()
model_dict = srnet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
srnet.load_state_dict(model_dict)
print(srnet)

wavelet_rec = WaveletTransform(scale=opt.upscale, dec=False)
criterion_m = torch.nn.MSELoss(size_average=True)

if opt.cuda:
    srnet = srnet.cuda()
    wavelet_rec = wavelet_rec.cuda()
    criterion_m = criterion_m.cuda()

#-----------------load dataset--------------------------
test_list, _ = loadFromFile(opt.testfiles, opt.testsize)
test_set = ImageDatasetFromFile(test_list, opt.testroot,
                                input_height=opt.output_height, input_width=opt.output_width,
                                output_height=opt.output_height, output_width=opt.output_width,
                                crop_height=None, crop_width=None,
                                is_random_crop=False, is_mirror=False, is_gray=False,
                                upscale=mag, is_scale_back=is_scale_back)

test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batchSize,
                                               shuffle=False, num_workers=int(opt.workers))

for titer, batch in enumerate(test_data_loader,0):
    start_time = time.time()
    srnet.eval()
    input, target = Variable(batch[0]), Variable(batch[1])
    if opt.cuda:
        input = input.cuda()
        target = target.cuda()
    
    wavelets = forward_parallel(srnet, input, opt.ngpu)
    prediction = wavelet_rec(wavelets)
    mse = criterion_m(prediction, target)
    psnr = 10 * math.log10(1 / (mse.item()) )
    info = "===> PSNR: {:.4f} dB time: {:4.4f}".format(psnr, time.time()-start_time)
    print(info)
    fileName = test_list[titer]
    print("saving file: " + fileName)
    save_images(prediction, fileName, path=opt.outf, nrow=opt.nrow)
    #org_file = os.path.join(opt.outf, "org_" + fileName)
    #cv2.imwrite(org_file, srnet)

