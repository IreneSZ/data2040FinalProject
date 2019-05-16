import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_path', type=str)
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(f'{opt.model_path}'))
else:
    model.load_state_dict(torch.load(f'{opt.model_path}', map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.clock()
out = model(image)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_path = Path('test_outputs')
out_path.mkdir(exist_ok=True, parents=True)
out_img.save(str(out_path / Path(opt.image_name).name))
