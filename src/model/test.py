import torch.nn as nn

import torch

from model import ImgEmbedding,Net,MultiHead,DecoderLayer # type: ignore

from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# hyper parameters:
in_channels = 1
out_channels = 256
head_num = 8
decoder_num = 12
kernel_size = 7
device = 'cuda'





net = Net(in_channels=in_channels,
          out_channels=out_channels,
          head_num=head_num,
          decoder_num=decoder_num,
          kernel_size=kernel_size,
          )

net = torch.load('./model/third.pth')

test = torch.zeros((64,1,28,28),dtype=torch.float32)
test = test.to(device)
out = net(test)

image = Image.open('./test.png').convert('L')

transform = transforms.ToTensor()
x = transform(image)
plt.subplot(2,1,1)
plt.imshow(image)
plt.show()
x = x.unsqueeze(0).to(device)

# x = torch.zeros((1,1,7,7),device = device)
net.generate(x)
print(x.shape)


