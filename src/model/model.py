from sys import stdin
import numpy as np

import torch.nn as nn
import torch
from torch import optim

import torchvision
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from einops import rearrange





class ImgEmbedding(nn.Module):
    
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int) -> None:
        super().__init__()
        self.down_size = (kernel_size,kernel_size)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=kernel_size,bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels=out_channels,out_channels=in_channels,kernel_size=kernel_size,stride=kernel_size,bias=False)
        self.deconv.weight = self.conv.weight
        self.pos_embedding = nn.Embedding(17,out_channels)

    def forward(self,x:torch.Tensor):
        
        thumbnail = nn.functional.interpolate(x,self.down_size,mode = 'bilinear',align_corners=False)
        thumbnail = self.conv(thumbnail)
        x = self.conv(x)
        B,C,W,H = x.shape
        # x = x.view((B,C,W*H))
        x = rearrange(x,'b c w h -> b c (w h)')
        out = torch.cat((thumbnail.squeeze(-1),x),dim=-1).permute(0,2,1)

        return out
    
    def decode(self,x:torch.Tensor):
        # (B,T,C)
        x = x.permute(0,2,1).unsqueeze(3).permute(0,1,3,2)
        # (B,C,1,T)
        x = self.deconv(x)
        # (B,1,k,k*T)
        return x

        
class MultiHead(nn.Module):
    
    def __init__(self,feature_dim,head_num) -> None:
        super().__init__()
        self.head_num = head_num
        self.head_dim = feature_dim//head_num
        
        self.query = nn.Linear(feature_dim,feature_dim,bias=False)
        self.key = nn.Linear(feature_dim,feature_dim,bias=False)
        self.value = nn.Linear(feature_dim,feature_dim,bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(50,50)))
        
    def divide_head(self,x:torch.Tensor):
        B,T,C = x.shape
        x = x.view(B,T,self.head_num,self.head_dim).permute(0,2,1,3)
        return x
        
    def forward(self,x:torch.Tensor):
        B,T,C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = self.divide_head(Q)
        K = self.divide_head(K)
        V = self.divide_head(V)
        
        attention_scores = torch.matmul(Q,K.transpose(-1,-2))/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        attention_weights = attention_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_weights = nn.functional.softmax(attention_weights,dim = -1)
        attention_output = torch.matmul(attention_weights,V)
        
        out = attention_output.permute(0,2,1,3).contiguous()
        out = out.view(B,T,C)
        
        return out


class DecoderLayer(nn.Module):
    def __init__(self,feature_dim,head_num) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(feature_dim)
        self.mhd_attention = MultiHead(feature_dim=feature_dim,head_num=head_num)
        self.ln2 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Linear(feature_dim,feature_dim)

    def forward(self,x):
        res1 = x
        x = self.ln1(x)
        x = self.mhd_attention(x)
        x = x+res1
        
        res2 = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x+res2
        
        return x
        
        
class Net(nn.Module):
    def __init__(self,in_channels,out_channels,head_num,decoder_num,kernel_size) -> None:
        super().__init__()
        self.img_embedding = ImgEmbedding(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
        self.decoder = nn.ModuleList([])
        for i in range(decoder_num):
            self.decoder.append(DecoderLayer(out_channels,head_num))
        
        self.mlp = nn.Linear(out_channels,out_channels)
    
    def forward(self,x,label = None):
        x = self.img_embedding(x)
        for layer in self.decoder:
            x = layer(x)
        logits = self.mlp(x)
        out = self.img_embedding.decode(logits)
        
        return out
    
    def inference(self,x:torch.Tensor):
        # x(b c k k*l)
        x = self.img_embedding.conv(x)
        # b c 1 l
        x = x.squeeze(2).permute(0,2,1)
        # l c
        for layer in self.decoder:
            x = layer(x)
        x = self.mlp(x)
        out = self.img_embedding.decode(x)
        return out
        

        
        
    
    
    def generate(self,x:torch.Tensor,max_length=16):
        # x(B,C,H,W)
        
        for i in range(max_length):
            y = self.inference(x)
            now = y[:,:,:,i*7:i*7+7]
            x[:,:,:,i*7+7:i*7+14] = now
            
            img = rearrange(y[0].to('cpu'),'c h w -> h w c')
            plt.imshow(img.detach().numpy())
            plt.show()
    
        
    

if __name__ == '__main__':

    # hyper parameters:
    in_channels = 1
    out_channels = 1024
    head_num = 32
    decoder_num = 6
    kernel_size = 7
    device = 'cuda'
    batch_size = 64



            
            
    MNIST = torchvision.datasets.MNIST('./data',download=True,transform=torchvision.transforms.ToTensor())
    data = DataLoader(MNIST,batch_size=batch_size,shuffle=True,drop_last=True)



    net = Net(in_channels,out_channels,head_num,decoder_num,kernel_size)




    print('='*15+'started!'+'='*15)
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(),lr = 1e-4)

    eoi = torch.ones((batch_size,1,kernel_size,kernel_size),dtype = torch.float32)
    mse_loss = nn.MSELoss()

    for i in range(66):
        for imgs,_ in data:
            
            imgx = rearrange(imgs,'b c (h x) w -> b c x (h w)',x=kernel_size)
            label = torch.cat((imgx,eoi),dim=3)
            
            imgs = imgs.to(device)
            output = net(imgs)
            label = label.to(device)
            
            optimizer.zero_grad()
            loss = mse_loss(output,label)
            loss.backward()
            optimizer.step()
            
            
            print(loss.item())
            
        # if((i+1) % 10 ) == 0:
        #     torch.save(net,'./model/third_epoch'+str(i+1)+'.pth')


    torch.save(net,'./model/third.pth')

    data_iter = iter(data)
    imgs,label = next(data_iter)





    plt.subplot(3,1,1)
    plt.imshow(imgs[0].permute(1,2,0).detach().numpy())

    imgs = imgs.to(device)
    out = net(imgs)
    out = out.to('cpu')
    out = out[0].permute(1,2,0).detach().numpy()
    plt.subplot(3,1,2)
    plt.imshow(out)


    plt.subplot(3,1,3)
    label = rearrange(imgs,'b c (h x) w -> b c x (h w)',x=kernel_size)
    label = label.to('cpu')
    plt.imshow(label[0].permute(1,2,0).detach().numpy())



    plt.show()
