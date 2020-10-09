import torch
from torch import nn
from derivatives import rot_mac
import torch.nn.functional as F
from unet_parts import *

def get_Net(params):
	if params.net == "UNet1":
		pde_cnn = PDE_UNet1(params.hidden_size)
	elif params.net == "UNet2":
		pde_cnn = PDE_UNet2(params.hidden_size)
	elif params.net == "UNet3":
		pde_cnn = PDE_UNet3(params.hidden_size)
	return pde_cnn

class PDE_UNet1(nn.Module):
	#inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet1, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 3)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh(x[:,0:1]/400), 10*torch.tanh(x[:,1:2]/10)
		return a_new,p_new

class PDE_UNet2(nn.Module):
	#same as UNet1 but with delta a / delta p
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet2, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 2)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh((a_old+x[:,0:1])/400), 10*torch.tanh((p_old+x[:,1:2])/10)
		return a_new,p_new


class PDE_UNet3(nn.Module):
	#same as UNet2 but with scaling
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet3, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 4)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh((a_old+x[:,0:1]*torch.exp(3*torch.tanh(x[:,2:3]/3)))/400), 10*torch.tanh((p_old+x[:,1:2]*torch.exp(3*torch.tanh(x[:,3:4]/3)))/10)
		return a_new,p_new

