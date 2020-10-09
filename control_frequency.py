import get_param
import matplotlib.pyplot as plt
import matplotlib
from Logger import Logger,t_step
from pde_cnn import get_Net
import torch
from torch.optim import Adam
import numpy as np
from derivatives import vector2HSV,toCuda,toCpu,params,normal2staggered,staggered2normal,rot_mac
from torch.optim import Adam
import cv2
import math
import numpy as np
import time

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

mu = params.mu
rho = params.rho
dt = params.dt
w,h = params.width,params.height
plot = True
n_warmup_time_steps = 500
n_time_steps=200
target_freq = min(max(params.target_freq,2),8)
print(f"target frequency: {target_freq}")

# load fluid model:
logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
pde_cnn = toCuda(get_Net(params))
date_time,index = logger.load_state(pde_cnn,None,datetime=params.load_date_time,index=params.load_index)
pde_cnn.eval()
print(f"loaded {params.net}: {date_time}, index: {index}")

# setup opencv windows for in depth visualizations
cv2.namedWindow('legend',cv2.WINDOW_NORMAL)
vector = toCuda(torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]))
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('legend',image)
cv2.namedWindow('p',cv2.WINDOW_NORMAL)
cv2.namedWindow('v',cv2.WINDOW_NORMAL)

#generate environment for fluid simulation
def get_problem(w,h,object_x=50,object_y=50,object_w=5,object_h=10):
	v_cond = toCuda(torch.zeros(1,2,h,w))
	v_cond[0,1,10:(h-10),0:5]=1
	v_cond[0,1,10:(h-10),(w-5):w]=1

	cond_mask = toCuda(torch.zeros(1,1,h,w))
	cond_mask[0,:,0:3,:]=1
	cond_mask[0,:,(h-3):h,:]=1
	cond_mask[0,:,:,0:5]=1
	cond_mask[0,:,:,(w-5):w]=1
	cond_mask[0,:,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = 1

	flow_mask = 1-cond_mask
	a_old = toCuda(torch.zeros(1,1,h,w))
	p_old = toCuda(torch.zeros(1,1,h,w))
	return v_cond,cond_mask,flow_mask,a_old,p_old

# initialize flow_v (this is the variable, we want to optimize such that we reach the target frequency)
start_v = 0.3
if params.cuda:
	flow_v = torch.ones(1,1,1,1,requires_grad=True,device="cuda")
else:
	flow_v = torch.ones(1,1,1,1,requires_grad=True)

# initialize optimizer
optim = Adam([flow_v],lr=0.2)
E_fft_ys = []

# to obtain smoother gradients, we scale the v_y(t) curve with a gaussian
velocity_y_curve_scaler = torch.exp(-((toCuda(torch.arange(n_time_steps).unsqueeze(1))-n_time_steps/2)/n_time_steps*4)**2)

# optimization loop:
for epoch in range(200):
	v_cond,cond_mask,flow_mask,a_old,p_old = get_problem(w,h)
	
	v_cond = normal2staggered(v_cond)
	cond_mask_mac = (normal2staggered(cond_mask.repeat(1,2,1,1))==1).float()
	flow_mask_mac = (normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()
	
	# warm up simulation with n_warmup_time_steps
	with torch.no_grad():
		for t in range(n_warmup_time_steps):
			a_new,p_new = pde_cnn(a_old,p_old,flow_mask,start_v*flow_v*v_cond,cond_mask)
			p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			a_new = (a_new-torch.mean(a_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			a_old, p_old = a_new, p_new
	
	# simulation that should be used for gradient propagation
	velocity_y_curve = toCuda(torch.zeros(n_time_steps,2))
	for t in range(n_time_steps):
		a_new,p_new = pde_cnn(a_old,p_old,flow_mask,start_v*flow_v*v_cond,cond_mask)
		p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
		a_new = (a_new-torch.mean(a_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
		
		v_new = rot_mac(a_new)
		v_new = cond_mask_mac*v_cond*flow_v*start_v+flow_mask_mac*v_new
		velocity_y_curve[t,0] = torch.mean(v_new[0,0:1,40:(h-40),80:100])
		
		# visualize simulation progress
		if t%10==0:
			with torch.no_grad():
				p = flow_mask[0,0]*p_new[0,0].clone()
				p = p-torch.min(p)
				p = p/torch.max(p)
				cv2.imshow('p',toCpu(p).numpy())
				
				vector = staggered2normal(v_new)[0].clone()
				image = vector2HSV(vector.detach())
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				image[40:(h-40),80]=255
				image[40:(h-40),100]=255
				image[40,80:100]=255
				image[(h-40),80:101]=255
				cv2.imshow('v',image)
				
				key = cv2.waitKey(1)
		
		a_old, p_old = a_new, p_new
	
	# compute expectation value of vortex shedding frequency
	fft_y_curve = torch.sum(torch.fft(velocity_y_curve*velocity_y_curve_scaler,signal_ndim=1,normalized=True)**2,dim=1)
	norm_fft_y_curve = fft_y_curve[:15]/torch.sum(fft_y_curve[:15])
	E_fft_y = torch.sum(norm_fft_y_curve*torch.arange(0,15).cuda())
	
	# loss function to push vortex shedding frequency towards target frequency
	loss = (E_fft_y-target_freq)**2
	
	# propagate gradients throughout simulation and optimize 
	optim.zero_grad()
	loss.backward()
	optim.step()
	
	print(f"iteration {epoch}: E_fft_y = {E_fft_y}")
	E_fft_ys.append(E_fft_y.detach().cpu().numpy())
	
	# plot optimization progress
	font = {'family':'normal','weight':'bold','size':22}
	matplotlib.rc('font',**font)
	
	plt.figure(1)
	plt.clf()
	plt.plot((velocity_y_curve)[:,0].detach().cpu().numpy(),linestyle="dashed",color="r")
	plt.plot((velocity_y_curve*velocity_y_curve_scaler)[:,0].detach().cpu().numpy())
	plt.draw()
	plt.title(f"speed: {round(start_v*flow_v.detach().cpu().numpy()[0][0][0][0],3)}")
	plt.xlabel("time")
	plt.ylabel("$v_y$")
	plt.legend(["$v_y(t)$","gaussian $\cdot v_y(t)$"])
	plt.subplots_adjust(left=0.23,bottom=0.17)
	
	plt.figure(2)
	plt.clf()
	fft_y_curve = fft_y_curve / torch.max(fft_y_curve)
	plt.plot(fft_y_curve.detach().cpu().numpy()[:15])
	plt.vlines(E_fft_y.detach().cpu().numpy(),0,1,color='r')
	plt.vlines(target_freq,0,1,color='g')
	plt.draw()
	plt.title(f"speed: {round(start_v*flow_v.detach().cpu().numpy()[0][0][0][0],3)}")
	plt.xlabel("frequency")
	plt.ylabel("$|V_y|^2$")
	plt.legend(["$|V_y(f)|^2$","$E[|V_y(f)|^2]$","target"])
	plt.subplots_adjust(left=0.23,bottom=0.17)
	
	plt.figure(3)
	plt.clf()
	plt.plot(E_fft_ys,color="r")
	plt.hlines(target_freq,xmin=0,xmax=len(E_fft_ys)-1,color="g")
	plt.legend(["$E[|V_y(f)|^2]$","target"])
	plt.ylim((0,10))
	plt.xlabel("iteration")
	plt.ylabel("frequency")
	plt.subplots_adjust(left=0.23,bottom=0.17)
	plt.pause(0.0001)
	
	del v_cond,cond_mask,flow_mask,a_new,p_new,a_old,p_old,velocity_y_curve,v_new
