import get_param
import matplotlib.pyplot as plt
from Logger import Logger,t_step
from pde_cnn import get_Net
import torch
import numpy as np
from setups import Dataset
from derivatives import dx,dy,laplace,vector2HSV,rot_mac,toCuda,toCpu,params,normal2staggered,staggered2normal,dx_right,dy_bottom
from torch.optim import Adam
import cv2
import math
import numpy as np
import time
import os

torch.manual_seed(1)
torch.set_num_threads(4)
np.random.seed(6)

mu = params.mu
rho = params.rho
dt = params.dt
w,h = params.width,params.height
n_time_steps=params.average_sequence_length
save_movie=False#True#

# load fluid model:
logger = Logger(get_param.get_hyperparam(params),use_csv=False,use_tensorboard=False)
fluid_model = toCuda(get_Net(params))
date_time,index = logger.load_state(fluid_model,None,datetime=params.load_date_time,index=params.load_index)
fluid_model.eval()
print(f"loaded {params.net}: {date_time}, index: {index}")

# setup opencv windows:
cv2.namedWindow('legend',cv2.WINDOW_NORMAL) # legend for velocity field
vector = torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]).cuda()
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('legend',image)

cv2.namedWindow('p',cv2.WINDOW_NORMAL)
cv2.namedWindow('v',cv2.WINDOW_NORMAL)
cv2.namedWindow('a',cv2.WINDOW_NORMAL)

if save_movie:
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	movie_p = cv2.VideoWriter(f'plots/p_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0, (w,  h))
	movie_v = cv2.VideoWriter(f'plots/v_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0, (w-3,  h-3))
	movie_a = cv2.VideoWriter(f'plots/a_{get_param.get_hyperparam(params)}.avi', fourcc, 20.0, (w,  h))

# Mouse interactions:
def mousePosition(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = x
		dataset.mousey = y
cv2.setMouseCallback("p",mousePosition)
cv2.setMouseCallback("v",mousePosition)
cv2.setMouseCallback("a",mousePosition)

FPS = 0
quit = False

with torch.no_grad():
	while True:
		# create new environmet:
		# types to choose from: magnus, box, pipe, image
		# images to choose from: fish, cyber, smiley, wing
		# backgrounds to choose from: empty, cave1, cave2
		dataset = Dataset(w,h,1,1,interactive=True,average_sequence_length=n_time_steps,max_speed=params.max_speed,dt=dt,types=["magnus","image"],images=["fish","cyber","smiley","wing"],background_images=["empty"])
		
		FPS_Counter=0
		last_time = time.time()
		
		#simulation loop:
		for t in range(n_time_steps):
			v_cond,cond_mask,flow_mask,a_old,p_old = toCuda(dataset.ask())
			
			# convert v_cond,cond_mask,flow_mask to MAC grid:
			v_cond = normal2staggered(v_cond)
			cond_mask_mac = (normal2staggered(cond_mask.repeat(1,2,1,1))==1).float()
			flow_mask_mac = (normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()
			
			# MOST IMPORTANT PART: apply fluid model to advace fluid state
			a_new,p_new = fluid_model(a_old,p_old,flow_mask,v_cond,cond_mask)
			v_new = rot_mac(a_new)
			
			# normalize mean of p and a:
			p_new = (p_new-torch.mean(p_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			a_new = (a_new-torch.mean(a_new,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))
			
			
			if t%10==0: # print out results only at every 10th iteration
				print(f"t:{t} (FPS: {FPS})")
				
				# print out p:
				p = flow_mask[0,0]*p_new[0,0].clone()
				p = p-torch.min(p)
				p = p/torch.max(p)
				p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
				if save_movie:
					movie_p.write((255*p).astype(np.uint8))
				cv2.imshow('p',p)
				
				# print out v:
				v_new = flow_mask_mac*v_new+cond_mask_mac*v_cond
				vector = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
				image = vector2HSV(vector)
				image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
				if save_movie:
					movie_v.write((255*image).astype(np.uint8))
				cv2.imshow('v',image)
				
				# print out a:
				a = a_new[0,0].clone()
				a = a-torch.min(a)
				a = toCpu(a/torch.max(a)).unsqueeze(2).repeat(1,1,3).numpy()
				if save_movie:
					movie_a.write((255*a).astype(np.uint8))
				cv2.imshow('a',a)
				
				# keyboard interactions:
				key = cv2.waitKey(1)
				
				if key==ord('x'): # increase flow speed
					dataset.mousev+=0.1
				if key==ord('y'): # decrease flow speed
					dataset.mousev-=0.1
				
				if key==ord('s'): # increase angular velocity
					dataset.mousew+=0.1
				if key==ord('a'): # decrease angular velocity
					dataset.mousew-=0.1
				
				if key==ord('n'): # start new environmet
					break
				
				if key==ord('p'): # print image
					flow = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
					image = vector2HSV(flow)
					flow = toCpu(flow).numpy()
					fig, ax = plt.subplots()
					Y,X = np.mgrid[0:flow.shape[1],0:flow.shape[2]]
					linewidth = image[:,:,2]/np.max(image[:,:,2])
					ax.streamplot(X, Y, flow[1], flow[0], color='k', density=1,linewidth=2*linewidth)
					palette = plt.cm.gnuplot2
					palette.set_bad('k',1.0)
					pm = np.ma.masked_where(toCpu(cond_mask).numpy()==1, toCpu(p_new).numpy())
					plt.imshow(pm[0,0,2:-1,2:-1],cmap=palette)
					plt.axis('off')
					os.makedirs("plots",exist_ok=True)
					name = dataset.env_info[0]["type"]
					if name=="image":
						name = name+"_"+dataset.env_info[0]["image"]
					plt.savefig(f"plots/flow_and_pressure_field_{name}_{get_param.get_hyperparam(params)}.png", bbox_inches='tight')
					plt.show()
				
				if key==ord('q'): # quit simulation
					quit=True
					break
				
				FPS_Counter += 1
				if time.time()-last_time>=1:
					last_time = time.time()
					FPS=FPS_Counter
					FPS_Counter = 0
				
			dataset.tell(toCpu(a_new),toCpu(p_new))
		if quit:
			break
			
if save_movie:
	movie_p.release()
	movie_v.release()
	movie_a.release()

