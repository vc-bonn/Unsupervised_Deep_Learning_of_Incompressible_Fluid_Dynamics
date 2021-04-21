import torch
import numpy as np
from PIL import Image

# we can define domain boundaries inside these .png images.
# These images were not taken into account during training to test the generalization performance of our models.
cyber_truck = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/cyber.png'))).float(),dim=2)<100).float()
fish = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/fish.png'))).float(),dim=2)<100).float()
smiley = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/smiley.png'))).float(),dim=2)<100).float()
wing = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/wing_profile.png'))).float(),dim=2)<100).float()
background1 = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/background1.png'))).float(),dim=2)<100).float()
background2 = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/background2.png'))).float(),dim=2)<100).float()
background3 = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/background3.png'))).float(),dim=2)<100).float()

images = {"cyber":cyber_truck, "fish":fish, "smiley":smiley, "wing":wing}
backgrounds = {"empty":background1,"cave1":background2,"cave2":background3}

"""
ask-tell interface:
ask(): ask for batch os size batch_size to obtain v_cond(t),cond_mask(t),flow_mask(t),a(t),p(t)
tell(a,p): tell results for a(t+1),p(t+1) of batch
"""
#Attention: x/y are swapped (x-dimension=1; y-dimension=0)


class Dataset:
	def __init__(self,w,h,batch_size=100,dataset_size=1000,average_sequence_length=5000,interactive=False,max_speed=3,brown_damping=0.9995,brown_velocity=0.005,init_velocity=0,init_rho=None,n_cond=False,dt=1,types=["magnus","box","pipe"],images=["cyber","fish","smiley","wing"],background_images=["empty"]):
		"""
		create dataset
		:w: width of domains
		:h: height of domains
		:batch_size: batch_size for ask()
		:dataset_size: size of dataset
		:average_sequence_length: average length of sequence until domain gets reset
		:interactive: allows to interact with the dataset by changing the following variables:
			- mousex: x-position of obstacle
			- mousey: y-position of obstacle
			- mousev: velocity of fluid
			- mousew: angular velocity of ball
		:max_speed: maximum speed at dirichlet boundary conditions
		:brown_damping / brown_velocity: parameters for random motions in dataset
		:init_velocity: initial velocity of objects in simulation (can be ignored / set to 0)
		:init_rho / n_cond: ignore these values as well, as fluid density and neumann boundary conditions aren't considered yet
		:dt: time step size of simulation
		:types: list of environments that can be chosen from:
			- "magnus": train magnus effect on randomly moving / rotating balls of random radia
			- "box": train with randomly moving boxes of random width / height
			- "pipe": difficult pipe-environment that contains long range dependencies
			- "image": choose a random image from images as moving obstacle
		:images: list of images that can be chosen from, if 'image' is contained in types-list. You can simply add more images by adding them to the global images-dictionary.
		:background_images: you can also change the static background if the image-type is chosen.
		"""
		#CODO: add more different environemts; add neumann boundary conditions
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		self.a = torch.zeros(dataset_size,1,h,w)
		self.p = torch.zeros(dataset_size,1,h,w)
		self.v_cond = torch.zeros(dataset_size,2,h,w)# one could also think about p_cond... -> neumann
		self.cond_mask = torch.zeros(dataset_size,1,h,w)
		self.padding_x,self.padding_y = 5,3
		self.n_cond = n_cond
		if n_cond:
			self.n_cond_mask = torch.zeros(dataset_size,1,h,w)#neumann condition mask
		self.flow_mask = torch.zeros(dataset_size,1,h,w)
		self.env_info = [{} for _ in range(dataset_size)]
		self.interactive = interactive
		self.interactive_spring = 150#300#200#~ 1/spring constant to move object
		self.max_speed = max_speed
		self.brown_damping = brown_damping
		self.brown_velocity = brown_velocity
		self.init_velocity = init_velocity
		self.init_rho=init_rho
		if init_rho is not None:
			self.rho = torch.zeros(dataset_size,1,h,w)
		self.dt = dt
		self.types = types
		self.images = images
		self.background_images = background_images
			
		self.mousex = 0
		self.mousey = 0
		self.mousev = 0
		self.mousew = 0
		
		for i in range(dataset_size):
			self.reset_env(i)
		
		self.t = 0
		self.i = 0
	
	def reset_env(self,index):
		"""
		reset environemt[index] to a new, randomly chosen domain
		a and p are set to 0, so the model has to learn "cold-starts"
		"""
		self.a[index,:,:,:] = 0
		self.p[index,:,:,:] = 0
		if self.init_rho is not None:
			self.rho[index,:,:,:] = self.init_rho
		
		self.cond_mask[index,:,:,:]=0
		self.cond_mask[index,:,0:3,:]=1
		self.cond_mask[index,:,(self.h-3):self.h,:]=1
		self.cond_mask[index,:,:,0:5]=1
		self.cond_mask[index,:,:,(self.w-5):self.w]=1
		
		if self.n_cond:
			self.n_cond_mask[index,:,:,:]=0
		
		type = np.random.choice(self.types)

		if type == "magnus": # magnus effekt (1)
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity (1.5) (before: 3*(np.random.rand()-0.5)*2)
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			#CODO: implement this in a more elegant way by flipping environment
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_r = np.random.randint(5,20) # object radius (15)
			object_w = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity (3/object_r)
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate v_cond and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / v_cond
			self.cond_mask[index,:,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)] += mask_ball
			self.v_cond[index,0,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)] += v_ball[0]+object_vy
			self.v_cond[index,1,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)] += v_ball[1]+object_vx
			
			
			self.v_cond[index,1,10:(self.h-10),0:5]=flow_v
			self.v_cond[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
			
			if self.n_cond:
				if flow_v>0:
					self.cond_mask[index,:,10:(self.h-10),(self.w-5):self.w]=0
					self.n_cond_mask[index,:,10:(self.h-10),(self.w-5):self.w]=1
				if flow_v<0:
					self.cond_mask[index,:,10:(self.h-10),0:5]=0
					self.n_cond_mask[index,:,10:(self.h-10),0:5]=1
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r
			
		if type == "DFG_benchmark": # DFG benchmark setup from: http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity TODO: set to 0.3 / 1.5
			object_r = 0.05/0.41*(self.h-2*self.padding_y) # object radius
			
			object_y = 0.2/0.41*(self.h-2*self.padding_y)+self.padding_y
			object_x = 0.2/0.41*(self.h-2*self.padding_y)+self.padding_x
			
			object_vx,object_vy,object_w = 0,0,0 # object angular velocity
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate v_cond and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / v_cond
			x_pos1, y_pos1 = int((object_x-object_r)),int((object_y-object_r))
			x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
			self.cond_mask[index,:,y_pos1:y_pos2,x_pos1:x_pos2] += mask_ball
			self.v_cond[index,0,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[0]+object_vy
			self.v_cond[index,1,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[1]+object_vx
			
			# inlet / outlet flow
			profile_size = self.v_cond[index,0,(self.padding_y):-(self.padding_y),:(self.padding_x)].shape[0]
			flow_profile = torch.arange(0,profile_size,1.0)
			flow_profile *= 0.41/flow_profile[-1]
			flow_profile = 4*flow_profile*(0.41-flow_profile)/0.1681
			flow_profile = flow_profile.unsqueeze(1)
			self.v_cond[index,1,(self.padding_y):-(self.padding_y),:(self.padding_x)] = flow_v*flow_profile
			self.v_cond[index,1,(self.padding_y):-(self.padding_y),-(self.padding_x):] = flow_v*flow_profile
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r
			
		if type == "box":# block at random position
			object_h = np.random.randint(5,20) # object height / 2
			object_w = np.random.randint(5,20) # object width / 2
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			self.cond_mask[index,:,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = 1
			self.v_cond[index,0,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = object_vy
			self.v_cond[index,1,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = object_vx
			
			self.v_cond[index,1,10:(self.h-10),0:5]=flow_v
			self.v_cond[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
			
			if self.n_cond:
				if flow_v>0:
					self.cond_mask[index,:,10:(self.h-10),(self.w-5):self.w]=0
					self.n_cond_mask[index,:,10:(self.h-10),(self.w-5):self.w]=1
				if flow_v<0:
					self.cond_mask[index,:,10:(self.h-10),0:5]=0
					self.n_cond_mask[index,:,10:(self.h-10),0:5]=1
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			
		if type == "pipe":# "pipes-labyrinth"
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			self.v_cond[index,1,10:(self.h//4),0:5]=flow_v
			self.v_cond[index,1,(3*self.h//4):(self.h-10),(self.w-5):self.w]=flow_v
			
			if self.n_cond:
				if flow_v>0:
					self.cond_mask[index,:,(3*self.h//4):(self.h-10),(self.w-5):self.w]=0
					self.n_cond_mask[index,:,(3*self.h//4):(self.h-10),(self.w-5):self.w]=1
				if flow_v<0:
					self.cond_mask[index,:,10:(self.h//4),0:5]=0
					self.n_cond_mask[index,:,10:(self.h//4),0:5]=1
				
			self.cond_mask[index,:,(self.h//3-2):(self.h//3+2),0:(3*self.w//4)] = 1
			self.cond_mask[index,:,(2*self.h//3-2):(2*self.h//3+2),(self.w//4):self.w] = 1
			if np.random.rand()<0.5:
				self.cond_mask[index] = self.cond_mask[index].flip(1)
				self.v_cond[index] = self.v_cond[index].flip(1)
				if self.n_cond:
					self.n_cond_mask[index] = self.n_cond_mask[index].flip(1)
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.mousev = flow_v
		
		if type == "image":
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			image = np.random.choice(self.images)
			image_mask = images[image]
			object_h, object_w = image_mask.shape[0], image_mask.shape[1]
			background_image = np.random.choice(self.background_images)
			background_image_mask = backgrounds[background_image]
			
			self.cond_mask[index,:] = 1-(1-self.cond_mask[index,:])*(1-background_image_mask)
			self.cond_mask[index,:,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)] = 1-(1-self.cond_mask[index,:,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)])*(1-image_mask)
			self.v_cond[index,:]=0
			self.v_cond[index,1,20:(self.h-20),0:5]=flow_v
			self.v_cond[index,1,20:(self.h-20),(self.w-5):self.w]=flow_v
			self.v_cond[index,0:1,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)] = object_vy*image_mask
			self.v_cond[index,1:2,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)] = object_vx*image_mask
			
			self.env_info[index]["type"] = type
			self.env_info[index]["image"] = image
			self.env_info[index]["background_image"] = background_image
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
		
		self.flow_mask[index,:,:,:] = 1-self.cond_mask[index,:,:,:]
		if self.n_cond:
			self.flow_mask[index,:,:,:] = self.flow_mask[index,:,:,:]*(1-self.n_cond_mask[index,:,:,:])
	
	def update_envs(self,indices):
		"""
		update boundary conditions of environments[indices]
		"""
		for index in indices:
			
			if self.env_info[index]["type"] == "magnus":
				object_r = self.env_info[index]["r"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_w = self.env_info[index]["w"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = -object_vx
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = -object_vx
						
					if object_y < object_r + 10:
						object_y = object_r+10
						object_vy = -object_vy
					if object_y > self.h - object_r - 10:
						object_y = self.h - object_r - 10
						object_vy = -object_vy
					
				if self.interactive:
					flow_v = self.mousev
					object_w = self.mousew/object_r
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = 0
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = 0
						
					if object_y < object_r + 10:
						object_y = object_r+10
						object_vy = 0
					if object_y > self.h - object_r - 10:
						object_y = self.h - object_r - 10
						object_vy = 0
				
				
				# 1. generate mesh 2 x [2r x 2r]
				y_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
				
				# 3. generate v_cond and multiply with mask
				v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
				
				# 4. add masks / v_cond
				self.cond_mask[index,:,:,:]=0
				self.cond_mask[index,:,0:3,:]=1
				self.cond_mask[index,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,0:5]=1
				self.cond_mask[index,:,:,(self.w-5):self.w]=1
				
				self.cond_mask[index,:,int(object_y-object_r):int(object_y+object_r+1),int(object_x-object_r):int(object_x+object_r+1)] += mask_ball
				self.v_cond[index,0,int(object_y-object_r):int(object_y+object_r+1),int(object_x-object_r):int(object_x+object_r+1)] = v_ball[0]+object_vy
				self.v_cond[index,1,int(object_y-object_r):int(object_y+object_r+1),int(object_x-object_r):int(object_x+object_r+1)] = v_ball[1]+object_vx
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.v_cond[index,1,10:(self.h-10),0:5]=flow_v
				self.v_cond[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["flow_v"] = flow_v
			
			if self.env_info[index]["type"] == "DFG_benchmark":
				object_r = self.env_info[index]["r"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_w = self.env_info[index]["w"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_r + self.padding_x + 1:
						object_x = object_r + self.padding_x + 1
						object_vx = -object_vx
					if object_x > self.w - object_r - self.padding_x - 1:
						object_x = self.w - object_r - self.padding_x - 1
						object_vx = -object_vx
						
					if object_y < object_r + self.padding_y + 1:
						object_y = object_r + self.padding_y + 1
						object_vy = -object_vy
					if object_y > self.h - object_r - self.padding_y - 1:
						object_y = self.h - object_r - self.padding_y - 1
						object_vy = -object_vy
					
				if self.interactive:
					flow_v = self.mousev
					object_w = self.mousew/object_r
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_r + self.padding_x + 1:
						object_x = object_r + self.padding_x + 1
						object_vx = 0
					if object_x > self.w - object_r - self.padding_x - 1:
						object_x = self.w - object_r - self.padding_x - 1
						object_vx = 0
						
					if object_y < object_r + self.padding_y + 1:
						object_y = object_r + self.padding_y + 1
						object_vy = 0
					if object_y > self.h - object_r - self.padding_y - 1:
						object_y = self.h - object_r - self.padding_y - 1
						object_vy = 0
				
				self.v_cond[index,:,:,:]=0
				self.cond_mask[index,:,:,:]=0
				self.cond_mask[index,:,0:3,:]=1
				self.cond_mask[index,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,0:5]=1
				self.cond_mask[index,:,:,(self.w-5):self.w]=1
				
				# 1. generate mesh 2 x [2r x 2r]
				y_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
				
				# 3. generate v_cond and multiply with mask
				v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
				
				# 4. add masks / v_cond
				x_pos1, y_pos1 = int((object_x-object_r)),int((object_y-object_r))
				x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
				self.cond_mask[index,:,y_pos1:y_pos2,x_pos1:x_pos2] += mask_ball
				self.v_cond[index,0,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[0]+object_vy
				self.v_cond[index,1,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[1]+object_vx
				
				# inlet / outlet flow
				profile_size = self.v_cond[index,0,(self.padding_y):-(self.padding_y),:(self.padding_x)].shape[0]
				flow_profile = torch.arange(0,profile_size,1.0)
				flow_profile *= 0.41/flow_profile[-1]
				flow_profile = 4*flow_profile*(0.41-flow_profile)/0.1681
				flow_profile = flow_profile.unsqueeze(1)
				self.v_cond[index,1,(self.padding_y):-(self.padding_y),:(self.padding_x)] = flow_v*flow_profile
				self.v_cond[index,1,(self.padding_y):-(self.padding_y),-(self.padding_x):] = flow_v*flow_profile
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["w"] = object_w
				self.env_info[index]["flow_v"] = flow_v
			
			if self.env_info[index]["type"] == "box":
				object_h = self.env_info[index]["h"]
				object_w = self.env_info[index]["w"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_w + 10:
						object_x = object_w + 10
						object_vx = -object_vx
					if object_x > self.w - object_w - 10:
						object_x = self.w - object_w - 10
						object_vx = -object_vx
						
					if object_y < object_h + 10:
						object_y = object_h+10
						object_vy = -object_vy
					if object_y > self.h - object_h - 10:
						object_y = self.h - object_h - 10
						object_vy = -object_vy
					
				if self.interactive:
					flow_v = self.mousev
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_w + 10:
						object_x = object_w + 10
						object_vx = 0
					if object_x > self.w - object_w - 10:
						object_x = self.w - object_w - 10
						object_vx = 0
						
					if object_y < object_h + 10:
						object_y = object_h+10
						object_vy = 0
					if object_y > self.h - object_h - 10:
						object_y = self.h - object_h - 10
						object_vy = 0
				
				
				self.cond_mask[index,:,:,:]=0
				self.cond_mask[index,:,0:3,:]=1
				self.cond_mask[index,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,0:5]=1
				self.cond_mask[index,:,:,(self.w-5):self.w]=1
				
				self.cond_mask[index,:,int(object_y-object_h):int(object_y+object_h),int(object_x-object_w):int(object_x+object_w)] = 1
				self.v_cond[index,0,int(object_y-object_h):int(object_y+object_h),int(object_x-object_w):int(object_x+object_w)] = object_vy
				self.v_cond[index,1,int(object_y-object_h):int(object_y+object_h),int(object_x-object_w):int(object_x+object_w)] = object_vx
				
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				self.v_cond[index,1,10:(self.h-10),0:5]=flow_v
				self.v_cond[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["flow_v"] = flow_v
				
			if self.env_info[index]["type"] == "pipe":
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
				if self.interactive:
					flow_v = self.mousev
					self.v_cond[index] = self.v_cond[index]/self.env_info[index]["flow_v"]*flow_v
				self.env_info[index]["flow_v"] = flow_v
				
			if self.env_info[index]["type"] == "image":
				object_h = self.env_info[index]["h"]
				object_w = self.env_info[index]["w"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				
				image_mask = images[self.env_info[index]["image"]]
				background_image_mask = backgrounds[self.env_info[index]["background_image"]]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_w//2 + 10:
						object_x = object_w//2 + 10
						object_vx = -object_vx
					if object_x > self.w - object_w//2 - 10:
						object_x = self.w - object_w//2 - 10
						object_vx = -object_vx
						
					if object_y < object_h//2 + 10:
						object_y = object_h//2+10
						object_vy = -object_vy
					if object_y > self.h - object_h//2 - 10:
						object_y = self.h - object_h//2 - 10
						object_vy = -object_vy
					
				if self.interactive:
					flow_v = self.mousev
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
					object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
					
					if object_x < object_w//2 + 10:
						object_x = object_w//2 + 10
						object_vx = 0
					if object_x > self.w - object_w//2 - 10:
						object_x = self.w - object_w//2 - 10
						object_vx = 0
						
					if object_y < object_h//2 + 10:
						object_y = object_h//2+10
						object_vy = 0
					if object_y > self.h - object_h//2 - 10:
						object_y = self.h - object_h//2 - 10
						object_vy = 0
				
				
				self.cond_mask[index,:,:,:]=0
				self.cond_mask[index,:,0:3,:]=1
				self.cond_mask[index,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,0:5]=1
				self.cond_mask[index,:,:,(self.w-5):self.w]=1
				
				
				self.cond_mask[index,:] = 1-(1-self.cond_mask[index,:])*(1-background_image_mask)
				self.cond_mask[index,:,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)] = 1-(1-self.cond_mask[index,:,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)])*(1-image_mask)
				
				
				self.v_cond[index,:]=0
				self.v_cond[index,0,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)] = object_vy*image_mask
				self.v_cond[index,1,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)] = object_vx*image_mask
				
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.v_cond[index,1,20:(self.h-20),0:5]=flow_v
				self.v_cond[index,1,20:(self.h-20),(self.w-5):self.w]=flow_v
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["flow_v"] = flow_v
				
			self.flow_mask[index,:,:,:] = 1-self.cond_mask[index,:,:,:]
			if self.n_cond:
				self.cond_mask[index,:,:,:] = self.cond_mask[index,:,:,:]*(1-self.n_cond_mask[index,:,:,:])
				self.flow_mask[index,:,:,:] = self.flow_mask[index,:,:,:]*(1-self.n_cond_mask[index,:,:,:])
				
	
	def ask(self):
		"""
		ask for a batch of boundary and initial conditions
		:return: v_cond, cond_mask, flow_mask, a, p
		"""
		if self.interactive:
			self.mousev = min(max(self.mousev,-self.max_speed),self.max_speed)
			self.mousew = min(max(self.mousew,-self.max_speed),self.max_speed)
		
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		self.update_envs(self.indices)
		if self.n_cond and self.init_rho is not None:
			return self.v_cond[self.indices],self.cond_mask[self.indices],self.flow_mask[self.indices],self.a[self.indices],self.p[self.indices],self.rho[self.indices],self.n_cond_mask[self.indices]
		if self.n_cond:
			return self.v_cond[self.indices],self.cond_mask[self.indices],self.flow_mask[self.indices],self.a[self.indices],self.p[self.indices],self.n_cond_mask[self.indices]
		if self.init_rho is not None:
			return self.v_cond[self.indices],self.cond_mask[self.indices],self.flow_mask[self.indices],self.a[self.indices],self.p[self.indices],self.rho[self.indices]
		return self.v_cond[self.indices],self.cond_mask[self.indices],self.flow_mask[self.indices],self.a[self.indices],self.p[self.indices]
	
	def tell(self,a,p,rho=None):
		"""
		return the updated fluid state (a and p) to the dataset
		"""
		self.a[self.indices,:,:,:] = a.detach()
		self.p[self.indices,:,:,:] = p.detach()
		if self.init_rho is not None:
			self.rho[self.indices,:,:,:] = rho.detach()
		
		self.t += 1
		if self.t % (self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
			self.reset_env(int(self.i))
			self.i = (self.i+1)%self.dataset_size
