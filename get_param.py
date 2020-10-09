import argparse

def str2bool(v):
	"""
	'boolean type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def params():
	"""
	return parameters for training / testing / plotting of models
	:return: parameter-Namespace
	"""
	parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')

	# Training parameters
	parser.add_argument('--net', default="UNet2", type=str, help='network to train (default: UNet2)', choices=["UNet1","UNet2","UNet3"])
	parser.add_argument('--n_epochs', default=1000, type=int, help='number of epochs (after each epoch, the model gets saved)')
	parser.add_argument('--n_grad_steps', default=500, type=int, help='number of gradient descent steps')
	parser.add_argument('--hidden_size', default=20, type=int, help='hidden size of network (default: 20)')
	parser.add_argument('--n_batches_per_epoch', default=5000, type=int, help='number of batches per epoch (default: 5000)')
	parser.add_argument('--batch_size', default=100, type=int, help='batch size (default: 100)')
	parser.add_argument('--n_time_steps', default=1, type=int, help='number of time steps to propagate gradients (default: 1)')#note: this only works with static environments (and didn't bring any benefits anyway)
	parser.add_argument('--average_sequence_length', default=5000, type=int, help='average sequence length in dataset (default: 5000)')
	parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
	parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
	parser.add_argument('--loss_bound', default=20, type=float, help='loss factor for boundary conditions')
	parser.add_argument('--loss_cont', default=0, type=float, help='loss factor for continuity equation')
	parser.add_argument('--loss_nav', default=1, type=float, help='loss factor for navier stokes equations')
	parser.add_argument('--loss_rho', default=10, type=float, help='loss factor for keeping rho fixed')
	parser.add_argument('--loss_mean_a', default=0, type=float, help='loss factor to keep mean of a around 0')
	parser.add_argument('--loss_mean_p', default=0, type=float, help='loss factor to keep mean of p around 0')
	parser.add_argument('--regularize_grad_p', default=0, type=float, help='regularizer for gradient of p. evt needed for very high reynolds numbers (default: 0)')
	parser.add_argument('--max_speed', default=1, type=float, help='max speed for boundary conditions in dataset (default: 1)')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
	parser.add_argument('--lr_grad', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
	parser.add_argument('--clip_grad_norm', default=None, type=float, help='gradient norm clipping (default: None)')
	parser.add_argument('--clip_grad_value', default=None, type=float, help='gradient value clipping (default: None)')
	parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
	parser.add_argument('--log_grad', default=False, type=str2bool, help='log gradients during training (turn on for debugging)')
	parser.add_argument('--plot_sqrt', default=False, type=str2bool, help='plot sqrt of velocity value (to better distinguish directions at low velocities)')
	parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')
	parser.add_argument('--flip', default=False, type=str2bool, help='flip training samples randomly during training (default: False)')
	parser.add_argument('--integrator', default='imex', type=str, help='integration scheme (explicit / implicit / imex) (default: imex)',choices=['explicit','implicit','imex'])
	parser.add_argument('--loss', default='square', type=str, help='loss type to train network (default: square)',choices=['square'])
	parser.add_argument('--loss_multiplier', default=1, type=float, help='multiply loss / gradients (default: 1)')
	parser.add_argument('--target_freq', default=7, type=float, help='target frequency of optimal control algorithm (default: 7; choose value between 2-8)')

	# Setup parameters
	parser.add_argument('--width', default=300, type=int, help='setup width')
	parser.add_argument('--height', default=100, type=int, help='setup height')
	
	# Fluid parameters
	parser.add_argument('--rho', default=1, type=float, help='fluid density rho')
	parser.add_argument('--mu', default=1, type=float, help='fluid viscosity mu')
	parser.add_argument('--dt', default=1, type=float, help='timestep of fluid integrator')
	
	# Load parameters
	parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
	parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
	parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
	parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
	
	# parse parameters
	params = parser.parse_args()
	
	return params

def get_hyperparam(params):
	return f"net {params.net}; hs {params.hidden_size}; mu {params.mu}; rho {params.rho}; dt {params.dt};"
