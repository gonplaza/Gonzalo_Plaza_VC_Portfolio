# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:10:52 2023

@author: Gonzalo Plaza Molina
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size # maximum size of replay buffer
        self.index = 0
        self.size = 0 # current size of replay buffer
		
        self.device = device

        # create arrays for replay buffer and dead array to flag when agent reaches a terminal state
        self.state = np.zeros((max_size, state_dim)) 
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1)) # one reward per experience
        self.next_state = np.zeros((max_size, state_dim))
        self.end = np.zeros((max_size, 1))


    def add(self, state, action, reward, next_state, end):
        self.state[self.index] = state
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_state[self.index] = next_state
        self.end[self.index] = end # used to end learning if end = 1, otherwise end = 0
        self.index = (self.index + 1) % self.max_size # once max_size reached, substitute oldest experiences
        self.size = min(self.size + 1, self.max_size) # ensures size does not exceed maximum capacity of buffer

    def sample(self, batch_size):
        inds = np.random.randint(0, self.size, size=batch_size) # generate a batch of random indices between 0 and (not including) the size of the buffer

        # extract and return batch of experiences and convert the arrays to tensors
        return (
			torch.FloatTensor(self.state[inds]).to(self.device),
			torch.FloatTensor(self.action[inds]).to(self.device),
			torch.FloatTensor(self.reward[inds]).to(self.device),
			torch.FloatTensor(self.next_state[inds]).to(self.device),
			torch.FloatTensor(self.end[inds]).to(self.device)
		)


class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2, dt=0.002, x0=None):
        # parameters for the Ornstein-Uhlenbeck noise
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt  # Time increment
        self.x0 = x0
        self.state = np.copy(self.mu)
        self.reset()  # resets parameters when initialised

    def reset(self):
        # Resets to x0 if provided, else to mu
        self.state = self.x0 if self.x0 is not None else np.copy(self.mu)

    def noise(self):
        # Generates OU noise 
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def __repr__(self):
        # Textual representation including dt
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma}, theta={self.theta}, dt={self.dt})'



class Softmax:
	# function to calculate weights using the softmax operator
	def softmax(self, x):
		#x = x.detach().numpy()
		#exp_x = np.exp(x)
		#return exp_x / np.sum(exp_x)
		# PyTorch's softmax function is used directly
		return F.softmax(x, dim=-1)

    # calculate the weights for the Q-values, and return the final Q-value with the weights swapped
	def swap(self, q1, q2):
		#s_weights = self.softmax([q1, q2])
		#w1, w2 = s_weights
		q_values = torch.stack([q1, q2], dim=0)
		s_weights = self.softmax(q_values)
		# Extract softmax weights for each Q-value.
		w1, w2 = s_weights.unbind(dim=0)
		return w1*q2 +w2*q1



class Actor(nn.Module): # inherits neural network module
	def __init__(self, state_dim, action_dim, hidden_width, name): #states are inputs and actions are outputs
		super(Actor, self).__init__()
		
		self.state_dim = state_dim # number of input states
		self.action_dim = action_dim # number of output actions
		self.hidden_width = hidden_width # width of hidden layer

		self.layer1 = nn.Linear(self.state_dim, self.hidden_width) # first layer takes state as input and outputs to hidden layer
		nn.init.xavier_uniform_(self.layer1.weight)  # Xavier weight initialisation for layer 1
		nn.init.zeros_(self.layer1.bias) # bias initialisation for layer 1 to zero
		self.bn1 = nn.LayerNorm(hidden_width)  # normalisation for layer 1

		self.layer2 = nn.Linear(self.hidden_width, self.hidden_width)
		nn.init.xavier_uniform_(self.layer2.weight) 
		nn.init.zeros_(self.layer2.bias) 
		self.bn2 = nn.LayerNorm(hidden_width)	

		self.layer3 = nn.Linear(self.hidden_width, self.action_dim)
		nn.init.xavier_uniform_(self.layer3.weight) 
		nn.init.zeros_(self.layer3.bias) 

		self.checkpoint_file = name+'SSTD3_weights.pth'

	def forward(self, state):
		a = self.layer1(state)
		a = self.bn1(a)
		a = F.relu(a) # Rectified Linear Unit (ReLU) introduces non-linearity
		a = self.layer2(a)
		a = self.bn2(a)
		a = F.relu (a)
		a = (torch.tanh(self.layer3(a))+1)/2 # scales and shifts outputs between 0 and 1 - endowment
		return a
	
	def save_checkpoint(self):
		#print('saving checkpoint...')
		torch.save(self.state_dict(), self.checkpoint_file) # safes the state dictionary containing model parameters in checkpoint file
	
	def load_checkpoint(self):
		print('loading checkpoint...')
		self.load_state_dict(torch.load(self.checkpoint_file)) # loads previosuly saved state back into the model
		


class Critic(nn.Module): 
	def __init__(self, state_dim, action_dim, hidden_width, name): # states and actions are inputs
		super(Critic, self).__init__()
		
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_width = hidden_width

		self.layer1a = nn.Linear(self.state_dim + self.action_dim, self.hidden_width) # states and actions are inputs
		nn.init.xavier_uniform_(self.layer1a.weight) 
		nn.init.zeros_(self.layer1a.bias) 
		self.bn1a = nn.LayerNorm(hidden_width) 

		self.layer2a = nn.Linear(self.hidden_width, self.hidden_width)
		nn.init.xavier_uniform_(self.layer2a.weight) 
		nn.init.zeros_(self.layer2a.bias) 
		self.bn2a = nn.LayerNorm(hidden_width)
		
		self.layer3a = nn.Linear(self.hidden_width, 1) # output is single scalar Q-value estimation for an input state-action pair
		nn.init.xavier_uniform_(self.layer3a.weight) 
		nn.init.zeros_(self.layer3a.bias) 
		
        # layers for second Critic network
		self.layer1b = nn.Linear(self.state_dim + self.action_dim, self.hidden_width) 
		nn.init.xavier_uniform_(self.layer1b.weight) 
		nn.init.zeros_(self.layer1b.bias) 
		self.bn1b = nn.LayerNorm(hidden_width) 

		self.layer2b = nn.Linear(self.hidden_width, self.hidden_width)
		nn.init.xavier_uniform_(self.layer2b.weight) 
		nn.init.zeros_(self.layer2b.bias) 
		self.bn2b = nn.LayerNorm(hidden_width)
		
		self.layer3b = nn.Linear(self.hidden_width, 1)
		nn.init.xavier_uniform_(self.layer3b.weight) 
		nn.init.zeros_(self.layer3b.bias) 

		self.checkpoint_file = name+'SSTD3_weights.pth'

	def forward(self, state, action):
		sa = torch.cat([state, action], 1) # concatenates states and actions

        # estimate Q-value with both Critic networks
		q1 = self.layer1a(sa)
		q1 = self.bn1a(q1)
		q1 = F.relu(q1)
		q1 = self.layer2a(q1)
		q1 = self.bn2a(q1)
		q1 = F.relu(q1)
		q1 = self.layer3a(q1)
		
		q2 = self.layer1b(sa)
		q2 = self.bn1b(q2)
		q2 = F.relu(q2)
		q2 = self.layer2b(q2)
		q2 = self.bn2b(q2)
		q2 = F.relu(q2)
		q2 = self.layer3b(q2)

		return q1, q2
	
	def save_checkpoint(self):
		#print('saving checkpoint...')
		torch.save(self.state_dict(), self.checkpoint_file)
	
	def load_checkpoint(self):
		print('loading checkpoint...')
		self.load_state_dict(torch.load(self.checkpoint_file))
		


class Agent(object):
	def __init__(self, state_dim, action_dim, tau=0.005, gamma=0.99, net_width=256, actor_lr=1e-4, critic_lr=1e-3, \
			  batch_size = 64, policy_delay_freq = 2, max_size=100000):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.max_size = max_size
		self.gamma = gamma
		self.tau = tau # soft update coefficient 
		self.actor_lr = actor_lr # actor network learning rate
		self.critic_lr = critic_lr # critic netork learning rate
		self.batch_size = batch_size
		self.delay_freq = policy_delay_freq
		self.delay_counter = 0
		
		self.device = device

		self.actor = Actor(state_dim, action_dim, net_width, name="Actor").to(self.device) # Initialise actor network
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr) # Initialise the Adam optimiser for the actor network
		self.critic = Critic(state_dim, action_dim, net_width, name="Critic").to(self.device) # Initialise critic network
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		
        #target networks fo not need optimisers as parameters are updated using soft update
		self.actor_target = Actor(state_dim, action_dim, net_width, name="Actor_Target").to(self.device) # Initialise target actor network	
		self.critic_target = Critic(state_dim, action_dim, net_width, name="Critic_Target").to(self.device) # Initialise target critic network

		self.softmax_operator = Softmax()
		
		self.noise = OUNoise(size=action_dim, mu=0, theta=0.15, sigma=0.2) # initialise noise class
		
		self.memory = ReplayBuffer(self.state_dim, self.action_dim, self.max_size) # initialise replay buffer class

	def select_action(self, state, noise=True):# only used when interact with the env
		self.actor.eval() # switch actor network to evaluation mode (appropriate as not training)
		
		with torch.no_grad(): # disable gradient computation as not needed for action selection (not training) - saves computation time	
			# convert input state into 2D flat tensor to ensure the correct input shape and type to the neural network:
			observation = state.reshape(1, -1).to(dtype=torch.float32).to(self.device)
			a = self.actor.forward(observation).to(self.device) # select an action based on the observation
			
		self.actor.train() # switch back to training mode

		# add the Ornstein-Uhlenbeck noise for action exploration
		ou_noise = self.noise.noise()
		a += ou_noise
		
		return a.cpu().detach().numpy().flatten() # converts the action tensor to numpy array, moves it to the CPU (ensures compatibility), converts it to 1D, and detaches to ensure no gradient computation
	
	def remember(self, state, action, reward, new_state, end):
		self.memory.add(state, action, reward, new_state, end)
	
	def learn(self):
		if self.memory.size < self.batch_size:
			return # return if there are less experiences than the required batch size

		self.delay_counter += 1 # used for updating
		
		with torch.no_grad():
			state, action, reward, next_state, end = self.memory.sample(self.batch_size) # sample batch of experiences
			target_action = self.actor_target.forward(next_state)
			"""Target smoothing noise is chosen not to be added here as it adds too much variaiblity - there is already exploration noise"""
			# noise = torch.randn_like(target_action) * 0.0025 # Scale the standard deviation of the noise to 0.02
			# noise = noise.clamp(-0.005, 0.005) # clamp noise to prevent excessively large perturbations
			# smoothed_target_action = target_action + noise # apply random noise to smooth the target action

		# target_Q1, target_Q2 = self.critic_target.forward(next_state, smoothed_target_action) # target Q-values
		target_Q1, target_Q2 = self.critic_target.forward(next_state, target_action) # target Q-values
		target_Q = self.softmax_operator.swap(target_Q1, target_Q2) # apply the softmax operator
		y = reward + (self.gamma*target_Q*(1 - end)).detach() # TD target, no update if end=1


		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute the Mean Squared Error (MSE) loss function
		critic_loss = F.mse_loss(current_Q1, y.detach()) + F.mse_loss(current_Q2, y.detach()) # detach method used to prevent gradients from flowing into the target Q-values (should remain fixed during critic update)

		self.critic_optimizer.zero_grad() # clear gradients from previous iteration
		critic_loss.backward() # compute gradients of loss function
		self.critic_optimizer.step() # update the critic network parameters using Adam optimisation

        # only update actor network and both target networks parameters at periodic intervals
		if self.delay_counter % self.delay_freq == 0:
			# Update actor parameters - instead of gradient ascent of the mean of the Q-values, we do gradient descent on the negative mean
			q1, _ = self.critic(state, self.actor(state))
			actor_loss = -q1.mean() # mean works as only one of the two critic outputs used, as I should
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen parameters of the target networks using soft update
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = 0 # reset delay counter
			
	def save_model_parameters(self): # saves the model paramenters of all networks
		self.actor.save_checkpoint()
		self.actor_target.save_checkpoint()
		self.critic.save_checkpoint()
		self.critic_target.save_checkpoint()
		
	def load_model_parameters(self): # loads the model parameters for all networks
		self.actor.load_checkpoint()
		self.actor_target.load_checkpoint()
		self.critic.load_checkpoint()
		self.critic_target.load_checkpoint()
		
	def check_actor_params(self): # used for monitoring network parameters
		
        # store parameters of current actor network and original actor network
		current_actor_params = self.actor.named_parameters() 
		current_actor_dict = dict(current_actor_params)
		self.original_actor = copy.deepcopy(self.actor)
		original_actor_dict = dict(self.original_actor.named_parameters())
		
        # store parameters of current critic network and original critic network
		current_critic_params = self.critic.named_parameters()
		current_critic_dict = dict(current_critic_params)
		self.original_critic = copy.deepcopy(self.critic)
		original_critic_dict = dict(self.original_critic.named_parameters())
		
		print('Checking Actor parameters')
		
        # print current and original actor parameters
		for param in current_actor_dict:
			print(param, torch.equal(original_actor_dict[param], current_actor_dict[param]))
			
		print('Checking critic parameters')
		
        # print current and original critic parameters
		for param in current_critic_dict:
			print(param, torch.equal(original_critic_dict[param], current_critic_dict[param]))

        # pause and wait for user input to continue program - allows for manual inspection of printed parameters	
		input()

		
    
