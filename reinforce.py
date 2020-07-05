#This Python file aims at implementing a the REINFORCE Policy Gradient Approach to solve the CartPole-v0 task of the OpenAI Gym Environment

#Importing all the necessary frameworks
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as k
from keras.optimizers import Adam
import random
import gym

#Initializing the Agent class which controls the agents behaviour/improvement in the environment
class Agent(object):


	def __init__(self,state_size,action_size,gamma,alpha,n_hl1,n_hl2):

		self.state_size=state_size				#Number of features describing each state in the environment
		self.action_size=action_size			#Number of features describing each action in the environment
		self.gamma=gamma						#Discount factor for future rewards
		self.alpha=alpha						#Learning rate during training
		self.n_hl1=n_hl1						#Number of units in the first hidden layer of the network
		self.n_hl2=n_hl2						#Number of units in the second hidden layer of the network
		self.network=self.build_network()		#Building the network that takes states as inputs, and stochastic probabilities as output


	#A function to initialise/construct the neural network to calcuate stochastic action probabilities
	def build_network(self):

		inputs=Input(shape=[None,self.state_size])
		reward=Input(shape=[1])
		X=Dense(self.n_hl1, activation='relu')(inputs)
		X=Dense(self.n_hl2, activation='relu')(X)
		outputs=Dense(self.action_size, activation='softmax')(X)
		model=Model(inputs=[inputs,reward],outputs=outputs)

		def custom_loss(y_pred,y_true):
			probs=k.clip(y_pred,1e-10,1-1e-10)
			return k.sum(k.log(probs)*y_true)*reward

		model.compile(optimizer=Adam(learning_rate=self.alpha),loss=custom_loss)
		return model


	#A function to calculate the discounted return for a particular timestep in an episode
	def disounted_rewards(self,rewards):

		discounted_return=np.zeros((len(rewards)))
		current_return=0
		for t in reversed(range(len(rewards))):
			current_return=rewards[t]+self.gamma*current_return
			discounted_return[t]=current_return
		return discounted_return


	##Choosing a greedy action most of the times (sometimes, a random action to promote exploration)
	def choose_action(self,state,epsilon):

		policy_output=self.network.predict([state])
		if np.random.rand(1)>epsilon:
			return np.argmax(qvalues[0])
		else:
			return np.random.randint(self.action_size)


	#Updating/Fitting the agent's network (at the end of every episode)
	def update_network(episode):

		A=np.zeros((len(episode)))
		for i in range(len(episode)):
			A[i]=int(episode[i,1])
		self.network.fit([episode[:,0],episode[:,2]],A)


#A function to generate an episode
def generate_episode(env,agent,epsilon):

	#A buffer to store information about each timestep of the episode
	episode=[]
	#Resetting the starting state of the episode
	state_now=env.reset()
	while True:
		#Choosing the greedy action, claiming the reward, and proceeding to the bext state
		action=agent.choose_action(state_now,epsilon)
		state_next,reward,done,_=env.step(action)
		episode.append([state_now,action,reward,state_next])
		#If the episode is done, return the episode buffer, else go to the nect timestep
		if done==True:
			return np.array(episode)
		else:
			state_now=state_next


#Training the agent over a number of episodes
def train_agent(env,agent,num_episodes,epsilon):

	#Maintaining a buffer to keep track of progress through episodes
	reward_history=[]
	#Iterating over episodes
	for episode_i in range(num_episodes):
		#Generating an episode, discounting it's rewards, and updating the network using REINFORCE algorithm
		episode=generate_episode(env,agent,epsilon)
		reward_history.append(sum(episode[:,2]))
		episode[:,2]=agent.discounted_rewards(episode[:,2])
		agent.update_network(episode)
		#Keeping track of progress of successsive episodes (legth of episode, and average reward)
		if episode_i%100==0 and episode_i!=0:
			print("Length of Episode {} : {}".format(episode_i,len(episode)))
			print("Average reward claimed by the agent in episode {} : {}".format(episode_i,reward_history[-1]/len(episode)))


#Creating an environment, and an agent
env=gym.make("CartPole-v1")
state_size=env.observation_space.shape[0]
action_size=env.action_space.n
gamma=0.98
alpha=0.0001
n_hl1=64
n_hl2=128
num_episodes=2000
epsilon=0.1

agent=Agent(state_size,action_size,gamma,alpha,n_hl1,n_hl2)

#Training the agent using REINFORCE Algorithm of Policy Gradient Methods with the above mentioned parameters
train_agent(env,agent,num_episodes,epsilon)

