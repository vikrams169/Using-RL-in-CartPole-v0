#This Python file aims at implementing a the REINFORCE Policy Gradient Approach to solve the CartPole-v0 task of the OpenAI Gym Environment

#Importing all the necessary frameworks
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as k
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
import gym

#A function to keep track of average rewards
def running_reward_avg(rewards):
	output=[]
	for i in range(len(rewards)):
		output.append(sum(rewards[:i])/(i+1))
	return output

#Initializing the Agent class which controls the agents behaviour/improvement in the environment
class Agent(object):


	def __init__(self,env,gamma=0.98,alpha=0.01):

		self.env=env 											#The OpenAI Gym Environment
		self.state_size=self.env.observation_space.shape[0]		#Number of features describing each state in the environment
		self.action_size=self.env.action_space.n				#Number of features describing each action in the environment
		self.gamma=gamma										#Discount factor for future rewards
		self.alpha=alpha										#Learning rate during training
		self.n_hl1=16											#Number of units in the first hidden layer of the network
		self.n_hl2=16											#Number of units in the second hidden layer of the network
		self.network1,self.network2=self.build_network()		#Building the network that takes states as inputs, and stochastic probabilities as output
		self.reward_history=[]									#Reward history to keep track of rewards per episode
		self.episode_lengths=[]									#To keep track of the length of each episode


	#A function to initialise/construct the neural network to calcuate stochastic action probabilities
	def build_network(self):

		inputs=Input(shape=[self.state_size])
		reward=Input(shape=[1])
		X=Dense(self.n_hl1, activation='relu')(inputs)
		X=Dense(self.n_hl2, activation='relu')(X)
		outputs=Dense(self.action_size, activation='softmax')(X)
		model1=Model(inputs=[inputs,reward],outputs=outputs)

		def custom_loss(y_pred,y_true):
			probs=k.clip(y_pred,1e-10,1-1e-10)
			return -k.mean(k.log(probs)*y_true)*reward

		model1.compile(optimizer=Adam(learning_rate=self.alpha),loss=custom_loss)

		model2=Model(inputs=inputs,outputs=outputs)

		return model1,model2


	#A function to calculate the discounted return for a particular timestep in an episode
	def discounted_returns(self,rewards):

		discounted_returns=np.zeros((len(rewards)))
		current_return=0
		for t in reversed(range(len(rewards))):
			current_return=rewards[t]+self.gamma*current_return
			discounted_returns[t]=current_return
		return discounted_returns


	##Choosing a greedy action most of the times (sometimes, a random action to promote exploration)
	def choose_action(self,state,epsilon=0.2):

		policy_output=self.network2.predict(state.reshape([1,self.state_size]))
		if np.random.rand(1)>epsilon:
			return np.argmax(policy_output[0])
		else:
			return np.random.randint(self.action_size)


	#Updating/Fitting the agent's network (at the end of every episode)
	def update_network(self,episode):

		states=[]
		actions=[]
		rewards=[]
		for i in range(len(episode)):
			states.append(episode[i][0])
			actions.append(episode[i][1])
			rewards.append(episode[i][2])
		states=np.array(states)
		actions=np.eye(self.action_size)[np.array(actions)]
		returns=self.discounted_returns(rewards)
		self.network1.fit([states,returns],actions)


	#A function to generate an episode
	def generate_episode(self):

		#A buffer to store information about each timestep of the episode
		episode=[]
		#Maintainging the reward buffer, and initialising the step count
		reward_buffer=0
		j=0
		#Resetting the starting state of the episode
		state_now=self.env.reset()
		while j<1000:
			#Choosing the greedy action, claiming the reward, and proceeding to the bext state
			action=self.choose_action(state_now)
			state_next,reward,done,_=self.env.step(action)
			episode.append([state_now,action,reward])
			#Updating the reward buffer
			reward_buffer+=reward
			j+=1
			#If the episode is done, return the episode buffer, else go to the nect timestep
			if done==True:
				self.reward_history.append(reward_buffer)
				self.episode_lengths.append(j)
				return episode
			else:
				state_now=state_next


	#Training the agent over a number of episodes
	def train(self,num_episodes=5000):

		#Iterating over episodes
		for i in range(num_episodes):
			#Generating an episode
			episode=self.generate_episode()
			#Updating the network using REINFORCE algorithm, after discounting the rewards
			agent.update_network(episode)
			#Keeping track of progress of successsive episodes (legth of episode, and average reward)
			if (i+1)%100==0:
				print("Length of Episode {} : {}".format(i+1,self.episode_lengths[i]))
				print("Total reward claimed by the agent in episode {} : {}".format(i+1,self.reward_history[i]))


#Creating an environment, and an agent
env=gym.make("CartPole-v1")

agent=Agent(env)

#Training the agent using REINFORCE Algorithm of Policy Gradient Methods with the above mentioned parameters
agent.train()

#Plotting the results
fig, axs = plt.subplots(1,2)
axs[0].plot(running_reward_avg(agent.reward_history))
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()
