#This Python file aims at implementing a the Advantage Actor-Critic (A2C) approach to solve the CartPole-v0 task of the OpenAI Gym Environment

#Importing all the reqiured frameworks
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

		self.state_size=state_size									#Number of features describing each state in the environment
		self.action_size=action_size								#Number of features describing each action in the environment
		self.gamma=gamma											#Discount factor for future rewards
		self.alpha=alpha											#Learning rate during training
		self.n_hl1=n_hl1											#Number of units in the first hidden layer of the network
		self.n_hl2=n_hl2											#Number of units in the second hidden layer of the network
		self.actor,self.critic,self.policy=self.build_network()		#Building the networks that output different output layers


	#Defining a function that builds the actor, critic, and policy network to train/improve the model
	def build_network(self):

		inputs=Input(shape=[None,self.state_size])
		advantage=Input(shape=[None])
		X=Dense(self.n_hl1,activation='relu')(inputs)
		X=Dense(self.n_hl2,activation='relu')(X)
		actor_layer=Dense(self.action_size,activation='softmax')(X)
		critic_layer=Dense(1)(X)

		#Defining a custom loss function (in the keras format) for the policy gradient loss
		def custom_loss(y_pred,y_true):
			probs=k.clip(y_pred,1e-10,1-1e-10)
			return k.sum(k.log(probs)*y_true)*advantage

		#The policy network takes the state and reward obtained, to calculate probabilities for each action(stochastic policy)
		actor_model=Model(inputs=[inputs,advantage],outputs=actor_layer)
		actor_model.compile(optimizer=Adam(learning_rate=self.alpha),loss=custom_loss)

		#The critic network takes the state, and calculates the value of that state
		critic_model=Model(inputs=inputs,outputs=critic_layer)
		critic_model.compile(optimizer=Adam(learning_rate=self.alpha),loss='mean_squared_error')

		#The policy network is like the actor network, but doesnt take any reward, and just predicts the stochastic probabilities
		#Training is not done on this network, but on the actor network instead
		policy_model=Model(inputs=inputs,outputs=actor_layer)

		return self.actor_model,self.critic_model,self.policy_model


	#Choosing a greedy action, except certain times randomly (randomness can be modified with changing epsilon)
	def choose_action(self,state,epsilon)

		probs=self.policy.predict([state])
		if np.random.rand(1)<epsilon:
			return np.random.randint(self.action_size)
		else:
			return np.argmax(probs[0])


	#Updating the actor-critic networks after every timestep in an episode
	def update_network(self,state_now,action,reward,state_next):

		A=np.zeros((self.action_size))
		A[action]=1
		state_now_value=self.critic.predict([state_now])
		state_next_value=self.critic.predict([state_next])
		target_value_now=reward+self.gamma*state_next_value
		advantage=target_value_now-state_now_value
		self.actor.fit([state_now,advantage],A)
		self.critic.fit(state_now,target_value_now)


#Training the agent over a number of episodes, so it learns the policy to claim maximum reward
def train_agent(env,agent,num_episodes,epsilon):

	#Iterating over 'num_episodes'
	for episode_i in range(num_episodes):

		#Maintaining a buffer to store the reward obtained throughout each episode
		reward_history=[]
		#Resetting the starting state of each episode
		state_now=env.reset()
		while True:
			#Choosing an action as per policy, and going to the next state, claiming reward
			action=agent.choose_action(state,epsilon)
			state_next,reward,done,_=env.step(action)
			#Incrementing the reward history
			reward_history[-1]+=reward
			#Updating the networks
			agent.update_network(state_now,action,reward,state_next)
			#Seeing if the episode terminates
			if done==True:
				if episode_i%100==0:
					print("Length of Episode {} : {}".format(episode_i,len(reward_history)))
					print("Average reward claimed by the agent in episode {} : {}".format(episode_i,reward_history[-1]/len(reward_history)))
				#Appending the reward history to calculate reard in the next episode
				reward_history.append(0)
				break
			else:
				#Else setting the next timestep's state
				state_now=state_next


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

#Training the agent using Deep Q-Learning with experience replay with the above mentioned parameters
train_agent(env,agent,num_episodes,epsilon)