#This Python file aims at implementing a Deep Q-Learning approach to solve the CartPole-v0 task of the OpenAI Gym Environment

#Importing all the necessary frameworks
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import random
import gym



#Initializing the Agent class which controls the agents behaviour/improvement in the environment
class Agent(object):


	def __init__(self,state_size,action_size,gamma,alpha,memory_size,minibatch_size,update_freq,n_hl1,n_hl2):

		self.state_size=state_size						#Number of features describing each state in the environment
		self.action_size=action_size					#Number of features describing each action in the environment
		self.action_space=np.arange(self.action_size)	#Represents the integer value for each action
		self.gamma=gamma								#Discount factor for future rewards
		self.alpha=alpha								#Learning rate during training
		self.memory_size=memory_size					#Maximum size of the experience replay buffer
		self.minibatch_size=minibatch_size				#Size of minibatch used when updating samples using experience replay
		self.memory=deque(maxlen=self.memory_size)		#Initializing the experience replay buffer as a double ended queue
		self.n_hl1=n_hl1								#Number of units in the first hidden layer of the network
		self.n_hl2=n_hl2								#Number of units in the second hidden layer of the network
		self.network=self.build_model()					#Building the network that takes states as inputs, and Q-Values as output
		self.target_network=self.network       			#Initializing the target network as the original network
		self.update_freq=update_freq					#The update frequency to equalize the target network to the original one


	#Appending the newly observed data(state,action,reward,next_state,done) to the experience replay
	def append_data(self,data):

		self.memory.append(data)


	#Clearing the experience replay buffer (to be done at the end of each episode)
	def clear_experience_replay(self):

		self.memory.clear()


	#Getting a randomly sampled minibatch from the experience replay
	def get_minibatch(self):

		if len(self.memory) < self.minibatch_size:
			return list(self.memory)
		else:
			return random.sample(list(self.memory),self.minbatch_size)


	#Initializing the network that outputs Q-Values for each action of a given state
	def build_model(self):

		inputs=Input(shape=[None,self.state_size])
		X=Dense(self.n_hl1)(inputs)
		X=Dense(self.n_hl2)(X)
		outputs=Dense(self.action_size)(X)
		model=Model(inputs=inputs,outputs=outputs)
		model.compile(optimizer=Adam(learning_rate=self.alpha),loss="categorical_crossentropy")
		return model


	#Following the epsilon greedy policy to choose actions
	def epsilon_greedy_action(self,epsilon,qvalues):

		A=np.zeros(qvalues[0].shape)+epsilon/self.action_size
		greedy_action=np.argmax(qvalues[0])
		A[greedy_action]+=1-epsilon
		action=np.zeros((qvalues.shape[0]))
		return action


	#Getting the target Q-Values for a particular state, and next_state pair (under a specific action)
	def target_qvalues(self,qvalues,next_state,reward):

		q_nextstate=self.target_network.predict(next_state)
		max_q=np.argmax(q_nextstate[0])
		target_qvalues=qvalues
		target_qvalues[max_q]=reward+self.gamma*q_nextstate[0,max_q]
		return target_qvalues


	#Updating the network for the minibatch
	def update_network(self,minibatch):

		minibatch=np.array(minibatch)
		qvalues_minibatch=self.network.predict(minibatch[:,0])
		target_qvalues_minibatch=self.target_qvalues(qvalues,minibacth[:,3],minibatch[:,2])
		self.network.fit(minibatch[:,0],target_qvalues_minibatch,epochs=1)


	#Equalizing the target network to the original network at a particular update frequency
	def update_target_network(self):

		self.target_network=self.network


#Training the agent to learn to obtain maximum rewards the specified environment
def train_agent(env,agent,num_epsiodes,epsilon):

	#Training over episodes, "num_episodes" times 
	for episode_i in range(num_episodes):

		#Maintaining a buffer for the sequence of each timestep in an episode (for reference purposes)
		episode=[]
		#Each epiosde starts from the initial state of the environment
		state_now=env.reset()
		#Episode goes on for the minimum of 1000 timesteps, or when the episode finishes
		i=0
		while i<1000:
			#Claiming a reward for going from current state to next state using an action as per epsilon greedy policy
			qvalues=agent.network.predict([state_now])
			action=agent.epsilon_greedy_action(epsilon=epsilon,qvalues)
			state_next,reward,done,_=env.step(action)
			#Appending this data of the timestep to the episode buffer
			episode.append([state_now,action,reward,state_next])
			#Adding the same timestep's data to the experience replay memory of the agent
			agent.append_data([state_now,action,reward,state_next])
			#Sampling a minibatch from the experience replay buffer
			minibatch=agent.get_minibatch()
			#Updating the agent's network
			agent.update_network(minibatch)
			#Updating the target network as per update frequency
			if episode_i!=0 and episode_i%agent.update_freq==0:
				agent.update_target_network
			#Ending the episode if it has terminated, otherwise proceeding to the next timestep
			if done==True:
				if episode_i%10==0:
					print("Length of Episode {} : {}".format(episode_i,len(episode)))
				break
			else:
				state_now=state_next
			

#Creating an environment, and an agent
env=gym.make("CartPole-v1")
state_size=env.observation_space.shape[0]
action_size=env.action_space.n
gamma=0.98
alpha=0.0001
memory_size=5000
minibatch_size=32
update_freq=10
n_hl1=64
n_hl2=128
num_episodes=2000
epsilon=0.1

agent=Agent(state_size,action_size,gamma,alpha,memory_size,minibatch_size,update_freq,n_hl1,n_hl2)

#Training the agent using Deep Q-Learning with experience replay with the above mentioned parameters
train_agent(env,agent,num_episodes,epsilon)



