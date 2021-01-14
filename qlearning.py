#This Python file aims at implementing a Deep Q-Learning approach to solve the CartPole-v1 task of the OpenAI Gym Environment

#Importing all the necessary frameworks
import numpy as np
import tensorflow as tf
import keras 
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
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


	def __init__(self,env,gamma=0.98,alpha=0.01,memory_size=32,minibatch_size=16,update_freq=10):

		self.env=env 											#The OpenAI Gym Environment 
		self.state_size=self.env.observation_space.shape[0]		#Number of features describing each state in the environment
		self.action_size=self.env.action_space.n				#Number of features describing each action in the environment
		self.action_space=np.arange(self.action_size)			#Represents the integer value for each action
		self.gamma=gamma										#Discount factor for future rewards
		self.alpha=alpha										#Learning rate during training
		self.memory_size=memory_size							#Maximum size of the experience replay buffer
		self.minibatch_size=minibatch_size						#Size of minibatch used when updating samples using experience replay
		self.memory=deque(maxlen=self.memory_size)				#Initializing the experience replay buffer as a double ended queue
		self.n_hl1=16											#Number of units in the first hidden layer of the network
		self.n_hl2=16											#Number of units in the second hidden layer of the network
		self.network=self.build_model()							#Building the network that takes states as inputs, and Q-Values as output
		self.target_network=self.build_model()      			#Initializing the target network as the original network
		self.update_freq=update_freq							#The update frequency to equalize the target network to the original one
		self.reward_history=[]									#Reward history to keep track of rewards per episode
		self.episode_lengths=[]									#To keep track of the length of each episode


	#Appending the newly observed data(state,action,reward,next_state,done) to the experience replay
	#If the memeory is full, it automatically deques 
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
			return list(self.memory)[-self.minibatch_size:]


	#Initializing the network that outputs Q-Values for each action of a given state
	def build_model(self):

		inputs=Input(shape=[self.state_size],dtype="float32")
		X=Dense(self.n_hl1,kernel_initializer='RandomNormal',activation="relu")(inputs)
		X=Dense(self.n_hl2,kernel_initializer='RandomNormal',activation="relu")(X)
		outputs=Dense(self.action_size,kernel_initializer='RandomNormal')(X)
		model=Model(inputs=inputs,outputs=outputs)
		model.compile(optimizer=Adam(learning_rate=self.alpha),loss="mse")
		return model


	#Following the epsilon greedy policy to choose actions
	def epsilon_greedy_action(self,qvalues,epsilon=0.2):

		A=np.zeros((self.action_size))+epsilon/self.action_size
		greedy_action=np.argmax(qvalues[0])
		A[greedy_action]+=1-epsilon
		action=np.random.choice(self.action_space,p=A)
		return action


	#Getting the target Q-Values for a particular state, and next_state pair (under a specific action)
	def target_qvalues(self,qvalues,actions,rewards,state_next):

		q_statenext=self.network.predict(state_next.astype("float32"))
		max_q=np.argmax(q_statenext,axis=-1)
		target_qvalues=qvalues.copy()
		for i in range(qvalues.shape[0]):
			target_qvalues[i,actions[i]]=rewards[i]+self.gamma*q_statenext[i,max_q[i]]
		return target_qvalues


	#Updating the network for the minibatch
	def update_network(self,minibatch):

		state_now=[]
		actions=[]
		rewards=[]
		state_next=[]
		for i in range(len(minibatch)):
			state_now.append(minibatch[i][0])
			actions.append(minibatch[i][1])
			rewards.append(minibatch[i][2])
			state_next.append(minibatch[i][3])
		state_now=np.array(state_now)
		actions=np.array(actions)
		rewards=np.array(rewards)
		state_next=np.array(state_next)
		qvalues=self.network.predict(state_now)
		target_qvalues=self.target_qvalues(qvalues,actions,rewards,state_next)
		self.network.fit(state_now.astype("float32"),target_qvalues,epochs=1)


	#Equalizing the target network to the original network at a particular update frequency
	def update_target_network(self):

		self.target_network=keras.models.clone_model(self.network)
		self.target_network.build((None,self.state_size)) 
		self.target_network.compile(optimizer=Adam(learning_rate=self.alpha),loss="mse")
		self.target_network.set_weights(self.network.get_weights())


	#Training the agent to learn to obtain maximum rewards the specified environment
	def train(self,num_episodes=1000):

		#Training over episodes, "num_episodes" times 
		for i in range(num_episodes):

			#Maintaining a buffer for the sequence of each timestep in an episode (for reference purposes)
			reward_buffer=0
			#Each epiosde starts from the initial state of the environment
			state_now=self.env.reset()
			#Episode goes on for the minimum of 1000 timesteps, or when the episode finishes
			j=0
			while j<1000:
				#Claiming a reward for going from current state to next state using an action as per epsilon greedy policy
				qvalues=self.network.predict(state_now.reshape(1,self.state_size).astype("float32"))
				action=self.epsilon_greedy_action(qvalues)
				state_next,reward,done,_=self.env.step(action)
				#Updating reward buffer, and length of the episode
				reward_buffer+=reward
				j+=1
				#Adding the same timestep's data to the experience replay memory of the agent
				self.append_data([state_now,action,reward,state_next])
				#Sampling a minibatch from the experience replay buffer
				minibatch=self.get_minibatch()
				#Updating the agent's network
				self.update_network(minibatch)
				#Updating the target network as per update frequency
				if (j+1)%agent.update_freq==0:
					self.update_target_network()
				#Ending the episode if it has terminated, otherwise proceeding to the next timestep
				if done==True:
					self.reward_history.append(reward_buffer)
					self.episode_lengths.append(j)
					self.clear_experience_replay()
					if (i+1)%10==0:
						print("Reward in Episode {} : {}".format(i+1,self.reward_history[-1]))
						print("Length of Episode {} : {}".format(i+1,self.episode_lengths[-1]))
					break
				else:
					state_now=state_next
			

#Creating an environment, and an agent
env=gym.make("CartPole-v1")

agent=Agent(env)

#Training the agent using Deep Q-Learning with experience replay with the above mentioned parameters
agent.train()

#Plotting the results
fig, axs = plt.subplots(1,2)
axs[0].plot(running_reward_avg(agent.reward_history))
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()



