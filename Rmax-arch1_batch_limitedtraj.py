import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import softmax
import random
import pickle
from datetime import datetime
from collections import Counter

from tqdm import tqdm

from env_mp_simple import MetaGamesLimitedtraj
from rmax_1_batch_limitedtraj import RmaxAgentTraj, Memory

def discretize(number, radius):
    #[0,3,5,4,8] --> [0,3,6,3,9] for radius 3
    return np.round(np.divide(number, radius)) * radius

def Boltzmann(arr):
    #0.5 is just a temperature parameter, controls the spread of the softmax distribution
    #action_value = np.zeros(arr.shape[0])
    prob = softmax(arr/0.25)
    action_value = np.random.choice(len(arr), p=prob)
    return action_value

inner_gamma = 0         #inner game discount factor, 0 since it's a one shot game
meta_gamma = 0.8         #meta game discount factor

R_max = 1
meta_steps = 10

game = "MP"
epsilon = 0.2

plot_rew = [0]*3
plot_visit = [0]*3

y = [0]*3
mean = [0]*3
convolve = [0]*3

for i in range(2,3):
    hist_step= i+2

    # creating environment
    env = MetaGamesLimitedtraj(hist_step, meta_steps, game)

    # creating rmax agent
    memory = Memory()
    rmax = RmaxAgentTraj(R_max, meta_steps+1, meta_gamma, inner_gamma, epsilon, hist_step)

    meta_epi = int(3* rmax.m *rmax.ns)

    #reward tensor for plotting purposes [bs, episode, step, agents]
    plot_rew[i] = np.zeros((meta_epi, meta_steps, 2))
    #visited states vs reward array, [bs, number of s-a pairs, 2], 1 for cumulative reward, 1 for number of visitation
    plot_visit[i] = np.zeros((rmax.ns * rmax.na + 1, 2))    

    for episode in range(meta_epi): #for each meta-episode
        #reset environment 
        #initialise meta-state and meta-action randomly
        meta_s = env.reset()
        Q = rmax.Q
        
        for step in range(meta_steps):    #for each meta time step
            #--------------------------------------START OF INNER GAME--------------------------------------  
            #select our inner-action with Boltzmann sampling, oppo inner-action with epsilon greedy 
            our_action = Boltzmann(Q[rmax.find_meta_index(meta_s, "s").astype(int), :])

            #run inner game according to actions
            obs, reward, info, _ = env.step(our_action)

            #---------------------------------------END OF INNER GAME--------------------------------------
            #save reward, info for plotting              
            plot_rew[i][episode,step,0] = reward
            plot_rew[i][episode,step,1] = info

            #meta-action = action that corresponds to max Q(meta_s) = our inner Q
            meta_a = our_action

            #meta-state = discretized inner game Q table of all agents
            new_meta_s = obs

            #meta-reward = sum of rewards of our agent in inner game of K episodes & T timesteps
            our_REW = reward    
            memory.rewards.append(our_REW)

            #rmax update step
            rmax.update(memory, meta_s, meta_a, new_meta_s)

            plot_visit[i][(rmax.nSA >= rmax.m).sum(), 0] += reward
            plot_visit[i][(rmax.nSA >= rmax.m).sum(), 1] += 1
            #prepare meta_s for next step
            meta_s = new_meta_s
            
    plt.clf()
    y[i]= np.divide(plot_visit[i][:,0], plot_visit[i][:,1], out=np.zeros_like(plot_visit[i][:,0]), where=plot_visit[i][:,1]!=0)
    plt.plot(y[i], label= str(i+2) + " timesteps", lw=2)
    plt.xlabel("m-visited state-action pairs out of " + str(rmax.ns * rmax.na))
    plt.ylabel("Mean rewards")
    plt.legend()
    plt.savefig('visitation@eps'+ str(epsilon) + 'step' + str(i+2) + '.png', bbox_inches='tight')
    
    plt.clf()
    mean[i] = np.mean(plot_rew[i][:,:,0], axis=1)
    convolve[i] = np.convolve(mean[i], np.ones(int((len(plot_rew[i])/100))) / (len(plot_rew[i])/100), mode='valid')
    print("Average reward of our agent: " + str(np.mean(plot_rew[i][:,:,0])) + "\n Average reward of another agent: " + str(np.mean(plot_rew[i][:,:,1])))
    
    plt.plot(convolve[i], label= str(i+2) + " timesteps", lw=2)
    plt.xscale('log',base=16) 
    
    #reward at batch 0 only
    plt.xlabel("episodes \n Average reward of our agent: " + str(np.mean(plot_rew[i][:,:,0])) + 
             "\n Average reward of another agent: " + str(np.mean(plot_rew[i][:,:,1]))+
             "\n meta-episode= "+ str(meta_epi) + " meta_steps= " + str(meta_steps) + " meta_gamma= " + str(meta_gamma) + 
             "\n hist_step= " + str(hist_step)+ " epsilon=" + str(epsilon) +
             "\n % of visited states= " + str(round(100 * (rmax.nSA >= rmax.m).sum() / (rmax.nSA.shape[0] * rmax.nSA.shape[1]),3)) + "%")
    plt.ylabel("Mean rewards")
    plt.legend()
    plt.savefig('rewardVSepisode@eps'+ str(epsilon) + 'step' + str(i+2) + '.png', bbox_inches='tight')

    # Open a file and use dump()
    with open('convolve' + str(i) + 'epsilon' + str(epsilon) + '.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(convolve, file)

#Visitation vs Reward curve
plt.clf()

plt.plot(y[0], label='2 timesteps', lw=2)
plt.plot(y[1], label='3 timesteps',lw=2)
plt.plot(y[2], label='4 timesteps', lw=2)

plt.xlabel("m-visited state-action pairs out of " + str(rmax.ns * rmax.na))
plt.ylabel("Mean rewards")
plt.legend()

plt.savefig('visitation_all@eps' + str(epsilon) +'.png', bbox_inches='tight')


# In[171]:
#MA(metaepi/10)reward 
plt.clf()

plt.plot(convolve[0], label="2 timesteps", lw=2)
plt.plot(convolve[1], label="3 timesteps",lw=2)
plt.plot(convolve[2], label="4 timesteps", lw=2)

plt.xscale('log',base=16) 
#reward at batch 0 only
plt.xlabel("episodes \n meta_gamma= " + str(meta_gamma) + " epsilon=" + str(epsilon))
plt.ylabel("Mean rewards")

plt.legend()
plt.savefig('rewardVSepisode@eps' + str(epsilon) + '.png', bbox_inches='tight')


