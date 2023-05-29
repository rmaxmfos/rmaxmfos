import numpy as np
import random
from collections import deque
from scipy.special import softmax
import gym
from gym.spaces import Discrete, Tuple
from math import trunc

def proper_argmax(lst):
    max_vals = np.where(lst == np.max(lst))[0]
    max_indices = random.choice(max_vals)
    return max_indices

def action_on_scale(q_s):
    if q_s >=1 :
        return np.ones((self.bs))
    else:
        return np.zeros((self.bs))
    
def Boltzmann(arr):
    #0.5 is just a temperature parameter, controls the spread of the softmax distribution
    action_value = np.zeros(arr.shape[0])
    prob = softmax(arr/0.4)
    
    action_value = np.random.choice(np.arange(arr.shape[0]), p=prob)
    return action_value

class MetaGames:
    def __init__(self, bs, game, radius, meta_steps):
        self.bs = bs
        self.epsilon = 0.8
        self.lr = 0.75
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)
        self.innerq = np.zeros((self.bs, 2))
        self.radius = radius
        self.t = 0
        self.game = game
        self.meta_steps= meta_steps
        
    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.ones(shape=(self.bs, 2))

        self.init_action = np.random.randint(0,2, size=(self.bs, 2))
        self.t = np.zeros((self.bs, 1))
        return np.concatenate([self.init_action, self.init_action, self.discretise_q(self.innerq), self.t], axis=1) # OBS: INIT_ACTION, ACTION, DISCRETIZED oppo_Q, t

    def discretise_q(self, qtable):
        return np.round(np.divide(qtable, self.radius)) * self.radius

    def select_action(self):
        #select action for opponent only
        if np.random.random() < 1- self.epsilon:
            action = np.random.randint(0,2, (1, )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = np.argmax(self.innerq, axis=1)
        return action # returns action for all agents

    def step(self, our_action):
        opponent_action = self.select_action()
        action = np.stack([opponent_action, our_action], axis=1)
        r1 = np.zeros(self.bs)
        r2 = np.zeros(self.bs)
        
        if self.game == "MP":
        # PLAY MP GAME, GET REWARDS
            r1 = (opponent_action == our_action) * 1
            r2 = 1 - r1
            self.t += 1
        
        elif self.game == "PD":
        # PLAY PD GAME, GET REWARDS
            for i in range(self.bs):
                if our_action[i] == 0 and opponent_action[i] == 0:  #CC
                    action[i] = 0
                    r1[i] = 3/5
                    r2[i] = 3/5
                elif our_action[i] == 0 and opponent_action[i] == 1:  #CD
                    action[i] = 1
                    r1[i] = 0
                    r2[i] = 1
                elif our_action[i] == 1 and opponent_action[i] == 0:  #DC
                    action[i] = 2
                    r1[i] = 1
                    r2[i] = 0
                elif our_action[i] == 1 and opponent_action[i] == 1:  #DD
                    action[i] = 3
                    r1[i] = 1/5
                    r2[i] = 1/5
            self.t += 1
        
        # GENERATE OBSERVATION
        observation = np.concatenate([self.init_action, np.stack([opponent_action, our_action], axis=1), self.discretise_q(self.innerq), self.t], axis=1) 

        # UPDATE OPPONENT Q-TABLE
        self.innerq[:,opponent_action[0]] = self.lr * r2 + (1 - self.lr) * self.innerq[:,opponent_action[0]]

        # CHECK DONE
        done = self.t > 10

        return observation, r1, r2, {"r1": r1, "r2": r2}
    
class MetaGamesSimple:
    #meta-s = [discretized everyone's inner q, oppo init action, t]   meta-a = our inner q
    def __init__(self, bs, radius):
        self.epsilon = 0.95
        self.lr = 0.75
        self.t = 0
        self.bs = bs
        self.radius= radius
        self.innerq = np.zeros((self.bs,2))

    def reset(self):
        #Initialise inner Q table randomly
        self.innerq = np.random.randint(2, size=(self.bs, 2))
        self.init_action = np.random.randint(2, size=(self.bs, 1))
        self.t = np.zeros((self.bs, 1))
        return np.concatenate([self.discretise_q(self.innerq), self.init_action.T, self.t.T], axis=0).T # OBS: INNERQ, ACTION, TIMESTEP
    
    def discretise_q(self, qtable):
        return np.round(np.divide(qtable, self.radius)) * self.radius
        #return [int(i * self.radius) for i in qtable]

    def select_action(self):
        #select action for opponent only
        if np.random.random() < self.epsilon:
            action = np.random.randint(2, size=(self.bs, 1 )) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = np.reshape(np.argmax(self.innerq, axis=1), (self.bs, 1))
        return action # returns action for all agents

    def step(self, action):
        opponent_action = self.select_action()

        # PLAY GAME, GET REWARDS
        r1 = (opponent_action == action) * 1
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.concatenate([self.discretise_q(self.innerq), self.init_action.T, self.t.T], axis=0).T

        # UPDATE OPPONENT Q-TABLE
        self.innerq[:, opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[:, opponent_action]

        # CHECK DONE
        done = self.t > 10

        return observation, r1, done, {"r1": r1, "r2": r2}
    
################################################################################################################################
class MetaGamesLimitedtraj:
    #meta-s = [everyone' action history up to timestep hist_step, t]   meta-a = our inner q
    def __init__(self, hist_step, meta_steps, game):
        self.epsilon = 0.9
        self.lr = 0.75
        self.hist_step = hist_step
        self.meta_steps = meta_steps    
        self.game = game
        self.innerq = np.ones((2,)) * 0.5
        
    def reset(self):
        #Initialise inner Q table randomly
        #self.innerq = np.ones((2,)) * 0.5
        #self.innerq = np.random.rand(2)
        
        self.oppo_deque = [deque(np.random.randint(2, size=(self.hist_step)), maxlen=self.hist_step)]
        self.our_deque = [deque(np.random.randint(2, size=(self.hist_step)), maxlen=self.hist_step)]
        
#         self.oppo_deque = [deque(np.zeros(self.hist_step), maxlen=self.hist_step)]
#         self.our_deque = [deque(np.zeros(self.hist_step), maxlen=self.hist_step)]
        self.t = 0  #for zero-indexings

        return np.concatenate([self.oppo_deque, self.our_deque], axis=1).squeeze() # OBS: INNERQ, ACTION, TIMESTEP

    def select_action(self):
        #select action for opponent only
        if np.random.random() < 1-self.epsilon:
            action = np.random.randint(2) #convert tuple-->tensor
        else:
            #makes sure if indices have same Q value, randomise
            action = proper_argmax(self.innerq)
        return action # returns action for all agents

    def step(self, action):
        opponent_action = self.select_action()
        r1 = 0
        r2 = 0
        
        if self.game == "MP":
        # PLAY MP GAME, GET REWARDS
            r1 = (opponent_action == action) * 1
            r2 = 1 - r1
            self.t += 1
        
        elif self.game == "PD":
        # PLAY PD GAME, GET REWARDS
            if action == 0 and opponent_action == 0:  #CC
                r1 = 3/5
                r2 = 3/5
            elif action == 0 and opponent_action == 1:  #CD
                r1 = 0
                r2 = 1
            elif action == 1 and opponent_action == 0:  #DC
                r1 = 1
                r2 = 0
            elif action == 1 and opponent_action == 1:  #DD
                r1 = 1/5
                r2 = 1/5
            self.t += 1
        
        # GENERATE OBSERVATION
        self.oppo_deque[0].append(opponent_action)
        self.our_deque[0].append(action)

#        observation = np.concatenate([self.init_actions, self.oppo_deque, self.t], axis=1)
        observation = np.concatenate([self.oppo_deque, self.our_deque], axis=1).squeeze()
        # UPDATE OPPONENT Q-TABLE
        self.innerq[opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[opponent_action]

        # CHECK DONE
        done = self.t >= self.meta_steps

        return observation, r1, r2, opponent_action
################################################################################################################################    
class MetaGamesSimplest:
    #meta-s = [oppo_act, our_act, t], meta-a = our_act
    def __init__(self, bs, meta_steps):
        self.epsilon = 0.8
        self.lr = 0.9
        self.bs = bs
        self.max_steps = meta_steps
        #reward table with discretized dimensions, (actions, agents) (no states since num of state =1)

    def reset(self):
        #Initialise inner policy randomly
        #temp_inner_policy = np.random.randint(2, size=(self.bs,))
        #self.init_action = np.random.randint(2, size=(self.bs,))
        temp_inner_policy = np.zeros((self.bs,))
        self.init_action = np.zeros((self.bs,))
        #self.inner_policy = 1 - self.init_action
        self.innerq = np.zeros((self.bs,2))
        self.t = np.ones(self.bs) * -1  #for zero-indexing
        
        return np.stack([temp_inner_policy, self.init_action, self.t], axis=1) # OBS: INNER_ACT, ACTION, TIMESTEP
        #return np.stack([temp_inner_policy, self.init_action], axis=1) # OBS: INNER_ACT, ACTION, TIMESTEP
    
    def step(self, action):
        if np.random.random() < 1-self.epsilon:
            opponent_action = np.random.randint(2, size=(self.bs,))
        else:
            #opponent_action = self.inner_policy
            opponent_action = np.reshape(np.argmax(self.innerq, axis=1), (self.bs, ))

        # PLAY GAME, GET REWARDS
        r1 = (opponent_action == action) * 1
        r2 = 1 - r1
        self.t += 1

        # GENERATE OBSERVATION
        observation = np.stack([opponent_action, action, self.t], axis=1) 
        #observation = np.stack([opponent_action, action], axis=1) 
        
        # UPDATE OPPONENT POLICY
        #self.inner_policy = 1 - action
        self.innerq[range(self.bs), opponent_action] = self.lr * r2 + (1 - self.lr) * self.innerq[range(self.bs), opponent_action]

        # CHECK DONE
        done = self.t >= self.max_steps

        return observation, r1, r2, done
    
    
class MetaGamesTheOne:
    #meta-s = [oppo_act, our_act, our_r, t], meta-a = our_act
    ##for removing t in meta-s
    ###for removing our_r in meta-s
    def __init__(self, step, inner_steps):
        self.epsilon = 0.8
        self.lr = 0.7
        self.inner_steps = inner_steps
        self.oppo_r = 0
        self.our_r = 0
        self.radius = 1/2
        self.choice = np.tile(np.arange(0.1, 0.6, step), (1))
        self.interval = int(1/(2*step) + 1)
        self.done = 0
        
    def reset(self):
        #Initialise action randomly   #0 = WAIT, 1 = JUMP
        self.done = 0
        
        oppo_idx = np.random.randint(len(self.choice))
        #self.oppo_r = self.choice[oppo_idx].item()
        self.oppo_r = 0
        self.oppo_q = np.zeros((5, 2))
        
        our_idx = np.random.randint(len(self.choice))
        #self.our_r = self.choice[our_idx].item()
        self.our_r = 0
        self.our_q = np.zeros((5, 2))
        
        self.oppo_act = np.zeros((self.inner_steps))
        self.our_act = np.zeros((self.inner_steps))
        
        #self.t = 0 
       
        return [np.random.randint(2), np.random.randint(self.inner_steps)] # OBS: OPPO_ACT, OUR_ACT, OUR_R
        #return np.concatenate([self.innerq, self.outerq, self.t], axis=1) # OBS: OPPO_ACT, OUR_ACT, T
    
    def discretise_q(self, qtable):
        return np.round(np.divide(qtable, self.radius)) * self.radius
    
    def ind_to_q(self, ind):
        Q = np.zeros((5*2))
        
        poss_val = int(1//(1/3) + 1)     #inner q can only take 4 values: 0, 1/3, 2/3, 1
        for i in reversed(range(10)):
            if ind >= poss_val ** i :
                q, mod = divmod(ind, poss_val**i)
                Q[i] = q * self.radius
                ind = mod
            else:
                q = 0
                mod = 0
                Q[i] = 0
        return np.reshape(Q, (5,2))
    
    def step(self, our_action):
        # PLAY GAME, GET REWARDS
        observation = [0, 0]
        self.our_q = our_action
        self.done = 0
        self.oppo_r = 0
        self.our_r = 0
        self.oppo_act = np.zeros((self.inner_steps))
        self.our_act = np.zeros((self.inner_steps))

        for t in range(self.inner_steps):
            if self.done == 0:
                self.our_act[t] = int(Boltzmann(self.our_q[trunc(self.our_r/0.2001)]))        #0.2001 instead of 0.2 because of python float
                #opponent choose jump/wait
                if np.random.random() < 1- self.epsilon:
                    self.oppo_act[t] = int(np.random.randint(0, 2)) #convert tuple-->tensor
                else:
                    self.oppo_act[t] = int(proper_argmax(self.oppo_q[trunc(self.oppo_r/0.2001)]))
                
                oppo_idx = np.random.randint(self.choice.shape)
                oppo_advance = self.choice[oppo_idx].item()
                
                our_idx = np.random.randint(self.choice.shape)
                our_advance = self.choice[our_idx].item()

                #step for opponent
                if (self.oppo_act[t] == 0) and (self.oppo_r + oppo_advance > 1):    #if wait & crosses 1 before jump
                    self.oppo_r = 0
                    self.done = 1                                              #fking over
                    observation = [1, 4]
                    #print("opponent burst, FINAL REWARD= "+ str(self.oppo_r))

                elif (self.oppo_act[t] == 0) and (self.oppo_r + oppo_advance <= 1):   #if wait & still within 1
                    self.oppo_r += oppo_advance
                    #print("opponent wait @ x=" + str(self.oppo_r))

                elif self.oppo_act[t] == 1:                             #if jump and crosses
                    self.oppo_r += oppo_advance
                    self.done = 1                                              #fking over
                    observation = [1, t]
                    #print("opponent jumps & ends @ x=" + str(self.oppo_r)+ "@time= " + str(t))

                #step for us
                if (self.our_act[t] == 0) and (self.our_r + our_advance > 1):
                    self.our_r = 0
                    self.done = 1                                              #fking over
                    observation = [0, 4]
                    #print("we burst, FINAL REWARD= " + str(self.our_r))

                elif (self.our_act[t] == 0) and (self.our_r + our_advance <= 1):
                    self.our_r += our_advance
                    #print("we wait @ x=" + str(self.our_r))

                elif self.our_act[t] == 1:
                    self.our_r += our_advance
                    self.done = 1                                             #fking over
                    observation = [0, t]
                    #print("we jump & end @ x=" + str(self.our_r) + "@time= " + str(t))
                    
            else:
                self.oppo_act[t] = 1
                self.our_act[t] = 1
                
            # UPDATE OPPONENT POLICY
            self.oppo_q[t, int(self.oppo_act[t])] = self.lr * self.oppo_r + (1 - self.lr) * self.oppo_q[t, int(self.oppo_act[t])]
            #self.outerq[:, int(action)] = self.lr * self.our_r + (1 - self.lr) * self.innerq[:, int(action)]
                
        if self.our_r > self.oppo_r:
            self.our_r += 0.5
        elif self.our_r < self.oppo_r:
            self.oppo_r += 0.5
            
        # GENERATE OBSERVATION
        #observation = np.concatenate([opponent_action, our_action], axis=1)

        # UPDATE OPPONENT POLICY
        #self.oppo_q[self.t, opponent_action] = self.lr * self.oppo_r + (1 - self.lr) * self.innerq[:, opponent_action]
        #self.outerq[:, int(action)] = self.lr * self.our_r + (1 - self.lr) * self.innerq[:, int(action)]

        return observation, self.our_r, self.oppo_r, self.done
   
    
    
    
    
