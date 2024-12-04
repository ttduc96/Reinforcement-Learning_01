#############################################################################################
# Start date: 18/10/2024                                                                    #
# Project name: Jack's car problem                                                          #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import sys

#In [9]


#Problem parameters
class jcp:
    @staticmethod
    def max_cars():
        return 20
    
    @staticmethod
    def gamma():
        return 0.9
    
    @staticmethod
    def credit_reward():
        return 10
    
    @staticmethod
    def moving_reward():
        return -2

class poisson_:
    
    def __init__(self, lambda_v):
        self.lambda_v = lambda_v
        
        epsilon = 0.01
        
        # [α , β] is the range of n's for which the pmf value is above ε
        self.alpha = 0
        state = 1
        self.vals = {}
        summer = 0
        
        while(1):
            if state == 1:
                temp = poisson.pmf(self.alpha, self.lambda_v) 
                if(temp <= epsilon):
                    self.alpha+=1
                else:
                    self.vals[self.alpha] = temp
                    summer += temp
                    self.beta = self.alpha+1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.beta, self.lambda_v)
                if(temp > epsilon):
                    self.vals[self.beta] = temp
                    summer += temp
                    self.beta+=1
                else:
                    break    
        
        # normalizing the pmf, values of n outside of [α, β] have pmf = 0
        
        added_val = (1-summer)/(self.beta-self.alpha)
        for key in self.vals:
            self.vals[key] += added_val
   
    def f(self, n):
        try:
            Ret_value = self.vals[n]
        except(KeyError):
            Ret_value = 0
        finally:
            return Ret_value

class location:
    def __init__(self, req, ret):
        self.alpha = req
        self.beta = ret
        self.poisson_alpha = poisson_(self.alpha)
        self.poisson_beta = poisson_(self.beta)

#Location initialisation

A = location(3,3)
B = location(4,2)

#Initializing the value and policy matrices. Initial policy has zero value for all states.
value = np.zeros((jcp.max_cars()+1, jcp.max_cars()+1))
policy = value.copy().astype(int)

def apply_action(state, action):
    return [max(min(state[0] - action, jcp.max_cars()),0) , max(min(state[1] + action, jcp.max_cars()),0)]

def expected_reward(state, action):
    global value
    
    psi = 0
    new_state = apply_action(state, action)
    
    
    psi = psi + jcp.moving_reward()*abs(action)
    
    for Aalpha in range(A.poisson_alpha.alpha, A.poisson_alpha.beta):
        for Balpha in range(B.poisson_alpha.alpha, B.poisson_alpha.beta):
            for Abeta in range(A.poisson_beta.alpha, A.poisson_beta.beta):
                for Bbeta in range(B.poisson_beta.alpha, B.poisson_beta.beta):
                    
                    """
                    Aα : sample of cars requested at location A
                    Aβ : sample of cars returned at location A
                    Bα : sample of cars requested at location B
                    Bβ : sample of cars returned at location B
                    ζ  : probability of this event happening
                    """
                    eta = A.poisson_alpha.vals[Aalpha] * B.poisson_alpha.vals[Balpha] * A.poisson_beta.vals[Abeta] * B.poisson_beta.vals[Bbeta]
                    
                    valid_requests_A = min(new_state[0], Aalpha)
                    valid_requests_B = min(new_state[1], Balpha)
                    
                    rew = (valid_requests_A + valid_requests_B) * (jcp.credit_reward())
                    
                    new_s = [0,0]
                    new_s[0] = max(min(new_state[0] - valid_requests_A 
                                       + Aalpha, jcp.max_cars()), 0)
                    new_s[1] = max(min(new_state[1] - valid_requests_B + Bbeta,
                                       jcp.max_cars()), 0)
                    
                    #Bellman's equation
                    psi += eta * (rew + jcp.gamma() * value[new_s[0]][new_s[1]])
    return psi

def policy_evaluation():
    
    global value
    
    epsilon = policy_evaluation.epsilon
    
    policy_evaluation.epsilon /=10
    
    while(1):
        delta = 0
        
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                
                old_val = value[i][j]
                value[i][j] = expected_reward([i,j], policy[i][j])
                
                delta = max(delta, abs(value[i][j] - old_val))
                
                print('.', end = '')
                sys.stdout.flush()
        
        print(delta)
        sys.stdout.flush()
        
        if delta < epsilon:
            break

policy_evaluation.epsilon = 50

def policy_improvement():
    global policy
    
    policy_stable = True
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            old_action = policy[i][j]
            
            max_act_val = None
            max_act = None
            
            t12 = min(i,5)
            t21 = -min(j,5)
            
            for act in range(t21, t12 + 1):
                sigma = expected_reward([i,j], act)
                if max_act_val == None:
                    max_act_val = sigma
                    max_act = act
                elif max_act_val < sigma:
                    max_act_val = sigma
                    max_act = act
            
            policy[i][j] = max_act
            
            if old_action != policy[i][j]:
                policy_stable = False
    return policy_stable

def save_policy():
    save_policy.counter += 1
    ax = sns.heatmap(policy, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('policy'+str(save_policy.counter)+'.svg')
    plt.close()
    
def save_value():
    save_value.counter += 1
    ax = sns.heatmap(value, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig('value'+ str(save_value.counter)+'.svg')
    plt.close()


# In[41]:


save_policy.counter = 0
save_value.counter = 0


# In[13]:


while(1):
    policy_evaluation()
    ρ = policy_improvement()
    save_value()
    save_policy()
    if ρ == True:
        break
    