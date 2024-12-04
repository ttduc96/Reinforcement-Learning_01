#############################################################################################
# Start date: 18/10/2024                                                                    #
# Project name: Easy21                                                                      #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################


#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class const:
    @staticmethod
    def gamma():
        return 1
    
    @staticmethod
    def win():
        return 1
    
    @staticmethod
    def lose():
        return -1
    
    @staticmethod
    def draw():
        return 0
    
    @staticmethod
    def player_continue():
        return 0

def draw_card(player_sum):
    card = np.random.randint(1, 11)
    color = np.random.randint(1, 4)
    if (color == 3):
        return player_sum - card
    else:
        return player_sum + card
    
def bust(player_sum):
    if (player_sum > 21) or (player_sum < 1):
        return 1
    else:
        return 0
    
def step(state, action): # take input (state, action) and put it into the environment to output the new state
    dealer_card, player_sum = state
    if (action == 1):
        player_sum = draw_card(player_sum)
        if bust(player_sum):
            return ((dealer_card, player_sum), const.lose(), True)
        else:
            return ((dealer_card, player_sum), const.player_continue(), False)
    elif (action == 0):
        dealer_sum = dealer_card
        while (dealer_sum < 17):
            dealer_sum = draw_card(dealer_sum)
            if bust(dealer_sum):
                return ((dealer_card, player_sum), const.win(), True)
        if (dealer_sum < player_sum):
            return ((dealer_card, player_sum), const.win(), True)
        elif (dealer_sum == player_sum):
            return ((dealer_card, player_sum), const.draw(), True)
        else:
            return ((dealer_card, player_sum), const.lose(), True)
            
def r_argmax(q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top, ties = q_values[i], [i]
        elif q_values[i] == top:
            ties.append(i)
    return np.random.choice(ties)

class MonteCarloAgent:
    def __init__(self):
        self.q_values = np.zeros((11, 22, 2))
        self.epsilon = 0
        self.alpha_t = 0
        self.action_visits = np.zeros((11, 22, 2))
        self.default_visits = 100
        self.state_action_history = None
        
    def start(self, state):
        action = np.random.randint(2)
        self.action_visits[state[0], state[1], action] += 1
        self.state_action_history = []
        self.state_action_history.append((state, action))
        self.last_state = state
        self.last_action = action
        return action
    
    def step(self, state): # take the current state and output the appropriate action based on greedy epsilon
        self.epsilon = (self.default_visits / (self.default_visits + 
                np.sum(self.action_visits[self.last_state[0],\
                                          self.last_state[1]])))
        if np.random.random() < self.epsilon:
            action = np.random.randint(2)
        else:
            action = r_argmax(self.q_values[state[0], state[1]])
        self.last_state = state
        self.state_action_history.append((state, action))
        return action
    
    def end(self, reward):
        self.last_state, self.last_action = self.state_action_history.pop(0)
        for state, action in self.state_action_history:
            last_state_action = (self.last_state[0],\
                                 self.last_state[1],\
                                 self.last_action)
            self.alpha_t = 1 / self.action_visits[last_state_action]
            self.q_values[last_state_action] += self.alpha_t * \
                    (reward - self.q_values[last_state_action])
            
            self.action_visits[state[0], state[1], action] += 1
            self.last_state = state
            self.last_action = action
            
        last_state_action = (self.last_state[0],\
                             self.last_state[1],\
                             self.last_action)
        self.alpha_t = 1 / self.action_visits[last_state_action]
        self.q_values[last_state_action] += self.alpha_t * \
                (reward - self.q_values[last_state_action])

class TemporalDifferenceAgent:
    def __init__(self, lambda_value):
        self.lambda_value = lambda_value
        self.q_values = np.zeros((11, 22, 2))
        self.epsilon = 0
        self.alpha_t = 0
        self.value_visits = np.zeros((11, 22))
        self.action_visits = np.zeros((11, 22, 2))
        self.eligibility_traces = np.zeros((11, 22, 2))
        self.default_visits = 100
        self.last_state_action = None
        
    def start(self, state):
        action = np.random.randint(2)
        self.action_visits[state[0], state[1], action] += 1
        self.eligibility_traces[state[0], state[1], action] += 1
        self.value_visits[state[0], state[1]] += 1
        self.last_state = state
        self.last_action = action
        self.last_state_action = (state[0], state[1], action)
        return action
    
    def step(self, state, reward): # take the current state and output the appropriate action based on greedy epsilon
        self.alpha_t = 1 / self.action_visits[self.last_state_action]
        self.epsilon = (self.default_visits / (self.default_visits + 
                np.sum(self.action_visits[self.last_state[0],\
                                          self.last_state[1]])))
        if np.random.random() < self.epsilon:
            action = np.random.randint(2)
        else:
            action = r_argmax(self.q_values[state[0], state[1], :])
        
        state_action = (state[0], state[1], action)
        
        delta = reward + const.gamma()*self.q_values[state_action]\
            - self.q_values[self.last_state_action]
           
        self.q_values += self.alpha_t * delta * self.eligibility_traces
        self.eligibility_traces *= const.gamma()*self.lambda_value
        
        self.eligibility_traces[state_action] += 1
        self.action_visits[state_action] += 1
        self.value_visits[state[0], state[1]] += 1
        
        self.last_state = state
        self.last_state_action = (state[0], state[1], action)
        return action
    
    def end(self, reward):      
        self.alpha_t = 1 / self.action_visits[self.last_state_action]
        delta = reward - self.q_values[self.last_state_action]
        temp = self.alpha_t * delta * self.eligibility_traces       
        self.q_values += temp
            
    def get_q_values(self):
        return self.q_values
                
monteCarloAgent = MonteCarloAgent()
for episode in range(100000):
    state = (np.random.randint(1, 11), np.random.randint(1, 11))
    action = monteCarloAgent.start(state)
    terminal = False
    while True:
        state, reward, terminal = step(state, action)
        if not terminal:
            action = monteCarloAgent.step(state)
        else:
            monteCarloAgent.end(reward)
            break
        
optimal_values = np.amax(monteCarloAgent.q_values, axis=2)

lambdaas = list(np.arange(0, 11, 1.0) / 10)
def run_td_lambda(ExperimentAgentClass, num_episodes, show=[True] * 11,\
                  plot_all_lambdas=False):
    mses = []
    running_mses = [[] for _ in range(11)]
    for lambdaa in lambdaas:
        agent = ExperimentAgentClass(lambdaa)
        for episode in range(num_episodes):
            state = (np.random.randint(1, 11), np.random.randint(1, 11))
            action = agent.start(state)
            terminal = False
            while True:
                state, reward, terminal = step(state, action)
                if not terminal:
                    action = agent.step(state, reward)
                else:
                    agent.end(reward)
                    break
            current_mse = np.mean((monteCarloAgent.q_values \
                                   - agent.get_q_values()) ** 2) 
            running_mses[int(lambdaa * 10)].append(current_mse)
            
        final_mse = np.mean((monteCarloAgent.q_values \
                             - agent.get_q_values()) ** 2)
        mses.append(final_mse)
        
        if show[int(lambdaa * 10)]:
            optimal_values = np.amax(agent.get_q_values(), axis=2)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = []
            y = []
            for i in range(1, 11):
                for j in range(1, 22):
                    x.append(i)
                    y.append(j)
            z = optimal_values[1:, 1:].flatten()
            ax.set_xlabel('dealer card')
            ax.set_ylabel('player sum')
            ax.set_zlabel('value')
            ax.set_title('V*(s) for {}({})\nMSE {:0.3f} ({} episodes)' \
                    .format(ExperimentAgentClass.__name__, lambdaa,\
                    final_mse, num_episodes))
            surface = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
            plt.xticks(np.arange(1, 11))
            plt.yticks(np.arange(1, 22, 3))
            plt.show()
    
    for lambdaa in lambdaas:
        if show[int(lambdaa * 10)] or plot_all_lambdas:
            plt.plot(list(range(num_episodes)), running_mses[int(lambdaa * 10)])
    showing_lambdaas = (np.array(show) | np.array([plot_all_lambdas] * 11))
    num_columns = int(np.ceil(np.sum(showing_lambdaas) / 3))
    plt.legend(['λ={}'.format(lambdaa) for lambdaa in lambdaas ],\
               loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('Learning curve of {}'.format(ExperimentAgentClass.__name__))
    plt.xlabel('episode number')
    plt.ylabel('mean square error')
    plt.grid(True)
    plt.show()
        
    plt.plot(lambdaas, mses)
    plt.xlabel('lambda factor')
    plt.ylabel('mean square error')
    plt.title('MSE of {} V* after {} episodes'\
            .format(ExperimentAgentClass.__name__, num_episodes))
    plt.grid(True)
    plt.show()

# run_td_lambda(TemporalDifferenceAgent, 1000, show=[True, *[False] * 9, True ])
run_td_lambda(TemporalDifferenceAgent, 20000, show=[True, *[*[False] * 4, True] * 2], plot_all_lambdas=True)
