
from Environment.environment import Environment
from Agent.agent import Agent
import pickle
import numpy as np
import matplotlib.pyplot as plt

def simulate_easy21(iterations=1000):
    print "\n-------------------"
    print "Easy21 Environment Simulation"
    print "run for n. iterations: "+str(iterations)

    win = 0
    lose = 0
    for i in range(iterations):
        r = environment_easy21()
        if r > 0:
            win += 1
        else:
            lose += 1
    print "wins: %s" % (win)
    print "loses: %s" % (lose)

def environment_easy21(print_values=False):
    # run simulation once
    env = Environment()
    s = env.get_initial_state()
    a = env.get_hit_action()
    while(not s.term):
        if print_values:
            print("state = %s, %s" % (s.pl_sum, s.dl_sum))
        if(s.pl_sum >= 17):
            a = env.get_stick_action()
        else:
            a = env.get_hit_action()
        s = env.step(s,a)
        r = s.rew
    
    if print_values:
        print "Reward = %d" % (r)
    return r

def monte_carlo(iterations=1000000, n0=100):
    print "\n-------------------"
    print "Monte Carlo control"
    print "run for n. iterations: "+str(iterations)
    print "win percentage: "
    # learn
    env = Environment()
    agent = Agent(env, n0)
    agent.MC_control(iterations)
    # plot and store
    agent.show_statevalue_function()
    agent.store_Qvalue_function()


def sarsa(iterations=1000, mlambda=None, n0=100, avg_it=50):
    print "\n-------------------"
    print "TD control Sarsa"
    print "run for n. iterations: "+str(iterations)
    print "plot graph mse vs episodes for lambda equal 0 and lambda equal 1"
    print "list (standard output) win percentage for values of lambda 0, 0.1, 0.2, ..., 0.9, 1"
    monte_carlo_Q = pickle.load(open("Data/Qval_func_1000000_MC_control.pkl", "rb"))
    n_elements = monte_carlo_Q.shape[0]*monte_carlo_Q.shape[1]*2
    mse = []

    if not isinstance(mlambda,list):
        # if no value is passed for lambda, default 0.5
        l = 0.5 if mlambda==None else mlambda
        # learn
        env = Environment()
        agent = Agent(env, n0)
        agent.TD_control(iterations, l, avg_it)
        # plot results
        agent.show_statevalue_function()
    else:
        # test each value of lambda
        for l in mlambda:
            env = Environment()
            agent = Agent(env, n0)
            l_mse = agent.TD_control(iterations, l, avg_it)
            mse.append(l_mse)
        plt.plot(mlambda,mse)
        plt.ylabel('MSE')
        plt.show()

        # plot results
        agent.show_statevalue_function()


def linear_sarsa(iterations=1000, mlambda=None, n0=100, avg_it=100):
    print "\n-------------------"
    print "TD control Sarsa, with Linear function approximation"
    print "run for n. iterations: "+str(iterations)
    print "plot graph mse vs episodes for lambda equal 0 and lambda equal 1"
    print "list (std output) win percentage for values of lambda 0, 0.1, 0.2, ..., 0.9, 1"
    monte_carlo_Q = pickle.load(open("Data/Qval_func_1000000_MC_control.pkl", "rb"))
    n_elements = monte_carlo_Q.shape[0]*monte_carlo_Q.shape[1]*2
    mse = []
    if not isinstance(mlambda,list):
        # if no value is passed for lambda, default 0.5
        l = 0.5 if mlambda==None else mlambda
        # learn
        env = Environment()
        agent = Agent(env, n0)
        agent.TD_control_linear(iterations,l,avg_it)
        agent.show_statevalue_function()
    else:
        # test each value of lambda
        for l in mlambda:
            env = Environment()
            agent = Agent(env, n0)
            l_mse = agent.TD_control_linear(iterations,l,avg_it)
            mse.append(l_mse)
        plt.plot(mlambda,mse)
        plt.ylabel('mse')
        plt.show()

        # plot results
        agent.show_statevalue_function()

if __name__ == '__main__':

    # parameters
    lambdas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    iterationsMC = 1000000
    iterationsSRS = 1000
    iterationsSIM = 1000
    n0 = 500

    # functions

    # This function Simulates the Easy21 environment with a number of iterations.
    # The output will be the number of wins and loses of the player
    simulate_easy21(iterationsSIM)

    # Monte carlo controls the outcome of easy21 using e-greedy exploration strategy
    # n0 is a constant to calculate the nth e-value
    monte_carlo(iterationsMC,n0)

    # This function uses sarsa-lambda to predict the next state using mean-squared error approach
    # The function outputs episodes vs mse graphs after every 1000 steps 
    sarsa(iterationsSRS,lambdas,n0, avg_it=1)

    # This function linearises the salsa approach by linear value function approximation
    # We use constant exploration of e = 0.05 and a constant step-size of 0.01
    linear_sarsa(iterationsSRS,lambdas,n0, avg_it=1)