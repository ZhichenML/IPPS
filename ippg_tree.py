from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import logging
import sys
import datetime
import time
import functools
import copy
from bayes_opt import BayesianOptimization
from scipy import spatial
from neural_update import NeuralAgent
from sklearn.tree import DecisionTreeRegressor
from controllers import Controller
from utils import *


def programmatic_game(tree_program, track_name='practgt2.xml'):
    episode_count = 2
    max_steps = 100000
    window = 5

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, track_name=track_name)

    logging.info("TORCS Experiment Start with Priors on " + track_name)
    for i_episode in range(episode_count):
        ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                   list(ob.wheelSpinVel / 100.0), list(ob.track), [0, 0, 0]]
        newobs = [item for sublist in tempObs[:-1] for item in sublist]

        for j in range(max_steps):
            act_tree = tree_program.predict([newobs])
            action_prior = [act_tree[0][0], act_tree[0][1], act_tree[0][2]]

            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
            newobs = [item for sublist in tempObs[:-1] for item in sublist]

            ob, r_t, done, info = env.step(action_prior)
            if np.mod(j, 1000) == 0:
                logging.info("Episode " + str(i_episode) + " Distance " + str(ob.distRaced) + " Lap Times " + str(ob.lastLapTime))

            if done:
                print('Done. Steps: ', j)
                break

        env.end()  # This is for shutting down TORCS
        logging.info("Finish.")



def learn_policy(track_name):

    # Define Pi_0
    steer_prog = Controller([0.97, 0.05, 49.98], 0, 2, 0)
    accel_prog = Controller([3.97, 0.01, 48.79], 0.30, 5, 0, 0.0, 0.01, 'obs[-1][2][0] > -self.para_condition and obs[-1][2][0] < self.para_condition')
    brake_prog = Controller([0, 0, 0], 0, 2, 0)

    nn_agent = NeuralAgent(track_name=track_name)
    all_observations = []
    all_actions = []
    for i_iter in range(6):
        logging.info("Iteration {}".format(i_iter))
        # Learn/Update Neural Policy
        if i_iter == 0:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=200)
            observation_list, action_list = nn_agent.collect_data([steer_prog, accel_prog, brake_prog])
            all_observations += observation_list
            all_observations, _, all_actions = nn_agent.label_data([steer_prog, accel_prog, brake_prog], all_observations)
        else:
            nn_agent.update_neural(tree_program, episode_count=100, tree=True)
            observation_list, action_list = nn_agent.collect_data(tree_program, tree=True)
            all_observations += observation_list
            _, _, all_actions = nn_agent.label_data(tree_program, all_observations, tree=True)

        # Learn new programmatic policy
        tree_program = DecisionTreeRegressor()
        tree_program.fit(all_observations, all_actions)
        programmatic_game(tree_program, track_name=track_name)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--picktrack', default='practgt2.xml')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--logname', default='AdaptiveProgramIPPG_')
    args = parser.parse_args()

    random.seed(args.seed)
    logPath = 'logs'
    logFileName = args.logname + args.picktrack[:-4]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, logFileName)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    learn_policy(track_name=args.picktrack)
