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
from controllers import Controller
from utils import *

class ParameterFinder():
    def __init__(self, inputs, actions, steer_prog, accel_prog, brake_prog):
        self.inputs = inputs
        self.actions = actions
        self.steer = steer_prog
        self.accel = accel_prog
        self.brake = brake_prog

    def find_distance_paras(self, sp0, sp1, sp2, spt, ap0, ap1, ap2, apt, api, apc, bp0, bp1, bp2, bpt):
        self.steer.update_parameters([sp0, sp1, sp2], spt)
        self.accel.update_parameters([ap0, ap1, ap2], apt, api, apc)
        self.brake.update_parameters([bp0, bp1, bp2], bpt)
        steer_acts = []
        accel_acts = []
        brake_acts = []
        for window_list in self.inputs:
            steer_acts.append(clip_to_range(self.steer.pid_execute(window_list), -1, 1))
            accel_acts.append(clip_to_range(self.accel.pid_execute(window_list), 0, 1))
            brake_acts.append(clip_to_range(self.brake.pid_execute(window_list), 0, 1))
        steer_diff = spatial.distance.euclidean(steer_acts, np.array(self.actions)[:, 0])
        accel_diff = spatial.distance.euclidean(accel_acts, np.array(self.actions)[:, 1])
        brake_diff = spatial.distance.euclidean(brake_acts, np.array(self.actions)[:, 2])
        diff_total = -(steer_diff + accel_diff + brake_diff)/float(len(self.actions))
        return diff_total

    def pid_parameters(self, info_list):
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10}  # Optimizer configuration
        logging.info('Optimizing Controller')
        bo_pid = BayesianOptimization(self.find_distance_paras,
                                        {'sp0': info_list[0][0], 'sp1': info_list[0][1], 'sp2': info_list[0][2], 'spt': info_list[0][3],
                                         'ap0': info_list[1][0], 'ap1': info_list[1][1], 'ap2': info_list[1][2], 'apt': info_list[1][3], 'api': info_list[1][4], 'apc': info_list[1][5],
                                         'bp0': info_list[2][0], 'bp1': info_list[2][1], 'bp2': info_list[2][2], 'bpt': info_list[2][3]}, verbose=0)
        bo_pid.maximize(init_points=50, n_iter=100, kappa=5, **gp_params)
        return bo_pid.res['max']


def programmatic_game(steer, accel, brake, track_name='practgt2.xml'):
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
        window_list = [tempObs[:] for _ in range(window)]

        for j in range(max_steps):
            steer_action = clip_to_range(steer.pid_execute(window_list), -1, 1)
            accel_action = clip_to_range(accel.pid_execute(window_list), 0, 1)
            brake_action = clip_to_range(brake.pid_execute(window_list), 0, 1)
            action_prior = [steer_action, accel_action, brake_action]

            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
            window_list.pop(0)
            window_list.append(tempObs[:])

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

    programmatic_game(steer_prog, accel_prog, brake_prog, track_name=track_name)

    nn_agent = NeuralAgent(track_name=track_name)
    all_observations = []
    all_actions = []
    for i_iter in range(6):
        logging.info("Iteration {}".format(i_iter))
        # Learn/Update Neural Policy
        if i_iter == 0:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=200)
        else:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=100)

        # Collect Trajectories
        observation_list, action_list = nn_agent.collect_data([steer_prog, accel_prog, brake_prog])
        all_observations += observation_list
        # Relabel Observations
        all_actions = nn_agent.label_data([steer_prog, accel_prog, brake_prog], all_observations)

        # Learn new programmatic policy
        param_finder = ParameterFinder(all_observations, all_actions, steer_prog, accel_prog, brake_prog)

        steer_ranges = [tuple([create_interval(steer_prog.pid_info[0][const], 0.05) for const in range(3)]), create_interval(steer_prog.pid_info[1], 0.01)]
        accel_ranges = [tuple([create_interval(accel_prog.pid_info[0][const], 0.05) for const in range(3)]), create_interval(accel_prog.pid_info[1], 0.5), create_interval(accel_prog.pid_info[2], 0.1), create_interval(accel_prog.pid_info[3], 0.01)]
        brake_ranges = [tuple([create_interval(brake_prog.pid_info[0][const], 0.05) for const in range(3)]), create_interval(brake_prog.pid_info[1], 0.001)]
        pid_ranges = [steer_ranges, accel_ranges, brake_ranges]
        new_paras = param_finder.pid_parameters(pid_ranges)

        steer_prog.update_parameters([new_paras['max_params'][i] for i in ['sp0', 'sp1', 'sp2']], new_paras['max_params']['spt'])
        accel_prog.update_parameters([new_paras['max_params'][i] for i in ['ap0', 'ap1', 'ap2']], new_paras['max_params']['apt'], new_paras['max_params']['api'], new_paras['max_params']['apc'])
        brake_prog.update_parameters([new_paras['max_params'][i] for i in ['bp0', 'bp1', 'bp2']], new_paras['max_params']['bpt'])

        programmatic_game(steer_prog, accel_prog, brake_prog)

    logging.info("Steering Controller" + str(steer_prog.pid_info()))
    logging.info("Acceleration Controller" + str(accel_prog.pid_info()))
    logging.info("Brake Controller" + str(brake_prog.pid_info()))

    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trackfile', default='practgt2.xml')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--logname', default='AdaptiveProgramIPPG_')
    args = parser.parse_args()

    random.seed(args.seed)
    logPath = 'logs'
    logFileName = args.logname + args.trackfile[:-4]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, logFileName)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    learn_policy(track_name=args.trackfile)
