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
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import functools
import copy
from utils import *


class FunctionOU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


class NeuralAgent():
    def __init__(self, track_name='practgt2.xml'):
        BUFFER_SIZE = 100000
        TAU = 0.001  # Target Network HyperParameters
        LRA = 0.0001  # Learning rate for Actor
        LRC = 0.001  # Lerning rate for Critic
        state_dim = 29  # of sensors input
        self.batch_size = 32
        self.lambda_mix = 10.0
        self.action_dim = 3  # Steering/Acceleration/Brake


        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(sess)

        self.actor = ActorNetwork(sess, state_dim, self.action_dim, self.batch_size, TAU, LRA)
        self.critic = CriticNetwork(sess, state_dim, self.action_dim, self.batch_size, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer
        self.track_name = track_name

    def update_neural(self, controllers, episode_count=200, tree=False):
        OU = FunctionOU()
        vision = False
        GAMMA = 0.99
        EXPLORE = 100000.
        max_steps = 10000
        reward = 0
        done = False
        step = 0
        epsilon = 1

        if not tree:
            steer_prog, accel_prog, brake_prog = controllers


        # Generate a Torcs environment
        env = TorcsEnv(vision=vision, throttle=True, gear_change=False, track_name=self.track_name)

        window = 5
        lambda_store = np.zeros((max_steps, 1))
        lambda_max = 40.
        factor = 0.8

        logging.info("TORCS Experiment Start with Lambda = " + str(self.lambda_mix))


        for i_episode in range(episode_count):
            logging.info("Episode : " + str(i_episode) + " Replay Buffer " + str(self.buff.count()))
            if np.mod(i_episode, 3) == 0:
                ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
            else:
                ob = env.reset()

            #[ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, list(ob.wheelSpinVel / 100.0), list(ob.track)]
            s_t = np.hstack(
                (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

            total_reward = 0.
            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), [0, 0, 0]]
            window_list = [tempObs[:] for _ in range(window)]


            for j_iter in range(max_steps):
                if tree:
                    tree_obs = [sensor for obs in tempObs[:-1] for sensor in obs]
                    act_tree = controllers.predict([tree_obs])
                    steer_action = clip_to_range(act_tree[0][0], -1, 1)
                    accel_action = clip_to_range(act_tree[0][1], 0, 1)
                    brake_action = clip_to_range(act_tree[0][2], 0, 1)
                else:
                    steer_action = clip_to_range(steer_prog.pid_execute(window_list), -1, 1)
                    accel_action = clip_to_range(accel_prog.pid_execute(window_list), 0, 1)
                    brake_action = clip_to_range(brake_prog.pid_execute(window_list), 0, 1)
                action_prior = [steer_action, accel_action, brake_action]

                tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                           list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
                window_list.pop(0)
                window_list.append(tempObs[:])

                loss = 0
                epsilon -= 1.0 / EXPLORE
                a_t = np.zeros([1, self.action_dim])
                noise_t = np.zeros([1, self.action_dim])

                a_t_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                noise_t[0][0] = max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
                noise_t[0][1] = max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
                noise_t[0][2] = max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

                a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
                a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
                a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

                mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(3)]

                ob, r_t, done, info = env.step(mixed_act)

                s_t1 = np.hstack(
                    (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

                self.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

                # Do the batch update
                batch = self.buff.getBatch(self.batch_size)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])

                target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]

                loss += self.critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = self.actor.model.predict(states)
                grads = self.critic.gradients(states, a_for_grad)
                self.actor.train(states, grads)
                self.actor.target_train()
                self.critic.target_train()

                total_reward += r_t
                s_t = s_t1

                # Control prior mixing term
                if j_iter > 0 and i_episode > 50:
                    lambda_track = lambda_max * (1 - np.exp(-factor * np.abs(r_t + GAMMA * np.mean(target_q_values[-1] - base_q[-1]))))
                    lambda_track = np.squeeze(lambda_track)
                else:
                    lambda_track = 10.
                lambda_store[j_iter] = lambda_track
                base_q = copy.deepcopy(target_q_values)

                if np.mod(step, 2000) == 0:
                    logging.info("Episode " + str(i_episode) + " Distance " + str(ob.distRaced) + " Lap Times " + str(ob.lastLapTime))

                step += 1
                if done:
                    break
            else:
                env.end()
            self.lambda_mix = np.mean(lambda_store)
            logging.info("#### Episode Reward: " + str(total_reward))
            logging.info("####### Episode Length: " + str(j_iter))
            logging.info("########## Lap Times: " + str(ob.lastLapTime))
            logging.info("### Lambda Mix: " + str(self.lambda_mix))
            logging.info("#### Total Steps: " + str(step))
            logging.info("TOTAL REWARD @ " + str(i_episode) + "-th Episode  : Reward " + str(total_reward))
            logging.info("Total Step: " + str(j_iter) + "  Distance" + str(ob.distRaced))
        env.end()  # This is for shutting down TORCS
        logging.info("Finish.")
        return None

    def collect_data(self, controllers, tree=False):
        OU = FunctionOU()
        vision = False
        GAMMA = 0.99
        EXPLORE = 100000.
        max_steps = 10000
        reward = 0
        done = False
        step = 0
        epsilon = 1

        if not tree:
            steer_prog, accel_prog, brake_prog = controllers


        # Generate a Torcs environment
        env = TorcsEnv(vision=vision, throttle=True, gear_change=False, track_name=self.track_name)

        window = 5
        lambda_store = np.zeros((max_steps, 1))
        lambda_max = 40.
        factor = 0.8

        logging.info("TORCS Collection started with Lambda = " + str(self.lambda_mix))
        ob = env.reset(relaunch=True)

        s_t = np.hstack(
            (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

        total_reward = 0.
        tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                   list(ob.wheelSpinVel / 100.0), list(ob.track), [0, 0, 0]]
        window_list = [tempObs[:] for _ in range(window)]

        observation_list = []
        actions_list = []

        for j_iter in range(max_steps):
            if tree:
                tree_obs = [sensor for obs in tempObs[:-1] for sensor in obs]
                act_tree = controllers.predict([tree_obs])
                steer_action = clip_to_range(act_tree[0][0], -1, 1)
                accel_action = clip_to_range(act_tree[0][1], 0, 1)
                brake_action = clip_to_range(act_tree[0][2], 0, 1)
            else:
                steer_action = clip_to_range(steer_prog.pid_execute(window_list), -1, 1)
                accel_action = clip_to_range(accel_prog.pid_execute(window_list), 0, 1)
                brake_action = clip_to_range(brake_prog.pid_execute(window_list), 0, 1)

            action_prior = [steer_action, accel_action, brake_action]

            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
            window_list.pop(0)
            window_list.append(tempObs[:])

            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(3)]
            if tree:
                newobs = [item for sublist in tempObs[:-1] for item in sublist]
                observation_list.append(newobs[:])
            else:
                observation_list.append(window_list[:])
            actions_list.append(mixed_act[:])
            ob, r_t, done, info = env.step(mixed_act)

            s_t1 = np.hstack(
                (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

            self.buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = self.buff.getBatch(self.batch_size)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            total_reward += r_t
            s_t = s_t1
            if np.mod(step, 2000) == 0:
                logging.info(" Distance " + str(ob.distRaced) + " Lap Times " + str(ob.lastLapTime))

            step += 1
            if done:
                break
        env.end()

        return observation_list, actions_list

    def label_data(self, controllers, observation_list, tree=False):
        if not tree:
            steer_prog, accel_prog, brake_prog = controllers
        actions_list = []
        net_obs_list = []
        logging.info("Data labelling started with Lambda = " + str(self.lambda_mix))
        for window_list in observation_list:
            if tree:
                act_tree = controllers.predict([window_list])
                steer_action = clip_to_range(act_tree[0][0], -1, 1)
                accel_action = clip_to_range(act_tree[0][1], 0, 1)
                brake_action = clip_to_range(act_tree[0][2], 0, 1)
                net_obs_list.append(window_list)
            else:
                steer_action = clip_to_range(steer_prog.pid_execute(window_list), -1, 1)
                accel_action = clip_to_range(accel_prog.pid_execute(window_list), 0, 1)
                brake_action = clip_to_range(brake_prog.pid_execute(window_list), 0, 1)
                net_obs = [sensor for obs in window_list[-1] for sensor in obs]
                net_obs_list.append(net_obs[:29])

            action_prior = [steer_action, accel_action, brake_action]

            s_t = np.hstack([[net_obs[:29]]])
            a_t = self.actor.model.predict(s_t.reshape(1, 29))
            mixed_act = [a_t[0][k_iter] / (1 + self.lambda_mix) + (self.lambda_mix / (1 + self.lambda_mix)) * action_prior[k_iter] for k_iter in range(3)]

            actions_list.append(mixed_act[:])

        return net_obs_list, observation_list, actions_list

