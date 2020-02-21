import logging
import sys
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
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import timeit
from numpy.random import choice
import statistics


class FunctionOU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)



def run_ddpg(amodel, cmodel, train_indicator=0, seeded=1337, track_name='practgt2.xml'):
    OU = FunctionOU()
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic
    ALPHA = 0.9

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # of sensors input

    np.random.seed(seeded)

    vision = False

    EXPLORE = 100000.
    if train_indicator:
        episode_count = 600
    else:
        episode_count = 3
    max_steps = 20000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False, track_name=track_name)

    if not train_indicator:
        # Now load the weight
        logging.info("Now we load the weight")
        try:
            actor.model.load_weights(amodel)
            critic.model.load_weights(cmodel)
            actor.target_model.load_weights(amodel)
            critic.target_model.load_weights(cmodel)
            logging.info(" Weight load successfully")
        except:
            logging.info("Cannot find the weight")
            exit()

    logging.info("TORCS Experiment Start.")
    best_lap = 500

    for i_episode in range(episode_count):
        logging.info("Episode : " + str(i_episode) + " Replay Buffer " + str(buff.count()))
        if np.mod(i_episode, 3) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack(
            (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

        total_reward = 0.

        for j_iter in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack(
                (ob.speedX, ob.angle, ob.trackPos, ob.speedY, ob.speedZ, ob.rpm, ob.wheelSpinVel / 100.0, ob.track))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            if np.mod(step, 1000) == 0:
                logging.info("Episode {}, Distance {}, Last Lap {}".format(
                    i_episode, ob.distRaced, ob.lastLapTime))
                if ob.lastLapTime > 0:
                    if best_lap < ob.lastLapTime:
                        best_lap = ob.lastLapTime

            step += 1
            if done:
                break

        if train_indicator and i_episode > 20:
            if np.mod(i_episode, 3) == 0:
                logging.info("Now we save model")
                actor.model.save_weights("ddpg_actor_weights_periodic.h5", overwrite=True)
                critic.model.save_weights("ddpg_critic_weights_periodic.h5", overwrite=True)
        logging.info("TOTAL REWARD @ " + str(i_episode) + "-th Episode  : Reward " + str(total_reward))
        logging.info("Best Lap {}".format(best_lap))
    env.end()  # This is for shutting down TORCS
    logging.info("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--picktrack', default='practgt2.xml')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--mode', default=1, type=int)  # 0 - Run, 1- Train
    parser.add_argument('--actormodel', default='a')
    parser.add_argument('--criticmodel', default='c')
    parser.add_argument('--logname', default='TorcsDDPG_')
    args = parser.parse_args()

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

    run_ddpg(args.actormodel, args.criticmodel, train_indicator=args.mode, seeded=args.seed, track_name=args.picktrack)
