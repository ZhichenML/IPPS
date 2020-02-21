import gym
from gym import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
import logging

#try
class TorcsEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False, track_name='practice.xml'):
        self.track_name = track_name
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        os.system('pkill torcs')
        time.sleep(0.5)
        logging.info('Init TORCS Environment')
        if self.vision is True:
            config_string = 'torcs -s -r /usr/local/share/games/torcs/config/raceman/' + self.track_name + ' -nofuel -nodamage -nolaptime -vision &'
            os.system(config_string)
        else:
            #config_string = 'torcs -s -r /usr/local/share/games/torcs/config/raceman/' + self.track_name + ' -T -nofuel &'
            os.system('torcs -s -r /usr/local/share/games/torcs/config/raceman/' + self.track_name + ' -T -nofuel -nodamage -nolaptime &')
            #os.system('torcs -T -nofuel &')

        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)
        #os.system(config_string)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box( np.array([-1, 0, 0]), np.array([1, 1, 1]), dtype=np.float32)


        high = np.array([np.inf for _ in range(30)])
        self.observation_space = spaces.Box(low=-high, high=high)

        #if vision is False:
        #    high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
        #    low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        #    self.observation_space = spaces.Box(low=low, high=high)
        #else:
        #    high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
        #    low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf


    def step(self, u):
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        client.R.d['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                client.R.d['accel'] -= .2
        else:
            client.R.d['accel'] = this_action['accel']
            client.R.d['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            client.R.d['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            client.R.d['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    client.R.d['gear'] = 2
                if client.S.d['speedX'] > 80:
                    client.R.d['gear'] = 3
                if client.S.d['speedX'] > 110:
                    client.R.d['gear'] = 4
                if client.S.d['speedX'] > 140:
                    client.R.d['gear'] = 5
                if client.S.d['speedX'] > 170:
                    client.R.d['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs,  zhichen: send R.d
        client.respond_to_server()
        # Get the response of TORCS,  zhichen: set S.d
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)
        observation_next = self.observation

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        #
        # Termination judgement #########################
        #

        episode_terminate = False

        if self.terminal_judge_start < self.time_step:
            if abs(track.any()) > 1 or abs(trackPos) > 2.7:  # Episode is terminated if the car is out of track
                print('Out of Track', track.any(), trackPos)
                reward = -200
                episode_terminate = True
                client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if sp < self.termination_limit_progress:
                print('No progress', progress, sp)
                reward = -100
                episode_terminate = True
                client.R.d['meta'] = True

        #if np.cos(obs['angle']) < -0.5: # Episode is terminated if the agent runs backward
        #    print('Running Backward')
        #    reward = -1
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            print('Running Backward')
            reward = -100
            episode_terminate = True
            client.R.d['meta'] = True


        info = {}
        info['distRaced'] = observation_next.distRaced
        info['distFromStart'] = observation_next.distFromStart
        info['lastLapTime'] = observation_next.lastLapTime
        info['curLapTime'] = observation_next.curLapTime
        info['speed'] = sp

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server() # zhichen: send meta signal

        self.time_step += 1

        return observation_next, reward, client.R.d['meta'], info #self.get_obs()

    # zhichen: relaunch torcs env and rebuild client-server connection
    def reset(self, relaunch=False):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                logging.info("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment

        # self.client.shutdown()
        self.client = snakeoil3.Client(p=3101, vision=self.vision, track_name=self.track_name) #   # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        logging.info('Successfully reset S')

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()


    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            config_string = 'torcs -s -r /usr/local/share/games/torcs/config/raceman/' + self.track_name + ' -nofuel -nodamage -nolaptime -vision &'
            os.system(config_string)
        else:
            os.system('torcs -s -r /usr/local/share/games/torcs/config/raceman/' + self.track_name + ' -T -nofuel -nodamage -nolaptime &')
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)
            #config_string = 'torcs -s -r /usr/local/share/games/torcs/config/raceman/' + self.track_name + ' -T -nofuel &'
            #os.system(config_string)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['speedX', 'speedY', 'speedZ', 'angle', 'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel', 'lastLapTime', 'curLapTime','distRaced', 'distFromStart']
                     #'distFromStartLine']
            Observation = col.namedtuple('Observaion', names)
            return Observation(speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               lastLapTime=np.array(raw_obs['lastLapTime'], dtype=np.float32)/1.0,
                               curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32)/1.0,
                               distRaced=np.array(raw_obs['distRaced'], dtype=np.float32)/1.0,
                               distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32)/1.0)
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
