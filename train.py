from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

# Prevent numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
import gym
import numpy as np

import chainerrl
from chainerrl import experiments
import chainer.functions as F
import chainer.links as L
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import v_functions

from chainerrl.wrappers import atari_wrappers

import cv2
import math
from tqdm import tqdm
from actions import action_cr_l, action_cr_r, action_scl
from rewards import BDW, SquareBlackPadding
import selfplay_a3c
import selfplay_train_agent_async


class A3CLSTMSoftmax(chainer.Chain, selfplay_a3c.A3CModel, chainerrl.recurrent.RecurrentChainMixin):
    def __init__(self, batchsize, action_num):
        super(A3CLSTMSoftmax, self).__init__()
        with self.init_scope():
            self.batchsize = batchsize
            self.conv1 = L.Convolution2D(6, 8, ksize=3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(8, 8, ksize=3, stride=1, pad=1)
            self.conv3 = L.Convolution2D(8, 16, ksize=3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(16, 16, ksize=3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(16, 16, ksize=3, stride=1, pad=1)
            self.conv6 = L.Convolution2D(16, 16, ksize=3, stride=1, pad=1)

            self.bn1 = L.BatchNormalization(8)
            self.bn2 = L.BatchNormalization(16)
            self.bn3 = L.BatchNormalization(16)
            self.bn4 = L.BatchNormalization(16)
            self.bn5 = L.BatchNormalization(16)
            self.bn6 = L.BatchNormalization(1024)
            self.bn7 = L.BatchNormalization(1024)
            self.bn8 = L.BatchNormalization(1024)

            self.l7 = L.Linear(25620, 1024)
            self.l8 = L.Linear(1024, 1024)
            self.l9 = L.Linear(1024, 1024)
            self.lstm = L.LSTM(1024, 1024)
            self.pi = chainerrl.policies.SoftmaxPolicy(L.Linear(1024, action_num), min_prob=0.05)
            self.v = L.Linear(1024, 1)

    def pi_and_v(self, state):
        observations, opponent_observations, step_vector = state
        observations = observations.reshape((self.batchsize, 6, 40, 40))
        opponent_observations = opponent_observations.reshape((self.batchsize, 6, 40, 40))
        step_vector = step_vector.reshape((self.batchsize, 20))

        '''retarget'''
        h1 = F.relu(self.conv1(observations))
        h2 = F.relu(self.bn1(self.conv2(h1)))
        h3 = F.relu(self.bn2(self.conv3(h2)))
        h4 = F.relu(self.bn3(self.conv4(h3)))
        h5 = F.relu(self.bn4(self.conv5(h4)))
        h6 = F.relu(self.bn5(self.conv6(h5)))

        h6 = h6.reshape((self.batchsize, 25600))
        h7 = chainer.functions.concat([h6, step_vector], axis=1)
        h8 = F.relu(self.bn6(self.l7(h7)))
        h9 = F.relu(self.bn7(self.l8(h8)))
        h10 = F.relu(self.bn8(self.l9(h9)))
        h11 = self.lstm(h10)

        pout = self.pi(h11)
        vout = self.v(h11)

        '''opponent'''
        o_h1 = F.relu(self.conv1(opponent_observations))
        o_h2 = F.relu(self.bn1(self.conv2(o_h1)))
        o_h3 = F.relu(self.bn2(self.conv3(o_h2)))
        o_h4 = F.relu(self.bn3(self.conv4(o_h3)))
        o_h5 = F.relu(self.bn4(self.conv5(o_h4)))
        o_h6 = F.relu(self.bn5(self.conv6(o_h5)))

        o_h6 = o_h6.reshape((self.batchsize, 25600))
        o_h7 = chainer.functions.concat([o_h6, step_vector], axis=1)
        o_h8 = F.relu(self.bn6(self.l7(o_h7)))
        o_h9 = F.relu(self.bn7(self.l8(o_h8)))
        o_h10 = F.relu(self.bn8(self.l9(o_h9)))
        o_h11 = self.lstm(o_h10)

        opponent_pout = self.pi(o_h11)

        return pout, vout, opponent_pout


class SelfPlayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, batchsize, target_rate, rate, train_data, valid_data, test, action_num):
        super().__init__()
        self.action_space = gym.spaces.Discrete(action_num)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(batchsize, 6, 40, 40)
        )
        self.batchsize = batchsize
        self.target_rate = target_rate
        self.rate = rate
        self.test = test
        self.train_data = train_data
        self.valid_data = valid_data
        self.episode = 0
        self.reset()

    def reset(self):
        self.original_images = []
        self.retargeted_images = []
        self.opponent_images = []
        self.original_images_40pix = []
        self.retargeted_images_40pix = []
        self.opponent_images_40pix = []
        self.scale = []
        self.pix_array = []
        for i in range(self.batchsize):
            if(self.test):
                original = self.valid_data[(i + self.episode * self.batchsize) % len(self.valid_data)]
                self.target_rate = 0.75
            else:
                original = self.train_data[np.random.randint(len(self.train_data))]
            opponent = np.copy(original)
            retargeted = np.copy(original)
            pix = int(original.shape[1] * self.rate)

            self.original_images.append(original)
            self.retargeted_images.append(retargeted)
            self.opponent_images.append(opponent)

            height, width = original.shape[:2]
            scale = 40 / width
            self.scale.append(scale)
            original_40pix = cv2.resize(original, (40, int(height * scale)), interpolation=cv2.INTER_AREA)
            self.original_images_40pix.append(original_40pix)
            self.retargeted_images_40pix.append(original_40pix)
            self.opponent_images_40pix.append(original_40pix)

            self.pix_array.append(pix)

        self.done = False
        self.steps = 0
        self.step_vector = np.ones((self.batchsize, 20))
        if self.target_rate is None:
            self.target_rate = np.random.randint(10, 20) * 0.05
        self.target_steps = math.ceil(round((1-self.target_rate)/self.rate, 3))
        self.step_vector[:, :-self.target_steps] = 0
        self.episode += 1

        observation = self.observe()
        
        return observation

    def step(self, actions, pout, opponent_pout):

        opponent_actions = opponent_pout.sample().array

        for i in range(self.batchsize):
            action = actions[i]
            if action == 0:
                self.retargeted_images[i] = action_cr_l(self.retargeted_images[i], self.pix_array[i])
            elif action == 1:
                self.retargeted_images[i] = action_cr_r(self.retargeted_images[i], self.pix_array[i])
            elif action == 2:
                self.retargeted_images[i] = action_scl(self.retargeted_images[i], self.pix_array[i])
            # elif action == 3:
            #     self.retargeted_images[i] = action_sc(self.retargeted_images[i], self.pix_array[i])
            
            self.retargeted_images_40pix[i] = cv2.resize(self.retargeted_images[i], (int(self.retargeted_images[i].shape[1] * self.scale[i]), int(self.original_images[i].shape[0] * self.scale[i])), interpolation=cv2.INTER_AREA)

            opponent_action = opponent_actions[i]
            if opponent_action == 0:
                self.opponent_images[i] = action_cr_l(self.opponent_images[i], self.pix_array[i])
            elif opponent_action == 1:
                self.opponent_images[i] = action_cr_r(self.opponent_images[i], self.pix_array[i])
            elif opponent_action == 2:
                self.opponent_images[i] = action_scl(self.opponent_images[i], self.pix_array[i])
            # elif opponent_action == 3:
            #     self.opponent_images[i] = action_sc(self.opponent_images[i], self.pix_array[i])
            
            self.opponent_images_40pix[i] = cv2.resize(self.opponent_images[i], (int(self.opponent_images[i].shape[1] * self.scale[i]), int(self.original_images[i].shape[0] * self.scale[i])), interpolation=cv2.INTER_AREA)

        observation = self.observe()
        
        self.steps += 1
        self.step_vector[:, :-(self.target_steps-self.steps)] = 0
        self.done = self.is_done()
        reward = self.get_reward()
        
        return observation, reward, self.done, {}

    def render(self):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        pass

    def get_reward(self):
        reward = np.zeros(self.batchsize)
        for i in range(self.batchsize):

            if(self.test):
                if(self.done):
                    retargeted_score = - BDW(self.original_images_40pix[i], self.retargeted_images_40pix[i])
                    reward[i] = - retargeted_score
                else:
                    reward[i] = 0
            
            else:
                if(self.done):
                    retargeted_score = - BDW(self.original_images_40pix[i], self.retargeted_images_40pix[i])
                    opponent_score = - BDW(self.original_images_40pix[i], self.opponent_images_40pix[i])
                    if retargeted_score > opponent_score:
                        reward[i] = 1
                    elif retargeted_score == opponent_score:
                        reward[i] = -1
                    else:
                        reward[i] = -1
                else:
                    reward[i] = 0

        return reward
        
    def is_done(self):
        if (self.steps >= self.target_steps):
            return True
        else:
            return False
    
    def observe(self):
        observations = np.zeros((self.batchsize, 6, 40, 40), dtype=np.uint8)
        opponent_observations = np.zeros((self.batchsize, 6, 40, 40), dtype=np.uint8)

        for i in range(self.batchsize):
            original = SquareBlackPadding(self.original_images_40pix[i])
            retargeted = SquareBlackPadding(self.retargeted_images_40pix[i], self.original_images_40pix[i])
            opponent = SquareBlackPadding(self.opponent_images_40pix[i], self.original_images_40pix[i])

            observation = np.concatenate([original.transpose(2, 0, 1), retargeted.transpose(2, 0, 1)])
            observations[i, :, :, :] = observation

            opponent_observation = np.concatenate([original.transpose(2, 0, 1), opponent.transpose(2, 0, 1)])
            opponent_observations[i, :, :, :] = opponent_observation
        
        return (observations, opponent_observations, self.step_vector)


def main():

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    dataset_train = chainer.datasets.ImageDataset(args.train_dataset)
    dataset_valid = chainer.datasets.ImageDataset(args.valid_dataset)

    def transform(data):
        return cv2.cvtColor(data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR).astype(np.uint8)
    
    dataset_train = chainer.datasets.TransformDataset(dataset_train, transform)
    dataset_valid = chainer.datasets.TransformDataset(dataset_valid, transform)

    train_data = []
    for i in tqdm(range(len(dataset_train))):
        height, width = dataset_train[i].shape[:2]
        if height <= width:
            train_data.append(dataset_train[i])
    print("The number of images in train dataset is {}".format(len(train_data)))

    valid_data = []
    for i in tqdm(range(len(dataset_valid))):
        height, width = dataset_valid[i].shape[:2]
        if height <= width:
            valid_data.append(dataset_valid[i])
    print("The number of images in validation dataset is {}".format(len(valid_data)))

    import logging
    logging.basicConfig(level=args.logging_level)

    model = A3CLSTMSoftmax(args.batchsize, args.action_num)

    # Draw the computational graph and save it in the output directory.
    fake_obs = (chainer.Variable(np.zeros((args.batchsize, 6, 40, 40), dtype=np.float32)[None],name='observation'), chainer.Variable(np.zeros((args.batchsize, 6, 40, 40), dtype=np.float32)[None],name='observation'), chainer.Variable(np.zeros((args.batchsize, 20), dtype=np.float32)[None]))
    with chainerrl.recurrent.state_reset(model):
        # The state of the model is reset again after drawing the graph
        chainerrl.misc.draw_computational_graph(
            [model(fake_obs)],
            os.path.join(args.outdir, 'model'))

    opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    def phi(x):
        # Feature extractor
        observations, opponent_observations, step_vector = x
        return (np.asarray(observations, dtype=np.float32)/255, np.asarray(opponent_observations, dtype=np.float32)/255, np.asarray(step_vector, dtype=np.float32))

    agent = selfplay_a3c.SelfPlay_A3C(model, opt, t_max=args.t_max, gamma=0.99, batchsize=args.batchsize, action_num=args.action_num, beta=args.beta, phi=phi)

    if args.load:
        agent.load(args.load)

    def make_env(process_idx, test, action_num):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = SelfPlayEnv(args.batchsize, None, args.rate, train_data, valid_data, test, action_num)
        env.seed(int(env_seed))
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    if args.demo:
        env = make_env(0, True, args.action_num)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.lr = value

        lr_decay_hook = experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)

        def step_hook(env, agent, step):
            if step%1000 == 0:
                agent.save(args.outdir + '/' + str(step))

        selfplay_train_agent_async.selfplay_train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            global_step_hooks=[lr_decay_hook, step_hook],
            save_best_so_far_agent=False,
            action_num=args.action_num,
        )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--action_num', type=int, default=3)
    parser.add_argument('--target_rate', type=float, default=0.75)
    parser.add_argument('--rate', type=float, default=0.025)
    parser.add_argument('--train_dataset', type=str, default='dataset/MIRFLICKR_train.txt')
    parser.add_argument('--valid_dataset', type=str, default='dataset/MIRFLICKR_valid.txt')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--outdir', type=str, default='models',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--t-max', type=int, default=20)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--eval-n-steps', type=int, default=150)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    args = parser.parse_args()

    main()
 