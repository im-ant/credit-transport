# ============================================================================
# "Long Arms" environment, toy example to evaluate forward information and
# backward value transport
#
# NOTE:
#   - Should include in the future the higher level dependencies as in
#     https://github.com/openai/gym/blob/master/docs/creating-environments.md
#
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


class LogicalArmsEnv(gym.Env):
    """
    Description:
        The agent starts and is views two successive images before needing
        to make a decision. If the two images are the same, then the agent
        should go left (action 0), else it should go right (action 1) to
        receive a +1 reward. Incorrect decision results in -1 reward.

    Action: 0, 1

    """

    def __init__(self, num_arms=2,
                 corridor_length=5,
                 require_final_action=False,
                 img_size=(32, 32),
                 grayscale=True,
                 flatten_obs=False,
                 scale_observation=False,
                 training=True,
                 dataset_path='./cifar_data_tmp/'):

        # ==
        # Attributes
        self.corridor_length = corridor_length
        self.num_arms = num_arms
        self.require_final_action = require_final_action  # dummy var

        self.img_size = img_size
        self.grayscale = grayscale
        self.flatten_obs = flatten_obs
        self.scale_observation = scale_observation

        self.training = training

        # ==
        # Get dataset of images
        self.ds_dict = self._init_img_dataset(dataset_path)

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(self.num_arms)
        self.observation_space = self._init_obs_space()

        # ==
        # Initialize
        # State = (arm #, arm steps)
        self.state = (0, 0)
        self.prev_img = None

    def _init_img_dataset(self, dataset_path):
        """
        Initialize image dataset corresponding to each observaiton

        :param dataset_path:
        :return: dict containing the initial, final and each arm's entrance
                 image observation
                    {'initial': PIL.Image.Image,
                     '1': PIL.Image.Image,
                     '2': PIL.Image.Image,
                     ...,
                     'final': PIL.Image.Image}
                 torch.utils.data.dataset.Subset object of corridor images
                 to be sampled. Each entry contains (PIL.Image.Image, label)
        """

        # Define the image classes of each state
        init_class = 'automobile'
        logic_class = 'dog'
        corridor_class = 'bird'
        final_class = 'deer'

        # ==
        # Download / initialize dataset
        ds = CIFAR10(dataset_path, train=self.training,
                     download=True)

        # ==
        # Separate the indeces for the different classes
        init_idxs, logic_idxs, corri_idxs, final_idxs = [], [], [], []
        (init_idx, logic_idx, corri_idx, final_idx) = (
            ds.class_to_idx[init_class], ds.class_to_idx[logic_class],
            ds.class_to_idx[corridor_class], ds.class_to_idx[final_class]
        )

        for i in range(len(ds)):
            current_class = ds[i][1]
            if current_class == init_idx:
                init_idxs.append(i)
            elif current_class == logic_idx:
                logic_idxs.append(i)
            elif current_class == corri_idx:
                corri_idxs.append(i)
            elif current_class == final_idx:
                final_idxs.append(i)
            else:
                pass

        # ==
        # Construct random datasets for the states
        ds_dict = {
            'initial': Subset(ds, init_idxs),
            'logic': Subset(ds, logic_idxs),
            'corridor': Subset(ds, corri_idxs),
            'final': Subset(ds, final_idxs)
        }

        return ds_dict

    def _init_obs_space(self):
        """
        Initialize the observation space
        :return: gym.spaces observation space
        """

        # Get observation shape
        example_obs = self._process_img(self.ds_dict['initial'][0][0])
        obs_shape = np.shape(example_obs)

        # Get range
        if self.scale_observation:
            obs_low = 0
            obs_high = 1
        else:
            obs_low = 0
            obs_high = 255

        # Construct space
        obs_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32
        )

        return obs_space

    def _process_img(self, img):
        """
        Transform an PIL image to the correct observation space
        :param img: a single PIL.Image.Image object
        :return: numpy array of transformed image
        """

        # ==
        # Construct transforms
        trans_list = [transforms.Resize(self.img_size)]
        if self.grayscale:
            trans_list += [transforms.Grayscale(num_output_channels=1)]

        img_transforms = transforms.Compose(trans_list)

        # ==
        # Transform and output
        img = img_transforms(img)
        obs = np.array(img, dtype=np.float32)

        # Ensure channel is in first dimension (torch conv standard)
        if len(np.shape(obs)) == 2:
            obs = np.expand_dims(obs, axis=0)
        elif len(np.shape(obs)) == 3:
            # PIL have channel on dim 2, swap with dim 0
            obs = np.swapaxes(obs, 2, 0)
            pass
        else:
            raise RuntimeError

        # Scale values to [0, 1]
        if self.scale_observation:
            obs = obs / 255.0

        # (Optinal) Flatten to vector
        if self.flatten_obs:
            obs = obs.flatten()

        return obs

    def step(self, action):
        """

        :param action:
        :return:
        """

        # TODO assert action range?

        # ==
        # Update underlying state
        cur_arm, cur_step = self.state
        # If at the very first state, randomly go to one of the tests
        if (cur_arm == 0) and (cur_step == 0):
            # Randomly transition to 1 or 2
            arm_num = np.random.choice([1, 2])
            self.state = (arm_num, -1)
        # In the first logical test state
        elif cur_step == -1:
            self.state = (cur_arm, 0)
        # In the second logical test state
        elif cur_step == 0:
            if cur_arm == (action + 1):
                self.state = (1, 1)
            else:
                self.state = (2, 1)
        # In the hallway state or terminal state
        elif cur_step >= 1:
            # If in hallway, increment
            if cur_step <= (self.corridor_length + 1):
                self.state = (cur_arm, cur_step + 1)
            else:
                pass
        else:
            raise NotImplementedError

        # ==
        # Generate observation, reward and done
        reward = 0.0
        done = False

        # Initial state
        if self.state[0] == 0:
            rand_idx = np.random.randint(low=0,
                                         high=len(self.ds_dict['initial']))
            img = self.ds_dict['initial'][rand_idx][0]
        # First logic state
        elif self.state[1] == -1:
            rand_idx = np.random.randint(low=0,
                                         high=len(self.ds_dict['logic']))
            img = self.ds_dict['logic'][rand_idx][0]
        # Second logic state
        elif self.state[1] == 0:
            if self.state[0] == 1:
                img = self.prev_img
            else:
                rand_idx = np.random.randint(low=0,
                                             high=len(self.ds_dict['logic']))
                img = self.ds_dict['logic'][rand_idx][0]
        elif self.state[1] >= 1:
            # in corridor
            if self.state[1] <= self.corridor_length:
                rand_idx = np.random.randint(low=0,
                                             high=len(self.ds_dict['corridor']))
                img = self.ds_dict['corridor'][rand_idx][0]
            elif self.state[1] == (self.corridor_length + 1):
                rand_idx = np.random.randint(low=0,
                                             high=len(self.ds_dict['final']))
                img = self.ds_dict['final'][rand_idx][0]
            elif self.state[1] == (self.corridor_length + 2):
                img = self.prev_img
                done = True
                if self.state[0] == 1:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                img = self.prev_img
                done = True
        else:
            raise NotImplementedError

        # Process image
        obs = self._process_img(img)

        self.prev_img = img
        return obs, reward, done, {}

    def reset(self):
        # Reset state
        self.state = (0, 0)

        # ==
        # Generate observation
        rand_idx = np.random.randint(low=0,
                                     high=len(self.ds_dict['initial']))
        init_img = self.ds_dict['initial'][rand_idx][0]
        init_obs = self._process_img(init_img)

        self.prev_img = init_img
        return init_obs

    def render(self):
        """
        Output a render-able RGB image
        :return: np array , shape (C, H, W) in range [0, 255]
        """
        np_img = np.array(self.prev_img, dtype=np.uint8)
        np_img = np.swapaxes(np_img, 0, 2)
        return np_img

    def close(self):
        pass


if __name__ == '__main__':
    # FOR TESTING ONLY
    print('hello')

    seed = 5
    print('numpy seed:', seed)
    np.random.seed(seed)

    env = LogicalArmsEnv(num_arms=2,
                         corridor_length=4,
                         require_final_action=True,
                         img_size=(20, 20),
                         grayscale=True,
                         training_data=True,
                         flatten_obs=True)

    print('=== set-up ===')
    print(env)
    print(env.action_space)
    print(env.observation_space)

    print('=== start ===')
    cur_obs = env.reset()
    print(env.state, np.shape(cur_obs),
          '[', np.mean(cur_obs), ']'
                                 '(', np.min(cur_obs), np.max(cur_obs), ')')

    for step in range(10):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)
        tmp_rend = env.render()

        print(action, env.state, np.shape(cur_obs),
              '[', np.mean(cur_obs), ']',
              '(', np.min(cur_obs), np.max(cur_obs), ')',
              reward, done)

    print('\n\n=== again ===')
    cur_obs = env.reset()
    print(env.state, np.shape(cur_obs),
          '[', np.mean(cur_obs), ']'
                                 '(', np.min(cur_obs), np.max(cur_obs), ')')

    for step in range(10):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)
        tmp_rend = env.render()

        print(action, env.state, np.shape(cur_obs),
              '[', np.mean(cur_obs), ']',
              '(', np.min(cur_obs), np.max(cur_obs), ')',
              reward, done)
