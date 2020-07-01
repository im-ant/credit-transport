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
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms


class LongArmsEnv(gym.Env):
    """
    Description:
        The agent needs to pick between multiple arms, each with different
        rewards. The arm leads to a long hallway of partially observable
        timesteps before the reward is given.

        NOTE for now there are only two arms.

    Observation:

    Actions:

    States:
        Fully observable states describle by (arm #, arm steps), where (0,0)
        is the initial state (not in any arm). For example, (1,2) means the
        agent is twp states into the 1st arm. The agent receives image
        observations.

    """

    def __init__(self, corridor_length=5,
                 img_size=(32, 32), grayscale=True, flatten_obs=True,
                 dataset_path='./cifar_data_tmp/'):

        # ==
        # Attributes
        self.corridor_length = corridor_length
        self.num_arms = 2

        self.img_size = img_size
        self.grayscale = grayscale
        self.flatten_obs = flatten_obs

        # ==
        # Initialize spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = None  # TODO depends on image

        # ==
        # Get dataset of images
        dataset_tup = self._init_img_dataset(dataset_path)
        self.img_dict = dataset_tup[0]
        self.corridor_ds = dataset_tup[1]

        # ==
        # Initialize
        # State = (arm #, arm steps)
        self.state = (0, 0)

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
        arm_class = 'dog'
        corridor_class = 'bird'
        final_class = 'cat'

        # ==
        # Download / initialize dataset
        ds = CIFAR10(dataset_path, train=True, download=True)

        # ==
        # Separate the indeces for the different classes
        init_idxs, arm_idxs, corri_idxs, final_idxs = [], [], [], []
        (init_idx, arm_idx, corri_idx, final_idx) = (
            ds.class_to_idx[init_class], ds.class_to_idx[arm_class],
            ds.class_to_idx[corridor_class], ds.class_to_idx[final_class]
        )

        for i in range(len(ds)):
            current_class = ds[i][1]
            if current_class == init_idx:
                init_idxs.append(i)
            elif current_class == arm_idx:
                arm_idxs.append(i)
            elif current_class == corri_idx:
                corri_idxs.append(i)
            elif current_class == final_idx:
                final_idxs.append(i)
            else:
                pass

        # ==
        # Pick an image sample for the init, final and each arm states
        # Each sample has format: (PIL.Image.Image, class #)
        init_sample = ds[init_idxs[4]]
        final_sample = ds[final_idxs[64]]
        arm_samples = []
        for arm_j in range(self.num_arms):
            cur_arm_sample = ds[arm_idxs[2 * arm_j]]
            arm_samples.append(cur_arm_sample)

        # Save samples as PIL images
        img_obs_dict = {
            'initial': init_sample[0],
            'final': final_sample[0],
        }
        for arm_j in range(self.num_arms):
            img_obs_dict[str((arm_j + 1))] = arm_samples[arm_j][0]

        # ==
        # Set up the image sampler for the corridor
        corridor_ds = Subset(ds, corri_idxs)

        return img_obs_dict, corridor_ds

    def _init_obs_space(self):
        """
        Initialize the observation space
        :return: gym.spaces observation space
        """

        # TODO THIS

        pass

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

        img = np.array(img)
        if self.flatten_obs:
            img = img.flatten()

        return img

    def step(self, action):
        """

        :param action:
        :return:
        """

        # TODO assert action range?

        # ==
        # Update underlying state
        cur_arm, cur_step = self.state
        if cur_arm == 0:
            self.state = ((action + 1), 1)
        elif cur_step >= (self.corridor_length + 3):
            pass
        else:
            self.state = (cur_arm, cur_step + 1)

        # ==
        # Generate observation, reward and done
        obs = None
        reward = 0.0
        done = False

        # Get boolean
        is_after_corri = (self.state[1] == (self.corridor_length + 2))
        is_final_state = (self.state[1] >= (self.corridor_length + 3))

        # If in the first arm states
        # (assume agent can never "step" to the starting state)
        if self.state[1] == 1:
            img = self.img_dict[str(self.state[0])]

        # If in the state after corridor
        elif is_after_corri:
            img = self.img_dict['final']

        # If in the terminal states
        elif is_final_state:
            img = self.img_dict['final']
            done = True
            if self.state[0] == 1:
                reward = 1.0
            elif self.state[0] == 2:
                reward = -1.0

        # If in the corridor states
        else:
            img = next(iter(self.corridor_ds))[0]

        # Process image
        obs = self._process_img(img)

        return obs, reward, done, {}

    def reset(self):
        # Reset state
        self.state = (0, 0)

        # ==
        # Generate observation
        init_img = self.img_dict['initial']
        init_obs = self._process_img(init_img)

        return init_obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # FOR TESTING ONLY
    print('hello')

    env = LongArmsEnv()
    print(env)

    obs = env.reset()
    print(env.state, np.shape(obs), '[', np.min(obs), np.max(obs), ']')

    for step in range(10):

        obs, reward, done, info = env.step(0)

        print(env.state, np.shape(obs), '[', np.min(obs), np.max(obs), ']', reward, done)



