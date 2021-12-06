import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import os

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
base_dir = 'contents/0_GYM_Learn/'



class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.name = name
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min

    def set_position(self, x, y):
        # 减去图标的宽度
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, value, min_value, max_value):
        '''
        取值限制在上下限之间
        '''
        return max(min(value, max_value), min_value)


class Chopper(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Chopper, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread(os.path.join(base_dir,'choper_icons/chopper.png')) / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_w, self.icon_h))


class Bird(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Bird, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread(os.path.join(base_dir,'choper_icons/bird.png')) / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_w, self.icon_h))


class Fuel(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Fuel, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread(os.path.join(base_dir,'choper_icons/fuel.png')) / 255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_w, self.icon_h))


class ChoppeScape(Env):
    def __init__(self) -> None:
        super(ChoppeScape, self).__init__()

        # define a 2D observation space
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low=np.zeros(
            self.observation_shape), high=np.ones(self.observation_shape), dtype=np.float16)

        # define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(6,)

        # create a canvas to reder the environment image upon
        self.canvas = np.ones((600, 800, 3)) * 1

        # Define elements present inside the environment
        self.elements = []

        # Maximun fuel chopper can take at once
        self.max_fuel = 1000

        # Permissible area of helicopter to be
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_min = 0
        self.x_max = int(self.observation_shape[1])

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw helicopter on the canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y:y+elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Fuel Left: {} | Rewaeds: {}'.format(
            self.fuel_left, self.ep_return)

        # Put the info on canvas
        self.canvas = cv2.putText(
            self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    def reset(self):
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return = 0

        # Number of birds
        self.bird_count = 0
        self.fuel_count = 0

        # Determine a place to intialise the chopper in
        # 设置初始化位置
        x = random.randrange(
            int(self.observation_shape[0]*0.05), int(self.observation_shape[0]*0.10))
        y = random.randrange(
            int(self.observation_shape[1]*0.15), int(self.observation_shape[1]*0.20))

        # Initialise the chopper
        self.chopper = Chopper('chopper', self.x_max,
                               self.x_min, self.y_max, self.y_min)
        self.chopper.set_position(x, y)

        # Initialise the elements
        self.elements = [self.chopper]

        # Reset the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the elements on the canvas
        self.draw_elements_on_canvas()

        return self.canvas

    def render(self, mode='human'):
        assert mode in [
            'human', 'rgb_array'], "Mode {} is not supported".format(mode)

        if mode == 'human':
            cv2.imshow('Game', self.canvas)
            cv2.waitKey(10)
        elif mode == 'rgb_array':
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        return {0: 'Right', 1: 'Left', 2: 'Down', 3: 'Up', 4: 'Do Nothing'}

    def step(self, action):
        # Mark the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(
            action), 'Invalid action: {}'.format(action)

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step
        reward = 1

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(0, 5)
        elif action == 1:
            self.chopper.move(0, -5)
        elif action == 2:
            self.chopper.move(-5, 0)
        elif action == 3:
            self.chopper.move(5, 0)
        elif action == 4:
            self.chopper.move(0, 0)

        # Spawn a bird at the right edge with prob 0.01
        if random.random() < 0.01:

            # Spawn a bird
            spawned_bird = Bird('bird_{}'.format(
                self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.bird_count += 1

            # Spawn the bird at the right edge
            bird_x = self.x_max
            bird_y = random.randrange(self.y_min, self.y_max)
            spawned_bird.set_position(bird_x, bird_y)

            # Append the spawned bird to the elements
            self.elements.append(spawned_bird)

        if random.random() < 0.01:
            # Spawn a fuel
            spawned_fuel = Fuel('fuel_{}'.format(
                self.fuel_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.fuel_count += 1

            # Spawn the fuel at the right edge
            fuel_x = random.randrange(self.x_min, self.x_max)
            fuel_y = self.y_max
            spawned_fuel.set_position(fuel_x, fuel_y)

            # Append the spawned fuel to the elements
            self.elements.append(spawned_fuel)

        for elem in self.elements:
            if isinstance(elem, Bird):
                # If the bird has reached the left edge
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                else:
                    elem.move(-5, 0)

                # if the bird has collided
                if self.has_collided(self.chopper, elem):
                    reward = -10
                    done = True
                    self.elements.remove(self.chopper)

            if isinstance(elem, Fuel):
                # if the fuel has reached the top edge
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    elem.move(0, -5)

                if self.has_collided(self.chopper, elem):
                    self.elements.remove(elem)
                    self.fuel_left += 10
        self.ep_return += 1
        self.draw_elements_on_canvas()
        if self.fuel_left <= 0:
            done = True

        return self.canvas, reward, done, []

    def has_collided(self, elem1, elem2):
        # Check if the chopper has collided with any of the elements
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        # 如果距离小于图片的宽度，则认为碰撞
        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True
        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False


if __name__ == '__main__':
    env = ChoppeScape()
    obs = env.reset()
    plt.imshow(obs)
    plt.pause(100)
