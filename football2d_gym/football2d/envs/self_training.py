import gym
from gym import spaces
import pygame
import numpy as np
import re
import string
from pymunk import Vec2d
from football2d.envs.ball import Ball
from football2d.envs.player import Player_v0, Player_v1, Player_v2
from pprint import pprint
from collections import OrderedDict
import ipdb

CENTER = Vec2d(600, 400)
PITCH_GREEN = (0, 128, 0)
LINE_WHITE = (220, 220, 220)
GOAL_WHITE = (180, 180, 180)
TIME_WHITE = (240, 240, 240)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (220, 220, 0)
SKYBLUE = (135, 206, 250)
LINE_WIDTH = 3
GOAL_WIDTH = 3

TIME_TEXT_SIZE = 25
GOAL_TEXT_SIZE = 40
REWARD_TEXT_SIZE = 20
STATE_ACTION_TEXT_SIZE = 18

timeDelta = 0.02
FIX_PRECISION = 1e-3

class SelfTraining_v0(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, learn_to_kick=False, time_limit=120, randomize_position=False,
                 ball_position=(0, 0), player_position=(-100, 0)):
        super().__init__()
        self.time_limit = time_limit
        self.window_width = 1200  # The size of the PyGame window
        self.window_height = 900  # The size of the PyGame window
        self.center_point = CENTER

        self.ball_position = ball_position
        self.player_position = player_position

        self.randomize_position = randomize_position
        self.learn_to_kick = learn_to_kick

        # Position of ball and player are relative to the center point
        if self.randomize_position:
            self.ball = Ball(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)))
            self.player = Player_v0(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)))
        else:
            self.ball = Ball(Vec2d(*self.ball_position), Vec2d.zero())
            self.player = Player_v0(Vec2d(*self.player_position))
        if self.learn_to_kick:
            self.player.position = self.ball.position

        self.time = 0
        self.accumulated_reward = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            OrderedDict({
                "ball_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "ball_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            })
        )

        # 4 continuous action space
        # first two is player moving acceleration, along x and y
        # second two is player kicking momentum, along x and y
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4, ), dtype=np.float32)
        #self.action_space = spaces.Dict(
        #    OrderedDict({
        #        "player_acceleration": spaces.Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32),
        #        "kick_momentum": spaces.Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32),
        #    })
        #)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.text_surfs = dict()
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
                "ball_position": self.ball.observe_position(),
                "ball_speed": self.ball.observe_speed(),
                "player_position": self.player.observe_position(),
                "player_speed": self.player.observe_speed(),
               }

    def _get_info(self):
        return {
                "distance_to_ball": self.ball.distance_to(self.player.position),
                "distance_to_goal": self.ball.distance_to_right_goal(),
                "kicked_ball": self.player.kicked_ball
               }

    def _get_obs_strs(self):
        observation = self._get_obs()
        if "ball_speed" in observation.keys():
            observation["ball_speed_value"] = np.linalg.norm(observation["ball_speed"], keepdims=True)
        if "player_speed" in observation.keys():
            observation["player_speed_value"] = np.linalg.norm(observation["player_speed"], keepdims=True)
        obs_strs = {}
        for name, value in sorted(observation.items(), key=lambda x: x[0]):
            if type(value) is np.ndarray:
                if value.size == 1:
                    obs_strs[name] = "{:20}: {:9.2f}           ".format(re.sub("_", " ", string.capwords(name)),
                                                                        value[0])
                elif value.size == 2:
                    obs_strs[name] = "{:20}: ({:8.2f}, {:8.2f})".format(re.sub("_", " ", string.capwords(name)),
                                                                        value[0], value[1])

        return obs_strs


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Position of ball and player are relative to the center point
        if self.randomize_position:
            self.ball = Ball(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)))
            self.player = Player_v0(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)))
        else:
            self.ball = Ball(Vec2d(*self.ball_position), Vec2d.zero())
            self.player = Player_v0(Vec2d(*self.player_position))
        if self.learn_to_kick:
            self.player.position = self.ball.position

        self.time = 0
        self.accumulated_reward = 0

        self.terminated = False
        self.truncated = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        self.time += timeDelta

        self.player.act(action, self.ball)
        self.player.update()
        self.ball.update([self.player])

        reward = 0
        if not self.terminated and not self.truncated:
            if self.ball.home_goal:
                reward += 1
            elif self.ball.away_goal:
                reward += -1

            # extra negative reward if the player stays on the border
            if self.player.fixed_on_border:
                reward += -0.1

        self.accumulated_reward += reward

        if self.ball.goal: 
            #print("Game terminated. Goal.")
            self.terminated = True
        if self.time >= self.time_limit - FIX_PRECISION:
            #print("Game truncated. Reach time limit.")
            self.truncated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, self.terminated, self.truncated, info

    def add_customized_reward(self, reward):
        self.accumulated_reward += reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(self.__class__.__name__)
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(PITCH_GREEN)

        # First we draw the left half pitch
        pygame.draw.rect(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75, 60),
                (1050, 680),
            ),
            width=LINE_WIDTH
        )
        # Now we draw the center circle
        pygame.draw.circle(
            canvas,
            LINE_WHITE,
            (600, 400),
            91.5,
            width=LINE_WIDTH
        )
        # Now we draw the center line
        pygame.draw.line(
            canvas,
            LINE_WHITE,
            (600, 400 - 680 / 2),
            (600, 400 + 680 / 2 - LINE_WIDTH),
            width=LINE_WIDTH,
        )
        # Now we draw the goals
        pygame.draw.rect(
            canvas,
            GOAL_WHITE,
            pygame.Rect(
                (75 - 35 + LINE_WIDTH - GOAL_WIDTH * 2, 60 + 680 / 2 - 37.5 - GOAL_WIDTH),
                (35 + GOAL_WIDTH * 2, 75 + GOAL_WIDTH * 2),
            ),
            width=GOAL_WIDTH
        )
        pygame.draw.rect(
            canvas,
            GOAL_WHITE,
            pygame.Rect(
                (75 + 1050 - LINE_WIDTH, 60 + 680 / 2 - 37.5 - GOAL_WIDTH),
                (35 + GOAL_WIDTH * 2, 75 + GOAL_WIDTH * 2),
            ),
            width=GOAL_WIDTH
        )
        # Now we draw the boxes
        pygame.draw.rect(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75, 60 + 680 / 2 - 165 - 37.5),
                (165, 405),
            ),
            width=LINE_WIDTH
        )
        pygame.draw.rect(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75, 60 + 680 / 2 - 55 - 37.5),
                (55, 185),
            ),
            width=LINE_WIDTH
        )
        pygame.draw.rect(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 + 1050 - 165, 60 + 680 / 2 - 165 - 37.5),
                (165, 405),
            ),
            width=LINE_WIDTH
        )
        pygame.draw.rect(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 + 1050 - 55, 60 + 680 / 2 - 55 - 37.5),
                (55, 185),
            ),
            width=LINE_WIDTH
        )
        # Now draw penalty points
        pygame.draw.circle(
            canvas,
            LINE_WHITE,
            (75 + 110, 60 + 680 / 2),
            LINE_WIDTH,
            width=LINE_WIDTH
        )
        pygame.draw.circle(
            canvas,
            LINE_WHITE,
            (75 + 1050 - 110, 60 + 680 / 2),
            LINE_WIDTH,
            width=LINE_WIDTH
        )
        # Now draw center points
        pygame.draw.circle(
            canvas,
            LINE_WHITE,
            (600, 400),
            LINE_WIDTH,
            width=LINE_WIDTH
        )
        
        # Now we draw the box arcs
        pygame.draw.arc(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 + 110 - 91.5, 60 + 680 / 2 - 91.5),
                (91.5 * 2, 91.5 * 2),
            ),
            start_angle = -0.926,
            stop_angle  = 0.926,
            width=LINE_WIDTH
        )
        pygame.draw.arc(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 + 1050 - 110 - 91.5, 60 + 680 / 2 - 91.5),
                (91.5 * 2, 91.5 * 2),
            ),
            start_angle = -0.926 + np.pi,
            stop_angle  = 0.926 + np.pi,
            width=LINE_WIDTH
        )

        # Now we draw the corner arcs
        pygame.draw.arc(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 - 10, 60 - 10),
                (20, 20),
            ),
            start_angle = -np.pi / 2,
            stop_angle  = 0,
            width=LINE_WIDTH
        )
        pygame.draw.arc(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 - 10, 60 + 680 - 10 - LINE_WIDTH),
                (20, 20),
            ),
            start_angle = 0,
            stop_angle  = np.pi / 2,
            width=LINE_WIDTH
        )
        pygame.draw.arc(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 + 1050 - 10 - LINE_WIDTH, 60 - 10),
                (20, 20),
            ),
            start_angle = -np.pi,
            stop_angle  = -np.pi / 2,
            width=LINE_WIDTH
        )
        pygame.draw.arc(
            canvas,
            LINE_WHITE,
            pygame.Rect(
                (75 + 1050 - 10 - LINE_WIDTH, 60 + 680 - 10 - LINE_WIDTH),
                (20, 20),
            ),
            start_angle = np.pi / 2,
            stop_angle  = np.pi,
            width=LINE_WIDTH
        )

        # draw the time
        pygame.draw.rect(
            canvas,
            TIME_WHITE,
            pygame.Rect(
                (80, 20),
                (110, 25),
            ),
        )
        self.draw_text("{:02d}:{:02d}.{:01d}".format(int(self.time // 60), 
                                                     int(self.time % 60), 
                                                     int(self.time * 10 % 10)), 
                       TIME_TEXT_SIZE, 
                       font_name="consolas",
                       x=135, y=20, color=BLACK,
                       name="time")

        # draw goal celebration
        if self.ball.goal:
            if self.ball.home_goal:
                goal_colors = [self.player.color, WHITE]
            elif self.ball.away_goal:
                goal_colors = [SKYBLUE, BLACK]
            pygame.draw.rect(
                canvas,
                goal_colors[int(self.time * 8) % 2],
                pygame.Rect(
                    (500, 10),
                    (200, 40),
                ),
            )
            self.draw_text("GOAL!", 
                           GOAL_TEXT_SIZE, 
                           font_name="consolas",
                           x=600, y=15, 
                           color=goal_colors[int(self.time * 8) % 2 - 1],
                           name="goal")

        # print reward
        self.draw_text("reward:{:8.4f}".format(self.accumulated_reward), 
                       REWARD_TEXT_SIZE, 
                       font_name="consolas",
                       x=1030, y=25, color=BLACK,
                       name="reward")

        # print state
        obs_strs = self._get_obs_strs()
        state_y = 750
        for key, state_str in obs_strs.items():
            self.draw_text(state_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=300, y=state_y, color=BLACK,
                           name="state {}".format(key))
            state_y += STATE_ACTION_TEXT_SIZE

        # print action
        action_strs = self.player.get_action_strs()
        state_y = 750
        for key, action_str in action_strs.items():
            self.draw_text(action_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=900, y=state_y, color=BLACK,
                           name="action {}".format(key))
            state_y += STATE_ACTION_TEXT_SIZE


        # draw the ball and the player
        self.player.draw(canvas)
        self.ball.draw(canvas)


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.blit_text()
            self.player.blit_text(self.window)
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def blit_text(self):
        for name, surf_dict in self.text_surfs.items():
            surf = surf_dict["surf"]
            rect = surf_dict["rect"]
            self.window.blit(surf, rect)

    def draw_text(self, text, size, font_name=None, x=CENTER.x, y=CENTER.y, color=BLACK, background=TIME_WHITE, 
                  name="text"):
        font_name = pygame.font.get_default_font() if font_name is None else font_name
        font = pygame.font.SysFont(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.text_surfs[name] = {"surf": text_surface, "rect": text_rect}


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class SelfTraining_v1(SelfTraining_v0):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, learn_to_kick=False, time_limit=120, randomize_position=False,
                 ball_position=(0, 0), player_position=(-100, 0)):
        super().__init__(render_mode, learn_to_kick, time_limit, randomize_position, ball_position, player_position)
        if randomize_position:
            self.ball = Ball(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)), can_be_out=True)
            self.player = Player_v1(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)))
        else:
            self.ball = Ball(Vec2d(0, 0), Vec2d(0, 0), can_be_out=True)
            self.player = Player_v1(Vec2d(-100, 0))
        if self.learn_to_kick:
            self.ball.position = self.player.position

    def step(self, action):

        self.time += timeDelta

        self.player.act(action, self.ball)
        self.player.update()
        self.ball.update([self.player])

        reward = 0
        if not self.terminated and not self.truncated:
            if self.ball.goal:
                if self.ball.home_goal:
                    reward += 1
                elif self.ball.away_goal:
                    reward += -1
            elif self.ball.out:
                reward += -0.1

            # extra negative reward if the player stays on the border
            if self.player.fixed_on_border:
                reward += -0.1

        self.accumulated_reward += reward

        if self.ball.goal: 
            #print("Game terminated. Goal.")
            self.terminated = True
        elif self.ball.out:
            #print("Game terminated. Out.")
            self.terminated = True
        if self.time >= self.time_limit - FIX_PRECISION:
            #print("Game truncated. Reach time limit.")
            self.truncated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, self.terminated, self.truncated, info


class SelfTraining_v2(SelfTraining_v1):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, learn_to_kick=False, time_limit=120, randomize_position=False,
                 ball_position=(0, 0), player_position=(-100, 0)):
        super().__init__(render_mode, learn_to_kick, time_limit, randomize_position, ball_position, player_position)
        if randomize_position:
            self.ball = Ball(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)), can_be_out=True)
            self.player = Player_v2(Vec2d(np.random.uniform(-525, 525), np.random.uniform(-340, 340)))
        else:
            self.ball = Ball(Vec2d(0, 0), Vec2d(0, 0), can_be_out=True)
            self.player = Player_v2(Vec2d(-100, 0))
        if self.learn_to_kick:
            self.ball.position = self.player.position

        self.observation_space = spaces.Dict(
            {
                "ball_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "ball_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_direction": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_angular_speed": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )

        # 5 continuous action space
        # first two is player moving acceleration, along x and y
        # second two is player kicking momentum, along x and y
        # fifth is player angular acceleration
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5, ), dtype=np.float32)

    def _get_obs(self):
        return {
                "ball_position": self.ball.observe_position(),
                "ball_speed": self.ball.observe_speed(),
                "player_position": self.player.observe_position(),
                "player_speed": self.player.observe_speed(),
                "player_direction": self.player.observe_direction(),
                "player_angular_speed": self.player.observe_angular_speed(),
               }


