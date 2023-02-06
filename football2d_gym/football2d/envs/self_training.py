import gym
from gym import spaces
import pygame
import numpy as np
from pymunk import Vec2d
from football2d.envs.ball import Ball
from football2d.envs.player import Player
from pprint import pprint
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

timeDelta = 0.02

class SelfTraining_v0(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, time_limit=120, randomize_position=False):
        super().__init__()
        self.time_limit = time_limit
        self.window_width = 1200  # The size of the PyGame window
        self.window_height = 800  # The size of the PyGame window
        self.center_point = CENTER

        # Position of ball and player are relative to the center point
        if randomize_position:
            self.ball = Ball(Vec2d(*np.random.randint(-300, 300, 2)))
            self.player = Player(Vec2d(*np.random.randint(-300, 300, 2)))
        else:
            self.ball = Ball(Vec2d(0, 0))
            self.player = Player(Vec2d(-100, 0))
        self.time = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "ball_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "ball_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "player_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            }
        )

        # (2, 2) continuous action space
        # first vector is player moving acceleration, along x and y
        # second vector is player kicking momentum, along x and y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 2), dtype=np.float32)

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
                "distance_to_goal": self.ball.distance_to_right_goal()
               }


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        self.time += timeDelta

        self.player.act(action, self.ball)
        terminated = False
        truncated = False
        if self.ball.home_goal:
            reward = 1
        elif self.ball.away_goal:
            reward = -1
        else:
            reward = 0

        if self.ball.goal and self.ball.speed.length == 0:
            terminated = True
        if self.time > self.time_limit:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()
        #pprint(observation)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
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
        reward = 0
        if self.ball.goal:
            if self.ball.home_goal:
                goal_colors = [self.player.color, WHITE]
                reward = 1
            elif self.ball.away_goal:
                goal_colors = [SKYBLUE, BLACK]
                reward = -1
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
        self.draw_text("reward:{:2d}".format(reward), 
                       REWARD_TEXT_SIZE, 
                       font_name="consolas",
                       x=1070, y=25, color=BLACK,
                       name="reward")


        # draw the ball and the player
        self.player.update()
        self.ball.update()
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

    def __init__(self, render_mode=None, time_limit=120, randomize_position=False):
        super().__init__(render_mode, time_limit, randomize_position)
