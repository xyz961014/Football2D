import gym
from gym import spaces
import pygame
import numpy as np
import re
import string
from pymunk import Vec2d
from football2d.envs.ball import Ball
from football2d.envs.player import Player_v2, Team
from football2d.envs.field import Field
from pprint import pprint
from collections import OrderedDict
import ipdb

from football2d.envs.colors import (
    TIME_WHITE,
    WHITE,
    BLACK,
    RED,
    SKYBLUE
    )

TEAM_TEXT_SIZE = 16
SCORE_TEXT_SIZE = 20
TIME_TEXT_SIZE = 20
GOAL_TEXT_SIZE = 40
REWARD_TEXT_SIZE = 18
STATE_ACTION_TEXT_SIZE = 15

timeDelta = 0.02
FIXED_PRECISION = 1e-3
CELEBRATION_TIME = 5


class OneOnOneMatch(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, time_limit=200, duration=120,
                 ball_position=(0, 0), 
                 home_player_position=(-100, 0), away_player_position=(100, 0),
                 home_player_direction=(1, 0), away_player_direction=(-1, 0),
                 home_agent=None, away_agent=None,
                 training_side="home",
                 home_player_number=34, away_player_number=16, home_player_name="Xhaka", away_player_name="Rodri",
                 home_team_name="Arsenal", away_team_name="Man city",
                 celebration_time=CELEBRATION_TIME):
        self.time_limit = time_limit
        self.duration = duration
        self.window_width = 1200  # The size of the PyGame window
        self.window_height = 900  # The size of the PyGame window
        self.celebration_time = celebration_time

        self.ball_position = ball_position
        self.home_player_position = home_player_position
        self.away_player_position = away_player_position
        self.home_player_direction = home_player_direction
        self.away_player_direction = away_player_direction

        self.home_agent = home_agent
        self.away_agent = away_agent
        self.training_side = training_side

        self.field = Field()

        # Position of ball and player are relative to the center point
        self.ball = Ball(Vec2d(*self.ball_position), Vec2d.zero(), can_be_out=True)

        # Home team attacks rightwards; Away team attacks leftwards
        home_player = Player_v2(Vec2d(*self.home_player_position), direction=Vec2d(*self.home_player_direction),
                                number=home_player_number, name=home_player_name, side="home")
        self.home_team = Team([home_player], home_team_name, RED, attacking_right=True)
        away_player = Player_v2(Vec2d(*self.away_player_position), direction=Vec2d(*self.away_player_direction),
                                number=away_player_number, name=away_player_name, side="away")
        self.away_team = Team([away_player], away_team_name, SKYBLUE, attacking_right=False)

        self.time = 0
        self.game_time = 0
        self.home_accumulated_reward = 0
        self.away_accumulated_reward = 0

        self.home_goal_celebration = False
        self.away_goal_celebration = False
        self.celebration_count_down = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # Each team observes the field as they are attacking rightwards
        self.observation_space = spaces.Dict(
            OrderedDict({
                "ball_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "ball_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "own_player_positions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
                "own_player_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
                "own_player_directions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
                "own_player_angular_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 1,), dtype=np.float32),
                "opponent_player_positions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
                "opponent_player_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
                "opponent_player_directions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
                "opponent_player_angular_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 1,), dtype=np.float32),
            })
        )
        #self.observation_space = spaces.Dict(
        #    spaces.Dict({
        #        "home_observation": spaces.Dict({
        #            "ball_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        #            "ball_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        #            "own_player_positions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "own_player_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "own_player_directions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "own_player_angular_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 1,), dtype=np.float32),
        #            "opponent_player_positions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "opponent_player_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "opponent_player_directions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "opponent_player_angular_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 1,), dtype=np.float32),
        #        }),
        #        "away_observation": spaces.Dict({
        #            "ball_position": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        #            "ball_speed": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        #            "own_player_positions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "own_player_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "own_player_directions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "own_player_angular_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 1,), dtype=np.float32),
        #            "opponent_player_positions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "opponent_player_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "opponent_player_directions": spaces.Box(-np.inf, np.inf, shape=(1, 2,), dtype=np.float32),
        #            "opponent_player_angular_speeds": spaces.Box(-np.inf, np.inf, shape=(1, 1,), dtype=np.float32),
        #        })
        #    })
        #)

        # first two is player moving acceleration, along x and y
        # second two is player kicking momentum, along x and y
        # fifth is player angular acceleration
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5, ), dtype=np.float32)

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
        # Both teams observe as they are attacking rightwards
        if self.training_side == "home":
            return self.get_home_obs()
        elif self.training_side == "away":
            return self.get_away_obs()
        else:
            raise ValueError("Unsupported training side")
        #return {
        #        "home_observation": home_obs,
        #        "away_observation": away_obs
        #       }

    def get_home_obs(self):
        return self.home_team.observe(self.ball, self.away_team)

    def get_away_obs(self):
        return self.away_team.observe(self.ball, self.home_team)

    def _get_info(self):
        info = OrderedDict({
                "training_side": self.training_side,
                "ball_position": self.ball.observe_position(),
                "ball_speed": self.ball.observe_speed(),
                #"in_play": self.ball.in_play,
                #"is_restart": self.ball.is_restart,
                #"is_throw-in": self.ball.is_throw_in,
                "possession_range": np.array([self.ball.possession_range]),
                "status": self.ball.status
               })
        if self.ball.last_touch_player is not None:
            info["last_touch_player"] = "{} #{}".format(self.ball.last_touch_player.side, 
                                                        self.ball.last_touch_player.number)
        else:
            info["last_touch_player"] = "None"
        info["home_team"] = self.home_team.get_info()
        info["away_team"] = self.away_team.get_info()
        return info

    def _get_info_strs(self):
        info = self._get_info()
        info["total_time"] = np.array([self.time])
        if "ball_speed" in info.keys():
            info["ball_speed_value"] = np.linalg.norm(info["ball_speed"], keepdims=True)
        for team_key in ["home_team", "away_team"]:
            if team_key in info.keys():
                for player_key, player_data in info[team_key].items():
                    if "speed" in player_data.keys():
                        info[team_key][player_key]["speed_value"] = np.linalg.norm(player_data["speed"], keepdims=True)

        info_strs = {}
        for name, value in sorted(info.items(), key=lambda x: x[0]):
            if type(value) is np.ndarray:
                if value.size == 1:
                    info_strs[name] = "{:17}: {:9.2f}           ".format(re.sub("_", " ", string.capwords(name)),
                                                                         value[0])
                elif value.size == 2:
                    info_strs[name] = "{:17}: ({:8.2f}, {:8.2f})".format(re.sub("_", " ", string.capwords(name)),
                                                                         value[0], value[1])
            if type(value) in [bool, str]:
                info_strs[name] = "{:17}: {:>19} ".format(re.sub("_", " ", string.capwords(name)),
                                                                   str(value))
        home_info_strs = {}
        for player_name, player_data in sorted(info["home_team"].items(), key=lambda x: x[0]):
            for name, value in sorted(player_data.items(), key=lambda x: x[0]):
                if type(value) is np.ndarray:
                    name = "{} {}".format(player_name, name)
                    if value.size == 1:
                        home_info_strs[name] = "{:24}: {:9.2f}           ".format(
                            re.sub("_", " ", string.capwords(name)), value[0])
                    elif value.size == 2:
                        home_info_strs[name] = "{:24}: ({:8.2f}, {:8.2f})".format(
                            re.sub("_", " ", string.capwords(name)), value[0], value[1])

        away_info_strs = {}
        for player_name, player_data in sorted(info["away_team"].items(), key=lambda x: x[0]):
            for name, value in sorted(player_data.items(), key=lambda x: x[0]):
                if type(value) is np.ndarray:
                    name = "{} {}".format(player_name, name)
                    if value.size == 1:
                        away_info_strs[name] = "{:24}: {:9.2f}           ".format(
                            re.sub("_", " ", string.capwords(name)), value[0])
                    elif value.size == 2:
                        away_info_strs[name] = "{:24}: ({:8.2f}, {:8.2f})".format(
                            re.sub("_", " ", string.capwords(name)), value[0], value[1])

        return info_strs, home_info_strs, away_info_strs

    def _get_action_strs(self):
        home_action_strs = {}
        for player in self.home_team.players:
            action_strs = player.get_action_strs()
            for key, value in action_strs.items():
                home_action_strs["#{}_{}".format(player.number, key)] = "#{} {}".format(player.number, value)

        away_action_strs = {}
        for player in self.away_team.players:
            action_strs = player.get_action_strs()
            for key, value in action_strs.items():
                away_action_strs["#{}_{}".format(player.number, key)] = "#{} {}".format(player.number, value)

        return home_action_strs, away_action_strs

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Home team start the play
        self.start()

        self.time = 0
        self.game_time = 0
        self.home_accumulated_reward = 0
        self.away_accumulated_reward = 0

        self.home_goal_celebration = False
        self.away_goal_celebration = False
        self.celebration_count_down = 0

        self.terminated = False
        self.truncated = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def start(self, action=None, side="home"):
        self.ball = Ball(Vec2d(*self.ball_position), Vec2d.zero(), can_be_out=True, possession=side)
        self.home_team.reset()
        self.away_team.reset()

        if side == "home":
            self.home_team.set_start_positions([(0, 0)])
            self.home_team.start()
        else:
            self.away_team.set_start_positions([(0, 0)])
            self.away_team.start()

    def step(self, action):
        if "home_action" in action.keys() and action["home_action"] is not None:
            home_action = action["home_action"]
        else:
            if self.home_agent is not None:
                home_action = self.home_agent(self.get_home_obs())
            else:
                home_action = np.zeros((1, 5))
        if "away_action" in action.keys() and action["away_action"] is not None:
            away_action = action["away_action"]
        else:
            if self.away_agent is not None:
                away_action = self.away_agent(self.get_away_obs())
            else:
                away_action = np.zeros((1, 5))

        if self.render_mode == "human":
            # Get keyboard input
            keys = pygame.key.get_pressed()

            ##### Home player control #####
            if keys[pygame.K_d]:
                # Move right
                home_action[0, 0] = 1.
            if keys[pygame.K_a]:
                # Move left
                home_action[0, 0] = -1.
            if keys[pygame.K_w]:
                # Move up
                home_action[0, 1] = -1.
            if keys[pygame.K_s]:
                # Move down
                home_action[0, 1] = 1.
            if keys[pygame.K_q]:
                # Rotate counterclockwise
                home_action[0, 4] = -1.
            if keys[pygame.K_e]:
                # Rotate clockwise
                home_action[0, 4] = 1.

            if keys[pygame.K_t]:
                # Kick forward
                home_action[0, 2:4] = 1. * self.home_team.players[0].direction
            if keys[pygame.K_g]:
                # Kick backward
                home_action[0, 2:4] = -1. * self.home_team.players[0].direction
            if keys[pygame.K_f]:
                # Kick leftward
                home_action[0, 2:4] = 1. * self.home_team.players[0].direction.rotated(0.5 * np.pi)
            if keys[pygame.K_h]:
                # Kick rightward
                home_action[0, 2:4] = -1. * self.home_team.players[0].direction.rotated(-0.5 * np.pi)

            ##### Away player control #####
            if keys[pygame.K_KP6]:
                # Move right
                away_action[0, 0] = -1.
            if keys[pygame.K_KP4]:
                # Move left
                away_action[0, 0] = 1.
            if keys[pygame.K_KP8]:
                # Move up
                away_action[0, 1] = 1.
            if keys[pygame.K_KP5]:
                # Move down
                away_action[0, 1] = -1.
            if keys[pygame.K_KP7]:
                # Rotate counterclockwise
                away_action[0, 4] = -1.
            if keys[pygame.K_KP9]:
                # Rotate clockwise
                away_action[0, 4] = 1.

            if keys[pygame.K_UP]:
                # Kick forward
                away_action[0, 2:4] = -1. * self.away_team.players[0].direction
            if keys[pygame.K_DOWN]:
                # Kick backward
                away_action[0, 2:4] = 1. * self.away_team.players[0].direction
            if keys[pygame.K_LEFT]:
                # Kick leftward
                away_action[0, 2:4] = -1. * self.away_team.players[0].direction.rotated(0.5 * np.pi)
            if keys[pygame.K_RIGHT]:
                # Kick rightward
                away_action[0, 2:4] = 1. * self.away_team.players[0].direction.rotated(-0.5 * np.pi)


        self.time += timeDelta
        if self.ball.in_play:
            self.game_time += timeDelta

        self.home_team.act(home_action, self.ball)
        self.away_team.act(away_action, self.ball)

        self.home_team.update(self.ball)
        self.away_team.update(self.ball)
        self.ball.update(self.home_team.players + self.away_team.players)

        home_reward = 0
        away_reward = 0
        if not self.terminated and not self.truncated:
            if self.ball.home_goal and not self.home_goal_celebration:
                home_reward += 1
                away_reward -= 1
                self.home_team.goal += 1
                self.home_goal_celebration = True
                self.celebration_count_down = self.celebration_time
            elif self.ball.away_goal and not self.away_goal_celebration:
                away_reward += 1
                home_reward -= 1
                self.away_team.goal += 1
                self.away_goal_celebration = True
                self.celebration_count_down = self.celebration_time

            if self.celebration_count_down > 0:
                self.celebration_count_down -= timeDelta
                if not self.celebration_count_down > 0:
                    concede_side = "away" if self.home_goal_celebration else "home"
                    self.start(side=concede_side)
            else:
                self.home_goal_celebration = False
                self.away_goal_celebration = False

            ## extra negative reward if the player stays on the border
            #if self.player.fixed_on_border:
            #    reward += -0.1

        self.home_accumulated_reward += home_reward
        self.away_accumulated_reward += away_reward

        if self.game_time >= self.duration - FIXED_PRECISION: 
            self.terminated = True
        if self.time >= self.time_limit - FIXED_PRECISION:
            self.truncated = True

        observation = self._get_obs()
        info = self._get_info()
        info["home_reward"] = home_reward
        info["away_reward"] = away_reward
        if self.training_side == "home":
            reward = home_reward
        elif self.training_side == "away":
            reward = away_reward

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "human":
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
        self.field.draw(canvas)

        # draw the time
        pygame.draw.rect(
            canvas,
            TIME_WHITE,
            pygame.Rect(
                (80, 20),
                (110, 25),
            ),
        )
        self.draw_text("{:02d}:{:02d}.{:01d}".format(int(self.game_time // 60), 
                                                     int(self.game_time % 60), 
                                                     int(self.game_time * 10 % 10)), 
                       TIME_TEXT_SIZE, 
                       font_name="lucidaconsole",
                       x=135, y=22, color=BLACK,
                       name="game_time")
        # draw score
        pygame.draw.rect(
            canvas,
            WHITE,
            pygame.Rect(
                (190, 20),
                (150, 25),
            ),
        )
        pygame.draw.rect(
            canvas,
            TIME_WHITE,
            pygame.Rect(
                (340, 20),
                (80, 25),
            ),
        )
        pygame.draw.rect(
            canvas,
            WHITE,
            pygame.Rect(
                (420, 20),
                (150, 25),
            ),
        )
        self.draw_text(self.home_team.name[:16], 
                       TEAM_TEXT_SIZE, 
                       font_name="consolas",
                       x=265, y=25, color=BLACK,
                       name="home_team_name")
        self.draw_text(self.away_team.name[:16], 
                       TEAM_TEXT_SIZE, 
                       font_name="consolas",
                       x=495, y=25, color=BLACK,
                       name="away_team_name")
        self.draw_text("{}:{}".format(self.home_team.goal, self.away_team.goal), 
                       SCORE_TEXT_SIZE, 
                       font_name="lucidaconsole",
                       x=380, y=22, color=BLACK,
                       name="score")

        # draw goal celebration
        if self.celebration_count_down > 0:
            if self.home_goal_celebration:
                goal_colors = [self.home_team.color, WHITE]
            elif self.away_goal_celebration:
                goal_colors = [self.away_team.color, BLACK]
            pygame.draw.rect(
                canvas,
                goal_colors[int(self.time * 8) % 2],
                pygame.Rect(
                    (600, 10),
                    (200, 40),
                ),
            )
            self.draw_text("GOAL!", 
                           GOAL_TEXT_SIZE, 
                           font_name="consolas",
                           x=700, y=15, 
                           color=goal_colors[int(self.time * 8) % 2 - 1],
                           name="goal")
        else:
            if "goal" in self.text_surfs.keys():
                self.text_surfs.pop("goal")

        # print reward
        self.draw_text("home reward:{:8.4f}".format(self.home_accumulated_reward), 
                       REWARD_TEXT_SIZE, 
                       font_name="consolas",
                       x=1020, y=10, color=BLACK,
                       name="home reward")
        self.draw_text("away reward:{:8.4f}".format(self.away_accumulated_reward), 
                       REWARD_TEXT_SIZE, 
                       font_name="consolas",
                       x=1020, y=30, color=BLACK,
                       name="away reward")

        # print info and action
        info_strs, home_info_strs, away_info_strs = self._get_info_strs()
        home_action_strs, away_action_strs = self._get_action_strs()
        info_x = 200
        info_y = 760
        for key, info_str in info_strs.items():
            self.draw_text(info_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=info_x, y=info_y, color=BLACK,
                           name="info {}".format(key))
            info_y += STATE_ACTION_TEXT_SIZE
        # home team
        info_x = 560
        info_y = 760
        self.draw_text("Home players", 
                       STATE_ACTION_TEXT_SIZE + 3, 
                       font_name="consolas",
                       x=info_x, y=info_y - 15, color=BLACK,
                       name="home_players")
        for key, info_str in home_info_strs.items():
            self.draw_text(info_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=info_x, y=info_y, color=BLACK,
                           name="home_info {}".format(key))
            info_y += STATE_ACTION_TEXT_SIZE
        info_y += 10
        for key, action_str in home_action_strs.items():
            self.draw_text(action_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=info_x, y=info_y, color=BLACK,
                           name="home_action {}".format(key))
            info_y += STATE_ACTION_TEXT_SIZE
        # away team
        info_x = 1000
        info_y = 760
        self.draw_text("Away players", 
                       STATE_ACTION_TEXT_SIZE + 3, 
                       font_name="consolas",
                       x=info_x, y=info_y - 15, color=BLACK,
                       name="away_players")
        for key, info_str in away_info_strs.items():
            self.draw_text(info_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=info_x, y=info_y, color=BLACK,
                           name="away_info {}".format(key))
            info_y += STATE_ACTION_TEXT_SIZE
        info_y += 10
        for key, action_str in away_action_strs.items():
            self.draw_text(action_str, 
                           STATE_ACTION_TEXT_SIZE, 
                           font_name="consolas",
                           x=info_x, y=info_y, color=BLACK,
                           name="away_action {}".format(key))
            info_y += STATE_ACTION_TEXT_SIZE

        # print action 


        # draw the ball and the players
        self.home_team.draw(canvas)
        self.away_team.draw(canvas)
        self.ball.draw(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.blit_text()
            self.home_team.blit_text(self.window)
            self.away_team.blit_text(self.window)
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

    def draw_text(self, text, size, font_name=None, x=0, y=0, color=BLACK, background=TIME_WHITE, 
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


