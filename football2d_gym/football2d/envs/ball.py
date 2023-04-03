import pygame
import numpy as np
from pymunk import Vec2d
import ipdb

CENTER = Vec2d(600, 400)
BLACK = (0, 0, 0)
BALL_YELLOW = (220, 220, 0)
BALL_SIZE = 8
BALL_MASS = 0.45

RESISTANCE_FACTOR_1 = 15
RESISTANCE_FACTOR_2 = 8e-4

BOUNCE_FACTOR = 0.8
GOAL_BOUNCE_FACTOR = 0.1

deltaTime = 0.02

class Ball(object):
    def __init__(self, position: Vec2d, speed: Vec2d=None, mass=BALL_MASS, 
                 resistance_factor_1=RESISTANCE_FACTOR_1, resistance_factor_2=RESISTANCE_FACTOR_2,
                 can_be_out=False,
                 bounce_factor=BOUNCE_FACTOR, goal_bounce_factor=GOAL_BOUNCE_FACTOR,
                 color=BALL_YELLOW, size=BALL_SIZE):
        super().__init__()
        self.position = position
        self.speed = speed if speed is not None else Vec2d.zero()
        self.mass = mass # kg
        
        self.can_be_out = can_be_out
        self.resistance_factor_1 = resistance_factor_1
        self.resistance_factor_2 = resistance_factor_2
        self.bounce_factor = bounce_factor
        self.goal_bounce_factor = goal_bounce_factor

        # useless ball attributes
        self.color = color
        self.size = size

        self.acceleration = self.get_acceleration()
        self.curr_bounce_factor = bounce_factor # change due to different situations
        self.home_goal, self.away_goal = self.in_the_net()
        self.out = False
        self.rebound_from_player = False


    @property
    def goal(self):
        return self.home_goal or self.away_goal

    def observe_position(self):
        return np.array([self.position.x, self.position.y], dtype=np.float32)

    def observe_speed(self):
        return np.array([self.speed.x, self.speed.y], dtype=np.float32)

    def distance_to(self, other):
        return self.position.get_distance(other)

    def distance_to_right_goal(self):
        if -75 / 2 <= self.position.y <= 75 / 2:
            distance = np.abs(1050 / 2 - self.position.x)
        else:
            distance = min(self.position.get_distance(Vec2d(1050 / 2, 75 / 2)), 
                           self.position.get_distance(Vec2d(1050 / 2, -75 / 2)))
        in_the_net, _ = self.in_the_net()
        if in_the_net:
            return 0
        else:
            return distance

    def distance_to_left_goal(self):
        if -75 / 2 <= self.position.y <= 75 / 2:
            distance = np.abs(-1050 / 2 - self.position.x)
        else:
            distance = min(self.position.get_distance(Vec2d(-1050 / 2, 75 / 2)), 
                           self.position.get_distance(Vec2d(-1050 / 2, -75 / 2)))
        _, in_the_net = self.in_the_net()
        if in_the_net:
            return 0
        else:
            return distance

    def draw(self, canvas):
        # Now draw center points
        ball_pos = CENTER + self.position
        pygame.draw.circle(
            canvas,
            self.color,
            (ball_pos.x, ball_pos.y),
            self.size,
            width=0
        )
        pygame.draw.circle(
            canvas,
            BLACK,
            (ball_pos.x, ball_pos.y),
            self.size,
            width=2
        )

    def get_acceleration(self):
        force_1 = self.resistance_factor_1 * self.mass
        force_2 = self.resistance_factor_2 * (self.speed.length ** 2)
        self.acceleration = -self.speed.normalized() * (force_1 + force_2) / self.mass
        return self.acceleration

    def fix_position(self):
        fix_x = False
        fix_y = False
        if -75 / 2 < self.position.y < 75 / 2:
            x = min(max(self.position.x, -1050 / 2 - 35), 1050 / 2 + 35)
            self.position = Vec2d(x, self.position.y)
            fix_x = self.position.x <= -1050 / 2 - 35 or self.position.x >= 1050 / 2 + 35
        elif -1050 / 2 - self.size / 2 < self.position.x < 1050 / 2 + self.size / 2:
            y = min(max(self.position.y, -680 / 2 - self.size / 2), 680 / 2 + self.size / 2)
            self.position = Vec2d(self.position.x, y)
            fix_y = self.position.y <= -680 / 2 - self.size / 2 or self.position.y >= 680 / 2 + self.size / 2
        else:
            if self.goal:
                y = self.position.y
                if y <= -75 / 2:
                    y = -75 / 2
                    fix_y = True
                elif y >= 75 / 2:
                    y = 75 / 2
                    fix_y = True
                self.position = Vec2d(self.position.x, y)
            else:
                x = min(max(self.position.x, -1050 / 2 - self.size / 2), 1050 / 2 + self.size / 2)
                y = min(max(self.position.y, -680 / 2 - self.size / 2), 680 / 2 + self.size / 2)
                self.position = Vec2d(x, y)
                fix_x = self.position.x <= -1050 / 2 - self.size / 2 or self.position.x >= 1050 / 2 + self.size / 2
                fix_y = self.position.y <= -680 / 2 - self.size / 2 or self.position.y >= 680 / 2 + self.size / 2

        return fix_x, fix_y

    def fix_speed(self):
        if self.speed.length < 0.5:
            self.speed = Vec2d.zero()

    def out_of_pitch(self):
        out_of_x = False
        out_of_y = False
        if not -1050 / 2 - self.size / 2 < self.position.x < 1050 / 2 + self.size / 2:
            out_of_x = True
        if not -680 / 2 - self.size / 2 < self.position.y < 680 / 2 + self.size / 2:
            out_of_y = True
        self.out = out_of_x or out_of_y

        return out_of_x or out_of_y, (out_of_x, out_of_y)

    def in_the_net(self, position=None):
        if position is None:
            position = self.position
        home_goal = False # ball in the right net
        away_goal = False # ball in the left net
        if 1050 / 2 + self.size / 2 < position.x < 1050 / 2 + 35 and -75 / 2 < position.y < 75 / 2:
            home_goal = True
        if -1050 / 2 - 35 < position.x < -1050 / 2 - self.size / 2 and -75 / 2 < position.y < 75 / 2:
            away_goal = True

        return home_goal, away_goal

    def update(self, players):
        self.get_acceleration()
        self.speed = self.speed + self.acceleration * deltaTime
        self.position = self.position + self.speed * deltaTime
        self.fix_speed()
        fix_x, fix_y = self.fix_position()

        home_goal, away_goal = self.in_the_net()
        self.home_goal = self.home_goal or home_goal
        self.away_goal = self.away_goal or away_goal

        # rebound from border
        if home_goal or away_goal:
            self.curr_bounce_factor = self.goal_bounce_factor
        out_of_pitch, (out_of_x, out_of_y) = self.out_of_pitch()
        if out_of_pitch and not (home_goal or away_goal):
            if self.can_be_out and not self.goal:
                self.speed = Vec2d.zero()
            else:
                if fix_x:
                    self.speed = Vec2d(-self.speed.x, self.speed.y)
                if fix_y:
                    self.speed = Vec2d(self.speed.x, -self.speed.y)
            self.speed = self.speed * self.curr_bounce_factor
            self.curr_bounce_factor = self.bounce_factor

        # rebound from players
        for player in players:
            if self.in_rebound_range(player):
                self.rebound_from(player)
                break
        else:
            self.rebound_from_player = False

    def rebound_from(self, player):
        d_to_player = (player.position - self.position)
        if d_to_player.length > player.rebound_range:
            return
        
        # ball can not go throuth the player
        if not self.rebound_from_player and not player.kicked_ball:
            inner_angle = d_to_player.get_angle_between(self.speed)
            position_fix_len = np.sqrt(np.square(player.rebound_range) - np.square(d_to_player.length * np.sin(inner_angle))) - d_to_player.length * np.cos(inner_angle)
            self.position = self.position - self.speed.normalized() * position_fix_len
            rebound_angle = self.speed.get_angle_between(player.position - self.position)
            if rebound_angle < 0:
                self.speed = self.speed.rotated(np.pi + rebound_angle * 2)
            else:
                self.speed = self.speed.rotated(-np.pi + rebound_angle * 2)
            self.speed = (self.speed + player.speed) * self.bounce_factor + player.speed
            self.rebound_from_player = True


    def in_rebound_range(self, player):
        distance = (player.position - self.position).length
        if distance <= player.rebound_range:
            return True
        else:
            return False


    def kicked(self, momentum):
        self.speed = self.speed + momentum / self.mass
        
