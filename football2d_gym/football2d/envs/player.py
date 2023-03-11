import pygame
from pymunk import Vec2d
import numpy as np
import ipdb

CENTER = Vec2d(600, 400)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PLAYER_RED = (200, 0, 0)
PLAYER_SIZE = 15
PLAYER_MASS = 75

KICK_RANGE = 5
REBOUND_RANGE = 2
MAX_SPEED = 100
MAX_ACCELERATION = 100
MAX_MOMENTUM = 150
RUNUP_FACTOR = 1
RESISTANCE_FACTOR = 10

# v1
DIRECTION_UNCERTAINTY = 0.1
POWER_UNCERTAINTY = 20
SPEED_UNCERTAINTY = 1

# v2
MAX_BACKWARD_SPEED = 25
MAX_BACKWARD_ACCELERATION = 25
MAX_ANGULAR_SPEED = 10
MAX_ANGULAR_ACCELERATION = 50
HEADING_DIRECTION_UNCERTAINTY = 0.1
HEADING_POWER_UNCERTAINTY = 20
HEADING_POWER_DIFFICULTY = 75

deltaTime = 0.02

class Player_v0(object):
    def __init__(self, position: Vec2d, speed: Vec2d=None, mass=PLAYER_MASS,
                       kick_range=KICK_RANGE, rebound_range=REBOUND_RANGE,
                       max_speed=MAX_SPEED, max_acceleration=MAX_ACCELERATION,  
                       max_momentum=MAX_MOMENTUM,
                       resistance_factor=RESISTANCE_FACTOR,
                       name="Robben", number=10, 
                       color=PLAYER_RED, size=PLAYER_SIZE):
        super().__init__()
        # player state
        self.position = position
        self.speed = speed if speed is not None else Vec2d.zero()
        self.mass = mass # kg

        self.kick_range = kick_range
        self.rebound_range = rebound_range
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_momentum = max_momentum
        self.resistance_factor = resistance_factor
        
        self.acceleration = Vec2d.zero()
        self.action = None
        self.kicked_ball = False
        self.fixed_on_border = False

        # useless player attributes
        self.name = name
        self.number = number

        self.color = color
        self.size = size

        self.text_surfs = dict()

    def observe_position(self):
        return np.array([self.position.x, self.position.y], dtype=np.float32)

    def observe_speed(self):
        return np.array([self.speed.x, self.speed.y], dtype=np.float32)

    def draw_text(self, text, size, font_name=None, x=CENTER.x, y=CENTER.y, color=BLACK, name="text"):
        font_name = pygame.font.get_default_font() if font_name is None else font_name
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.text_surfs[name] = {"surf": text_surface, "rect": text_rect}

    def draw(self, canvas):
        # Now draw center points
        player_pos = CENTER + self.position
        pygame.draw.circle(
            canvas,
            self.color,
            (player_pos.x, player_pos.y),
            self.size,
            width=0
        )
        pygame.draw.circle(
            canvas,
            BLACK,
            (player_pos.x, player_pos.y),
            self.size,
            width=2
        )

        self.draw_text(str(self.number), self.size, 
                       x=player_pos.x, y=player_pos.y - 5, color=BLACK,
                       name="number")
        self.draw_text(self.name, self.size, 
                       x=player_pos.x, y=player_pos.y + self.size + 3, color=BLACK,
                       name="name")

    def blit_text(self, window):
        for name, surf_dict in self.text_surfs.items():
            surf = surf_dict["surf"]
            rect = surf_dict["rect"]
            window.blit(surf, rect)

    def fix_position(self):
        # Player's range of motion is larger than the pitch
        margin = 50
        fix_x = False
        fix_y = False
        if -75 / 2 < self.position.y < 75 / 2:
            x = min(max(self.position.x, -1050 / 2 - 35), 1050 / 2 + 35)
            self.position = Vec2d(x, self.position.y)
            fix_x = self.position.x <= -1050 / 2 - 35 or self.position.x >= 1050 / 2 + 35
        elif -1050 / 2 < self.position.x < 1050 / 2:
            y = min(max(self.position.y, -680 / 2 - margin), 680 / 2 + margin)
            self.position = Vec2d(self.position.x, y)
            fix_y = self.position.y <= -680 / 2 - margin or self.position.y >= 680 / 2 + margin
        else:
            if self.in_the_net():
                y = self.position.y
                if y <= -75 / 2:
                    y = -75 / 2
                    fix_y = True
                elif y >= 75 / 2:
                    y = 75 / 2
                    fix_y = True
                self.position = Vec2d(self.position.x, y)
            else:
                x = min(max(self.position.x, -1050 / 2 - margin), 1050 / 2 + margin)
                y = min(max(self.position.y, -680 / 2 - margin), 680 / 2 + margin)
                self.position = Vec2d(x, y)
                fix_x = self.position.x <= -1050 / 2 - margin or self.position.x >= 1050 / 2 + margin
                fix_y = self.position.y <= -680 / 2 - margin or self.position.y >= 680 / 2 + margin

        return fix_x, fix_y

    def get_acceleration(self):
        force = self.resistance_factor * self.mass
        acceleration = self.acceleration - self.speed.normalized() * force / self.mass
        return acceleration

    def get_action_strs(self):
        action = self.action if self.action is not None else np.zeros(5)
        action_strs = {}
        action_strs["acceleration"]  = "{:20}: ({:8.4f}, {:8.4f})".format("Acceleration", 
                                                                          action[0], action[1])
        action_strs["kick_momentum"] = "{:20}: ({:8.4f}, {:8.4f})".format("Kick momentum", 
                                                                          action[2], action[3])
        return action_strs

    def act(self, action, ball):
        self.action = action

        move_acceleration = Vec2d(*action[0:2] * self.max_acceleration)
        if move_acceleration.length > self.max_acceleration:
            move_acceleration = move_acceleration.normalized() * self.max_acceleration
        self.acceleration = move_acceleration

        kick_momentum = Vec2d(*action[2:4] * self.max_momentum)
        if kick_momentum.length > self.max_momentum:
            kick_momentum = kick_momentum.normalized() * self.max_momentum
        kick_momentum += kick_momentum.normalized() * kick_momentum.normalized().dot(self.speed) * RUNUP_FACTOR 

        # Kick the ball if in range
        if (self.position - ball.position).length < self.kick_range:
            ball.kicked(kick_momentum)
            self.kicked_ball = True
        else:
            self.kicked_ball = False

    def in_the_net(self, position=None):
        if position is None:
            position = self.position
        home_goal = False # ball in the right net
        away_goal = False # ball in the left net
        if 1050 / 2 < position.x < 1050 / 2 + 35 and -75 / 2 < position.y < 75 / 2:
            home_goal = True
        if -1050 / 2 - 35 < position.x < -1050 / 2 and -75 / 2 < position.y < 75 / 2:
            away_goal = True

        return home_goal or away_goal

    def update(self):
        acceleration = self.get_acceleration()
        self.speed = self.speed + acceleration * deltaTime
        if self.speed.length > self.max_speed:
            self.speed = self.speed.normalized() * self.max_speed

        old_position = self.position
        old_fix_x, old_fix_y = self.fix_position()

        self.position = self.position + self.speed * deltaTime
        fix_x, fix_y = self.fix_position()
        if fix_x or fix_y:
            self.fixed_on_border = True
            self.speed = Vec2d.zero()
            if old_fix_x or old_fix_y:
                self.position = old_position
        else:
            self.fixed_on_border = False


class Player_v1(Player_v0):
    def __init__(self, position: Vec2d, speed: Vec2d=None, mass=PLAYER_MASS,
                       kick_range=KICK_RANGE, rebound_range=REBOUND_RANGE,
                       max_speed=MAX_SPEED, max_acceleration=MAX_ACCELERATION,  
                       max_momentum=MAX_MOMENTUM,
                       resistance_factor=RESISTANCE_FACTOR,
                       name="Robben", number=10, 
                       color=PLAYER_RED, size=PLAYER_SIZE,
                       # new attributes for v1
                       direction_uncertainty=DIRECTION_UNCERTAINTY,
                       power_uncertainty=POWER_UNCERTAINTY,
                       speed_uncertainty=SPEED_UNCERTAINTY
                       ):
        super().__init__(position, speed, mass, kick_range, rebound_range, max_speed, max_acceleration, max_momentum,
                         resistance_factor, name, number, color, size)
        self.direction_uncertainty = direction_uncertainty
        self.power_uncertainty = power_uncertainty
        self.speed_uncertainty = speed_uncertainty

    def act(self, action, ball):
        self.action = action

        move_acceleration = Vec2d(*action[0:2] * self.max_acceleration)
        if move_acceleration.length > self.max_acceleration:
            move_acceleration = move_acceleration.normalized() * self.max_acceleration
        self.acceleration = move_acceleration

        kick_momentum = Vec2d(*action[2:4] * self.max_momentum)
        if kick_momentum.length > self.max_momentum:
            kick_momentum = kick_momentum.normalized() * self.max_momentum

        # adjust noise factor according to player speed
        speed_factor = self.speed.length / self.max_speed * self.speed_uncertainty
        strength_factor = kick_momentum.length / self.max_momentum
        direction_factor = self.direction_uncertainty * strength_factor * speed_factor
        power_factor = self.power_uncertainty * strength_factor * speed_factor

        # randomize direction
        direction_noise = np.random.normal() * direction_factor
        kick_momentum = kick_momentum.rotated(direction_noise)

        # randomize power
        power_noise = np.random.normal() * power_factor
        kick_momentum = kick_momentum.normalized() * min(max(0, kick_momentum.length + power_noise), self.max_momentum)

        kick_momentum += kick_momentum.normalized() * kick_momentum.normalized().dot(self.speed) * RUNUP_FACTOR 
        # Kick the ball if in range
        if (self.position - ball.position).length < self.kick_range:
            ball.kicked(kick_momentum)


class Player_v2(Player_v1):
    def __init__(self, position: Vec2d, speed: Vec2d=None, mass=PLAYER_MASS,
                       kick_range=KICK_RANGE, rebound_range=REBOUND_RANGE,
                       max_speed=MAX_SPEED, max_acceleration=MAX_ACCELERATION,  
                       max_momentum=MAX_MOMENTUM,
                       resistance_factor=RESISTANCE_FACTOR,
                       name="Robben", number=10, 
                       color=PLAYER_RED, size=PLAYER_SIZE,
                       direction_uncertainty=DIRECTION_UNCERTAINTY,
                       power_uncertainty=POWER_UNCERTAINTY,
                       speed_uncertainty=SPEED_UNCERTAINTY,
                       # v2 attributes
                       direction=Vec2d(1, 0), angular_speed=0,
                       max_backward_speed=MAX_BACKWARD_SPEED, max_backward_acceleration=MAX_BACKWARD_ACCELERATION,
                       max_angular_speed=MAX_ANGULAR_SPEED, max_angular_acceleration=MAX_ANGULAR_ACCELERATION,
                       heading_direction_uncertainty=HEADING_DIRECTION_UNCERTAINTY,
                       heading_power_uncertainty=HEADING_POWER_UNCERTAINTY,
                       heading_power_difficulty=HEADING_POWER_DIFFICULTY
                       ):
        super().__init__(position, speed, mass, kick_range, rebound_range, max_speed, max_acceleration, max_momentum,
                         resistance_factor, name, number, color, size,
                         direction_uncertainty, power_uncertainty, speed_uncertainty)
        self.direction = direction
        self.angular_speed = angular_speed

        self.max_backward_speed = max_backward_speed
        self.max_backward_acceleration = max_backward_acceleration
        self.max_angular_speed = max_angular_speed
        self.max_angular_acceleration = max_angular_acceleration

        self.heading_direction_uncertainty = heading_direction_uncertainty
        self.heading_power_uncertainty = heading_power_uncertainty
        self.heading_power_difficulty = heading_power_difficulty

        self.angular_acceleration = 0

    def draw(self, canvas):
        super().draw(canvas)
        player_pos = CENTER + self.position
        player_direction_pos = player_pos + self.direction.normalized() * (self.size + 5)
        direction_end_pos_1 = player_pos + self.direction.normalized().rotated(0.5) * (self.size + 1)
        direction_end_pos_2 = player_pos + self.direction.normalized().rotated(-0.5) * (self.size + 1)
        pygame.draw.line(
            canvas,
            BLACK,
            (player_direction_pos.x, player_direction_pos.y),
            (direction_end_pos_1.x, direction_end_pos_1.y),
            width=2
        )
        pygame.draw.line(
            canvas,
            BLACK,
            (player_direction_pos.x, player_direction_pos.y),
            (direction_end_pos_2.x, direction_end_pos_2.y),
            width=2
        )

    def observe_direction(self):
        return np.array([self.direction.x, self.direction.y], dtype=np.float32)

    def observe_angular_speed(self):
        return np.array([self.angular_speed, ], dtype=np.float32)

    def get_angular_acceleration(self):
        # assume no resistance force for turning
        return self.angular_acceleration

    def get_action_strs(self):
        action = self.action if self.action is not None else np.zeros(5)
        action_strs = {}
        action_strs["acceleration"]  = "{:20}: ({:8.2f}, {:8.2f})".format("Acceleration", 
                                                                          action[0], action[1])
        action_strs["kick_momentum"] = "{:20}: ({:8.2f}, {:8.2f})".format("Kick momentum", 
                                                                          action[2], action[3])
        action_strs["angular_acc"]   = "{:20}: {:9.2f}           ".format("Angular acceleration", 
                                                                          action[4])
        return action_strs

    def act(self, action, ball):
        self.action = action

        heading_factor_move = (1 - Vec2d.dot(self.direction.normalized(), self.speed.normalized())) / 2 # 0-1 value
        move_acceleration = Vec2d(*action[0:2] * self.max_acceleration)
        max_acceleration = self.max_acceleration \
                           - heading_factor_move * (self.max_acceleration - self.max_backward_acceleration)
        if move_acceleration.length > max_acceleration:
            move_acceleration = move_acceleration.normalized() * max_acceleration
        self.acceleration = move_acceleration

        kick_momentum = Vec2d(*action[2:4] * self.max_momentum)
        if kick_momentum.length > self.max_momentum:
            kick_momentum = kick_momentum.normalized() * self.max_momentum

        turn_acceleration = action[4] * self.max_angular_acceleration
        if turn_acceleration > self.max_angular_acceleration:
            turn_acceleration = self.max_angular_acceleration
        if turn_acceleration < -self.max_angular_acceleration:
            turn_acceleration = -self.max_angular_acceleration
        self.angular_acceleration = turn_acceleration

        # v2: heading control
        heading_factor_kick = (1 - Vec2d.dot(self.direction.normalized(), kick_momentum.normalized())) / 2 # 0-1 value

        # adjust noise factor according to player speed
        speed_factor = self.speed.length / self.max_speed * self.speed_uncertainty
        strength_factor = kick_momentum.length / self.max_momentum
        direction_factor = (self.direction_uncertainty + heading_factor_kick * self.heading_direction_uncertainty) \
                           * strength_factor * speed_factor
        power_factor = (self.power_uncertainty + heading_factor_kick * self.heading_power_uncertainty) \
                       * strength_factor * speed_factor

        # randomize direction
        direction_noise = np.random.normal() * direction_factor
        kick_momentum = kick_momentum.rotated(direction_noise)

        # randomize power
        power_noise = np.random.normal() * power_factor
        adjusted_momentum = kick_momentum.length + power_noise - heading_factor_kick * self.heading_power_difficulty
        kick_momentum = kick_momentum.normalized() * min(max(0, adjusted_momentum), self.max_momentum)

        kick_momentum += kick_momentum.normalized() * kick_momentum.normalized().dot(self.speed) * RUNUP_FACTOR 
        # Kick the ball if in range
        if (self.position - ball.position).length < self.kick_range:
            ball.kicked(kick_momentum)


    def update(self):
        # angular update
        angular_acceleration = self.get_angular_acceleration()
        self.angular_speed = self.angular_speed + angular_acceleration * deltaTime
        if self.angular_speed > self.max_angular_speed:
            self.angular_speed = self.max_angular_speed
        if self.angular_speed < -self.max_angular_speed:
            self.angular_speed = -self.max_angular_speed
        self.direction = self.direction.rotated(self.angular_speed * deltaTime)

        # modify speed limit based on moving direction
        heading_factor_move = (1 - Vec2d.dot(self.direction.normalized(), self.speed.normalized())) / 2 # 0-1 value

        acceleration = self.get_acceleration()
        self.speed = self.speed + acceleration * deltaTime
        max_speed = self.max_speed - heading_factor_move * (self.max_speed - self.max_backward_speed)
        if self.speed.length > max_speed:
            self.speed = self.speed.normalized() * max_speed

        old_position = self.position
        old_fix_x, old_fix_y = self.fix_position()

        self.position = self.position + self.speed * deltaTime
        fix_x, fix_y = self.fix_position()
        if fix_x or fix_y:
            self.fixed_on_border = True
            self.speed = Vec2d.zero()
            if old_fix_x or old_fix_y:
                self.position = old_position
        else:
            self.fixed_on_border = False


