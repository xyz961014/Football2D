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

deltaTime = 0.02

class Player(object):
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
        fix_x = False
        fix_y = False
        if -75 / 2 < self.position.y < 75 / 2:
            x = min(max(self.position.x, -1050 / 2 - 35), 1050 / 2 + 35)
            self.position = Vec2d(x, self.position.y)
            fix_x = self.position.x <= -1050 / 2 - 35 or self.position.x >= 1050 / 2 + 35
        elif -1050 / 2 < self.position.x < 1050 / 2:
            y = min(max(self.position.y, -680 / 2), 680 / 2)
            self.position = Vec2d(self.position.x, y)
            fix_y = self.position.y <= -680 / 2 or self.position.y >= 680 / 2
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
                x = min(max(self.position.x, -1050 / 2), 1050 / 2)
                y = min(max(self.position.y, -680 / 2), 680 / 2)
                self.position = Vec2d(x, y)
                fix_x = self.position.x <= -1050 / 2 or self.position.x >= 1050 / 2
                fix_y = self.position.y <= -680 / 2 or self.position.y >= 680 / 2

        return fix_x, fix_y

    def get_acceleration(self):
        force = self.resistance_factor * self.mass
        acceleration = self.acceleration - self.speed.normalized() * force / self.mass
        return acceleration

    def act(self, action, ball):
        move_acceleration = Vec2d(*action[0] * self.max_acceleration)
        if move_acceleration.length > self.max_acceleration:
            move_acceleration = move_acceleration.normalized() * self.max_acceleration
        self.acceleration = move_acceleration

        kick_momentum = Vec2d(*action[1] * self.max_momentum)
        if kick_momentum.length > self.max_momentum:
            kick_momentum = kick_momentum.normalized() * self.max_momentum
        kick_momentum += kick_momentum.normalized() * kick_momentum.normalized().dot(self.speed) * RUNUP_FACTOR 
        # Kick the ball if in range
        if (self.position - ball.position).length < self.kick_range:
            ball.kicked(kick_momentum)

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
            self.speed = Vec2d.zero()
            if old_fix_x or old_fix_y:
                self.position = old_position


