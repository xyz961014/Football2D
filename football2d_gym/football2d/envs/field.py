import numpy as np
import re
import string
from pymunk import Vec2d
import pygame
from pprint import pprint
from collections import OrderedDict
import ipdb

from football2d.envs.colors import (
    PITCH_GREEN,
    LINE_WHITE,
    GOAL_WHITE
    )


CENTER = Vec2d(600, 400)
LINE_WIDTH = 3
GOAL_WIDTH = 3


class Field(object):
    def __init__(self):
        super().__init__()

    def draw(self, canvas):
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

