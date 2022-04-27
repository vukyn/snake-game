import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


SPEED = 15
BLOCK_SIZE = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
Point = namedtuple('Point', 'x, y')
Font = pygame.font.Font('font/Press Start 2P.ttf', 15)


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 1. Move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 2. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score

        # 3. Place new food if needed
        if self.head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 4. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 5. Return reward, game over and score
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(
                pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = Font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            # right turn (r -> d -> l -> u)
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx]
        else:
            # left turn (r -> u -> l -> d)
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx]

        x = self.head.x
        y = self.head.y
        self.direction = new_direction

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        if self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        if self.direction == Direction.UP:
            y -= BLOCK_SIZE
        if self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hit boundaries
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hit itself
        if pt in self.snake[1:]:
            return True

        return False
