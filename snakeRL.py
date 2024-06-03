import pygame
import random
import numpy as np
from matplotlib import pyplot as plt

SCREEN_WIDTH = 800  # height and width of the display window
SCREEN_HEIGHT = 800
GRID_SIZE = 20  # size of game arena
red = (255, 0, 0)  # food color
green = (0, 255, 0)  # body color
blue = (0, 0, 255)  # head color


class Snake:
    def __init__(self):
        """Initialise the snake, food and grid attributes 0 -> empty cell 1 -> snake body  2 -> snake head 3 -> food"""
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype='int32')
        self.food = []
        '''y->0 1 2 3 4
		x = 0[0 0 0 0 0]
		x = 1[0 0 0 0 0]
		x = 2[0 1 1 0 0]
		x = 3[0 0 0 0 0]
		x = 4[0 0 0 0 0]
		   0
		   |
		1-----3
		   |
		   2
		'''
        # direction dictionary
        self.dir_dict = {2: [-1, 0],  # up
                         1: [0, -1],  # left
                         3: [1, 0],  # down
                         0: [0, 1]}  # right
        food_pos = self.find_empty_cells(k=1)[0]
        self.food = food_pos
        directions_available = []
        position_safe = False
        head_position = []
        while not position_safe:
            directions_available.clear()
            tmp = self.find_empty_cells(k=1)[0]
            if 0 <= tmp[0] - 2 and 0 <= tmp[1] - 2 and tmp[0] + 2 < GRID_SIZE and tmp[1] + 2 < GRID_SIZE:
                if self.grid[tmp[0] - 1][tmp[1]] == 0 and self.grid[tmp[0] - 2][tmp[1]] == 0 and \
                        self.grid[tmp[0] + 1][tmp[1]] == 0 and self.grid[tmp[0] + 2][tmp[1]] == 0:
                    directions_available.append(2)
                    directions_available.append(3)
                    head_position = tmp
                    position_safe = True
                if self.grid[tmp[0]][tmp[1] - 1] == 0 and self.grid[tmp[0]][tmp[1] - 2] == 0 and \
                        self.grid[tmp[0]][tmp[1] + 1] == 0 and self.grid[tmp[0]][tmp[1] + 2] == 0:
                    directions_available.append(0)
                    directions_available.append(1)
                    head_position = [tmp]
                    position_safe = True
        self.direction = random.choice(directions_available)

        self.pos = head_position

        if self.direction == 3:
            self.pos = self.pos + [(self.pos[0][0] - 1, self.pos[0][1]), (self.pos[0][0] - 2, self.pos[0][1])]
        elif self.direction == 2:
            self.pos = self.pos + [(self.pos[0][0] + 1, self.pos[0][1]), (self.pos[0][0] + 2, self.pos[0][1])]
        elif self.direction == 1:
            self.pos = self.pos + [(self.pos[0][0], self.pos[0][1] + 1), (self.pos[0][0], self.pos[0][1] + 2)]
        elif self.direction == 0:
            self.pos = self.pos + [(self.pos[0][0], self.pos[0][1] - 1), (self.pos[0][0], self.pos[0][1] - 2)]

        self.previous_pos = self.pos.copy()

        self.length = len(self.pos)
        self.is_alive = True  # to check if the snake is alive
        self.length_limit = GRID_SIZE * GRID_SIZE - 1
        self.ate_food = False  # check whether the snake ate the food
        self.cell_width = SCREEN_WIDTH // GRID_SIZE
        self.cell_height = SCREEN_HEIGHT // GRID_SIZE

        self.grid[self.food] = 2
        for p in self.pos:
            self.grid[p] = -1
        self.grid[self.pos[0]] = 1

    def hash(self, n):
        x, y = n
        return x * self.grid.shape[1] + y

    def inv_hash(self, n):
        cols = self.grid.shape[1]
        return (n // cols, n % cols)

    def find_empty_cells(self, k=1):
        """to return k random empty cells from the grid"""
        flat_map = self.grid.flatten()
        flat_map = (flat_map == 0)
        empty_cells = np.arange(GRID_SIZE * GRID_SIZE)[flat_map]
        empty_cells = np.random.choice(list(empty_cells), k, replace=False)
        empty_cells = [self.inv_hash(h) for h in empty_cells]
        return empty_cells

    def check_food(self):
        """check if food is eaten by the snake If eaten, create new food"""
        if self.pos[0] == self.food:
            self.ate_food = True
            self.food = self.find_empty_cells()[0]
            self.grid[self.food] = 2

    def check_collision(self, new_head):
        """check if snake has collided with itself or the wall"""
        self.is_alive = True
        x, y = new_head
        if new_head in self.pos[1:]:
            self.is_alive = False
        elif x >= GRID_SIZE or x < 0:
            self.is_alive = False
        elif y >= GRID_SIZE or y < 0:
            self.is_alive = False

    def update(self):
        """update the body of the snake and check for collision"""
        self.previous_pos = self.pos.copy()
        if self.is_alive:
            head = self.pos[0]
            dirn = self.dir_dict[int(self.direction)]
            new_head = (head[0] + dirn[0],
                        head[1] + dirn[1])  # find position of new head

            self.check_collision(new_head)

            if self.is_alive:
                # if alive, update the snake body
                self.grid[head] = -1
                self.pos = [new_head] + self.pos
                self.grid[new_head] = 1
                if not self.ate_food:
                    tail = self.pos.pop()
                    self.grid[tail] = 0
                else:
                    self.ate_food = False
                self.length = len(self.pos)

    def draw(self, screen):
        """draw the snake and food on the screen"""
        screen.fill((0, 0, 0))
        x, y = self.food
        pygame.draw.rect(screen, red, (y * self.cell_width,
                                       x * self.cell_height,
                                       self.cell_width,
                                       self.cell_height))
        # body
        for x, y in self.pos[1:]:
            pygame.draw.rect(screen, green, (y * self.cell_width,
                                             x * self.cell_height,
                                             self.cell_width,
                                             self.cell_height))
        # head
        x, y = self.pos[0]
        pygame.draw.rect(screen, blue, (y * self.cell_width,
                                        x * self.cell_height,
                                        self.cell_width,
                                        self.cell_height))


class Pygame2D:
    def __init__(self, grid_size=20, mode='bot', max_steps_without_eating=1000):
        """Initialise pygame and display attributes"""
        global GRID_SIZE
        GRID_SIZE = grid_size
        allowed_modes = ['bot', 'human']
        assert mode in allowed_modes, "Wrong mode for gym env. Should be from ['bot', 'human']"

        self.mode = mode
        self.done = False
        self.no_op_action = 1  # action which does nothing
        self.human_action = self.no_op_action
        self.steps_without_eating = 0
        self.max_steps_without_eating = max_steps_without_eating
        self.snake = Snake()
        self.game_speed = 60
        if self.mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake 2D")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 30)
            self.game_speed = 12

        # set rewards
        # self.dead_penalty = -1000
        # self.food_reward = GRID_SIZE * len(self.snake.pos)
        # self.move_penalty = -1

        self.timeout_penalty = -50
        self.dead_penalty = -20
        self.food_reward = 10
        self.move_penalty = 0
        self.distance_reward = 0.1

    def get_human_action(self):
        assert self.mode == 'human', "return_action() not usable without 'human' mode for gym env."
        action = self.human_action
        # self.human_action = self.no_op_action
        return action

    def action(self, action):
        """update state by taking action check for collisions and food 0 -> right	1 -> left 	2 -> up 3 -> down"""
        if (action == 0 and self.snake.direction != 1) or \
                (action == 1 and self.snake.direction != 0) or \
                (action == 2 and self.snake.direction != 3) or \
                (action == 3 and self.snake.direction != 2):
            self.snake.direction = action
        # if action == 1:	# do nothing
        self.snake.update()
        self.snake.check_food()
        if self.snake.ate_food:
            self.steps_without_eating = 0
        else:
            self.steps_without_eating += 1

    def evaluate(self):
        """compute reward of the snake"""
        reward = self.move_penalty
        distance_diff = (abs(self.snake.food[0] - self.snake.previous_pos[0][0]) + abs(self.snake.food[1] -
                                                                                       self.snake.previous_pos[0][
                                                                                           1])) - (
                                abs(self.snake.food[0] - self.snake.pos[0][0]) +
                                abs(self.snake.food[1] - self.snake.pos[0][1]))
        if not self.snake.is_alive:
            reward += self.dead_penalty
        elif self.steps_without_eating >= self.max_steps_without_eating:
            reward += self.timeout_penalty
        if self.snake.ate_food:
            reward += self.food_reward
        else:
            reward += self.distance_reward * distance_diff
        return reward

    def is_done(self):
        """check for terminal condition or crash"""
        if not self.snake.is_alive or self.snake.length >= self.snake.length_limit or self.done or \
                self.steps_without_eating >= self.max_steps_without_eating:
            #if self.steps_without_eating >= self.max_steps_without_eating:
            #    print('***********************************************************************************************'
            #          '*****************************************************************')
            self.done = False
            return True
        return False

    def observe(self):
        obs = np.zeros((GRID_SIZE * 2 - 1, GRID_SIZE * 2 - 1, 1), dtype=np.uint8)
        # Fill obs with appropriate values for snake, food, and empty spaces
        snake_x = self.snake.pos[0][0]
        snake_y = self.snake.pos[0][1]
        for i in range(GRID_SIZE):
            pos_x = i - snake_x + GRID_SIZE - 1
            for j in range(GRID_SIZE):
                pos_y = j - snake_y + GRID_SIZE - 1
                if self.snake.grid[i][j] == 2:
                    obs[pos_x, pos_y, 0] = 255  # Food
                elif self.snake.grid[i][j] == 0 or self.snake.grid[i][j] == 1:
                    obs[pos_x, pos_y, 0] = 127  # Empty Space
        return obs

    def view(self):
        """render the state of the game on the screen"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.done = True

        self.snake.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(self.game_speed)

    def run_game_loop(self):
        """run 1 iteration of game loop for human"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.done = True
                if self.mode == 'human':
                    # note down key presses
                    if event.key == pygame.K_RETURN:
                        self.done = True
                    elif event.key == pygame.K_LEFT:
                        self.human_action = 1
                    elif event.key == pygame.K_UP:
                        self.human_action = 2
                    elif event.key == pygame.K_RIGHT:
                        self.human_action = 0
                    elif event.key == pygame.K_DOWN:
                        self.human_action = 3
                    elif event.key == pygame.K_p:
                        print(self.snake.grid)

        action = self.get_human_action()
        self.action(action)
        reward = self.evaluate()
        done = self.is_done()
        #        print(f'reward {reward}', f'action {action}', f'Previous Pos {self.snake.previous_pos}',
        #              f'Pos {self.snake.pos}')

        self.snake.draw(self.screen)
        # swap pygame buffers and render it
        pygame.display.flip()
        self.clock.tick(self.game_speed)

        return reward, done

    def close(self):
        """quit the pygame window"""
        pygame.display.quit()
        pygame.quit()
