import math
import os
import pygame
from Shapes import Dock
from Shapes import Player
import numpy as np
import matplotlib.pyplot as plt

pygame.init()
width = 800
height = 600
BG = (144, 201, 120)
player = None

def draw_bg():
    screen.fill(BG)

sign = -1
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('MPC Simulation')
clock = pygame.time.Clock()
dock_instance = Dock()

image_path = os.path.join(os.getcwd(), "bot.png")
image = pygame.image.load(image_path)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))  # Create a main figure with two subplots
# plt.ion()  # Turn on interactive mode for animation

pygame.display.flip()

run = True
closedPath = False
pointList = []
coordinates = []

# Time to run the program in seconds
run_time = 60  # Change this to your desired time limit
start_time = pygame.time.get_ticks() / 1000  # Get the start time in seconds

def draw_points():
    if pointList is not None:
        for point in pointList:
            pygame.draw.circle(screen, (255, 0, 0), point, 2)

def distance(point1, point2):
    return math.fabs(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

def create_route():
    if closedPath:
        for i in range(0, len(pointList) - 1):
            dist = int(distance(pointList[i], pointList[i + 1]) + 1)
            coordinates.append(np.linspace(pointList[i], pointList[i + 1], dist + 1))

        dist = int(distance(pointList[-1], pointList[0]))
        coordinates.append(np.linspace(pointList[-1], pointList[0], dist))

        matrix = np.vstack(coordinates)
        zero_column = np.zeros((matrix.shape[0], 1))
        extended_matrix = np.hstack((matrix, zero_column))
        extended_matrix = np.round(extended_matrix)

        for _ in range(len(extended_matrix) - 1):
            dx = extended_matrix[_ + 1][0] - extended_matrix[_][0]
            dy = extended_matrix[_ + 1][1] - extended_matrix[_][1]
            angle = sign * math.atan2(dy, dx)
            extended_matrix[_][2] = angle
        print("Len = " + str(len(extended_matrix)))

        return extended_matrix


def draw_trajectory():
    if closedPath:
        for i in range(0, len(pointList) - 1):
            pygame.draw.line(screen, (255, 255, 255), pointList[i], pointList[i + 1])
        pygame.draw.line(screen, (255, 255, 255), pointList[-1], pointList[0])

v = []
ii = []
xx = []
yy = []
aa = []
trajectory =[]
while run:
    draw_bg()
    draw_points()
    dock_instance.draw(screen, width)
    draw_trajectory()

    # Check if time limit is reached

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(pygame.mouse.get_pos())
            pointList.append(pygame.mouse.get_pos())
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            closedPath = True
            print("path is done, close the path")
            trajectory = create_route()
            player = Player(0.1, trajectory)
            print(trajectory)

    if player is not None:
        #player = Player(0.1, trajectory)
        #player.prediction_horizon()
        player.move_and_rotate(screen)

        # print("Here")
        # ii.append(player.ii)
        # xx.append(player.x_sim[player.ii - 1])
        # yy.append(trajectory[player.ii - 1])
        # print(xx[player.ii - 1])
        # ax1.clear()
        # ax1.plot(ii, [x[0] for x in xx], marker='o', color='green', label='X_sim')  # Plot on the second subplot for yy
        # ax1.plot(ii, [x[1] for x in xx], marker='o', color='red', label='Y_sim')  # Plot on the second subplot for yy
        #
        # ax1.plot(ii, [y[0] for y in yy], marker='x', color='green', label='X_ref')  # Plot on the second subplot for yy
        # ax1.plot(ii, [y[1] for y in yy], marker='x', color='red', label='Y_ref')  # Plot on the second subplot for yy
        #
        # ax1.legend()
        # ax2.clear()
        # ax2.plot(ii, [y[2] for y in yy], marker='o', color='green', label='Angle_ref')  # Plot on the third subplot for angles
        # ax2.plot(ii, [-x[2] for x in xx], marker='x', color='red', label='Angle_sim')  # Plot on the third subplot for angles
        # ax2.set_ylim(-math.pi, math.pi)
        # ax2.legend()
        # ax3.clear()
        # if len(player.acc) >= 1:
        #     acc_diff = [(player.acc[i] - player.acc[i - 1]) / 0.1 for i in range(1, len(player.acc))]
        #     ax3.plot(ii[1:], acc_diff, color='black')
        #     #ax3.set_ylim(-6, 6)
        # plt.pause(0.1)  # Pause to allow

    # plt.draw()
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
