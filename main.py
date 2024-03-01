import os

import pygame
from Shapes import Dock
from Shapes import Player
import numpy as np
import matplotlib.pyplot as plt

pygame.init()
width = 800
height = 600
#draw_bg
BG = (144, 201, 120)

def draw_bg():
    screen.fill(BG)


screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('MPC Simulation')
clock = pygame.time.Clock()
dock_instance = Dock()

image_path = os.path.join(os.getcwd(), "bot.png")
image = pygame.image.load(image_path)

player = Player(image, 300, 400)
pygame.display.flip()

run = True


while run:
    draw_bg()
    dock_instance.draw(screen, width)

    player.move_and_rotate(1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    player.draw(screen)
    pygame.display.flip()
    clock.tick(15)

pygame.quit()


