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

surface = pygame.Surface((50, 20), pygame.SRCALPHA)
surface.fill("black")

player = Player(surface, 288, 400)
pygame.display.flip()

dt = 0.05
vl = 0
vr = 0
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
    clock.tick(5)

pygame.quit()

