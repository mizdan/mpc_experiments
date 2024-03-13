import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the screen
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Surface Example')

# Create a surface
surface = pygame.Surface((width // 2, height // 2))  # Creating a surface half the size of the screen

# Fill the surface with a color
surface.fill((255, 0, 0))  # Filling the surface with red color

# Draw on the surface
pygame.draw.rect(surface, (0, 255, 0), (50, 50, 100, 100))  # Drawing a green rectangle on the surface

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))  # Filling the screen with black color

    # Blit the surface onto the screen
    screen.blit(surface, (width // 4, height // 4))  # Blitting the surface at the center of the screen

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
