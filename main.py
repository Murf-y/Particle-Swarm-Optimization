import random
import math
from tqdm import tqdm

import pygame
from pygame.locals import *

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)
PARTICLE_COLOR = (0, 0, 0)
PARTICLE_RADIUS = 5

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Particle Swarm Optimization")


class Particle:
    def __init__(self, num_variables, bounds):
        self.position = [random.uniform(
            bounds[i][0], bounds[i][1]) for i in range(num_variables)]
        self.velocity = [random.uniform(-1, 1) for _ in range(num_variables)]
        self.best_position = self.position[:]
        self.best_fitness = float('inf')

        self.hash = hash(tuple(self.velocity) +
                         tuple(random.random() for _ in range(10)))

    def evaluate(self):
        x, y,  = self.position
        fitness = (x - 2) ** 2 + (y - 2) ** 2
        return fitness


class PSO:
    def __init__(self, num_particles, num_variables, bounds, max_iterations):
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.num_particles = num_particles
        self.num_variables = num_variables
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.global_best_position = [2.0] * num_variables
        self.global_best_fitness = float('inf')
        self.particles = [Particle(num_variables, bounds)
                          for _ in range(num_particles)]

        self.history = {particle.hash: [particle.position[:]]
                        for particle in self.particles}

        self.history[hash('global_best')] = [
            self.global_best_position[:]]

        self.iteration_used = 0

    def optimize(self):
        for _ in tqdm(range(self.max_iterations)):
            for particle in self.particles:
                fitness = particle.evaluate()
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position[:]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position[:]
                    self.history[hash('global_best')].append(
                        particle.position[:])

            for particle in self.particles:
                for i in range(self.num_variables):
                    r1, r2 = random.random(), random.random()
                    particle.velocity[i] = (self.inertia_weight * particle.velocity[i] +
                                            self.cognitive_weight * r1 * (particle.best_position[i] - particle.position[i]) +
                                            self.social_weight * r2 * (self.global_best_position[i] - particle.position[i]))
                    particle.position[i] += particle.velocity[i]
                self.history[particle.hash].append(particle.position[:])

            self.iteration_used += 1


def draw(history, hash_key, iteration, color=PARTICLE_COLOR):
    if iteration >= len(history[hash_key]):
        return
    particle_position_at_iteration = history[hash_key][iteration]
    pygame.draw.circle(screen, color, (int(particle_position_at_iteration[0] * 10 + SCREEN_WIDTH / 2),
                                       int(particle_position_at_iteration[1] * 10 + SCREEN_HEIGHT / 2)), PARTICLE_RADIUS)


def main():
    num_particles = 20
    num_variables = 2
    bounds = [[-100, 100]] * num_variables
    max_iterations = 500

    optimizer = PSO(num_particles, num_variables, bounds, max_iterations)
    optimizer.optimize()

    print("Global best position: ", optimizer.global_best_position)
    print("Global best fitness: ", optimizer.global_best_fitness)

    running = True
    clock = pygame.time.Clock()
    history = optimizer.history
    iteration = 0
    while running:
        # Limit frame speed to 60 fps
        clock.tick(10)

        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # Draw background
        screen.fill(BACKGROUND_COLOR)

        # Draw particles
        for particle in optimizer.particles:
            draw(history, particle.hash, iteration)

        draw(history, hash('global_best'), iteration, color=(255, 0, 0))
        # Update the screen
        pygame.display.flip()

        # Update iteration
        iteration += 1


if __name__ == "__main__":
    main()
