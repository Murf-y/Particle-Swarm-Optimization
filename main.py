import random
import math
from tqdm import tqdm


class Particle:
    def __init__(self, num_variables, bounds):
        self.position = [random.uniform(
            bounds[i][0], bounds[i][1]) for i in range(num_variables)]
        self.velocity = [random.uniform(-1, 1) for _ in range(num_variables)]
        self.best_position = self.position[:]
        self.best_fitness = float('inf')

    def evaluate(self):
        # equation to solve z= x * exp(-x**2 - y**2)
        x = self.position[0]
        y = self.position[1]

        fitness = x * math.exp(-x**2 - y**2)
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
        self.global_best_position = [0.0] * num_variables
        self.global_best_fitness = float('inf')
        self.particles = [Particle(num_variables, bounds)
                          for _ in range(num_particles)]

    def optimize(self):
        for iteration in tqdm(range(self.max_iterations)):
            for particle in self.particles:
                fitness = particle.evaluate()
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position[:]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position[:]

            for particle in self.particles:
                for i in range(self.num_variables):
                    r1, r2 = random.random(), random.random()
                    particle.velocity[i] = (self.inertia_weight * particle.velocity[i] +
                                            self.cognitive_weight * r1 * (particle.best_position[i] - particle.position[i]) +
                                            self.social_weight * r2 * (self.global_best_position[i] - particle.position[i]))
                    particle.position[i] += particle.velocity[i]


def main():

    # Minimize the equation z= x * exp(-x**2 - y**2)

    num_particles = 10
    num_variables = 2
    bounds = [[-100, 100]] * num_variables
    max_iterations = 100000

    optimizer = PSO(num_particles, num_variables, bounds, max_iterations)
    optimizer.optimize()

    print("Global Best Position: ", optimizer.global_best_position)
    print("Global Best Fitness: ", optimizer.global_best_fitness)


if __name__ == "__main__":
    main()
