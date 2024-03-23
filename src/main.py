import numpy as np
import random
import matplotlib.pyplot as plt

# Define the problem parameters
num_customers = 20
max_vehicles = 2
min_demand = 1
max_demand = 20
capacity = 100
depot = (50, 50)  # Depot location

text =[]

# Generate random customer demands
np.random.seed(0)
customer_locations = np.random.rand(num_customers, 2) * 100
customer_demands = np.random.randint(min_demand, max_demand, size=num_customers)



# Genetic Algorithm Parameters
population_size = 50
num_generations = 200
mutation_rate = 0.1
elite_percentage = 0.1

# Calculate distances between locations
distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
for i in range(num_customers + 1):
    for j in range(num_customers + 1):
        distance_matrix[i, j] = np.linalg.norm(
            np.array(depot if i == 0 else customer_locations[i - 1]) -
            np.array(depot if j == 0 else customer_locations[j - 1])
        )

# Genetic Algorithm Functions
def initialize_population():
    population = []
    for _ in range(population_size):
        chromosome = list(range(1, num_customers + 1))
        chromosome.sort(key=lambda x: distance_matrix[0][x])  # Sort customers by distance to the depot
        population.append(chromosome)
    return population

def evaluate_fitness(chromosome):
    routes = [[] for _ in range(max_vehicles)]
    vehicle_capacities = [capacity] * max_vehicles
    for customer in chromosome:
        min_route_idx = np.argmin([distance_matrix[0][customer], distance_matrix[customer][0]])
        for idx, route in enumerate(routes):
            if vehicle_capacities[idx] >= customer_demands[customer - 1]:
                route.append(customer)
                vehicle_capacities[idx] -= customer_demands[customer - 1]
                break

    total_distance = sum(
        sum(distance_matrix[i][j] for i, j in zip(route, route[1:]))
        for route in routes
    )
    return 1 / total_distance if total_distance > 0 else float('-inf'), routes

def selection(population):
    tournament_size = 3
    selected_parents = []
    for _ in range(2):
        tournament = random.sample(population, tournament_size)
        best_parent = max(tournament, key=lambda x: evaluate_fitness(x)[0])
        selected_parents.append(best_parent)
    return selected_parents

def crossover(parent1, parent2):
    crossover_point = random.randint(0, num_customers - 1)
    child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]
    return child1, child2

def swap_mutation(chromosome):
    idx1, idx2 = random.sample(range(num_customers), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def reverse_sequence_mutation(chromosome):
    start_idx = random.randint(0, num_customers - 2)
    end_idx = random.randint(start_idx + 1, num_customers - 1)
    chromosome[start_idx:end_idx + 1] = reversed(chromosome[start_idx:end_idx + 1])
    return chromosome

def mutate(chromosome):
    if random.random() < mutation_rate:
        mutation_type = random.choice(["swap", "reverse", None])
        if mutation_type == "swap":
            return swap_mutation(chromosome)
        elif mutation_type == "reverse":
            return reverse_sequence_mutation(chromosome)
    return chromosome

def elitism(population):
    elite_size = int(population_size * elite_percentage)
    elite = sorted(population, key=lambda x: evaluate_fitness(x)[0], reverse=True)[:elite_size]
    return elite

def genetic_algorithm():
    population = initialize_population()
    best_fitnesses = []
    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = elitism(population) + new_population
        best_fitnesses.append(max([evaluate_fitness(chromosome)[0] for chromosome in population]))
        print(f"Generation {generation + 1}, Best Fitness: {best_fitnesses[-1]}")
    best_chromosome = max(population, key=lambda x: evaluate_fitness(x)[0])
    best_fitness, best_routes = evaluate_fitness(best_chromosome)
    return best_chromosome, best_fitness, best_routes, best_fitnesses

def plot_routes(routes):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    titles = []
    for idx, route in enumerate(routes):
        route_points = [depot] + [customer_locations[customer - 1] for customer in route] + [depot]
        plt.plot([point[0] for point in route_points], [point[1] for point in route_points], 
                 '->', color=colors[idx % len(colors)])
        
        for i, point in enumerate(route_points):
            if i != 0 and i != len(route_points) - 1:
                customer_id = route[i - 1]
                demand = customer_demands[customer_id - 1]
                plt.text(point[0], point[1], f'{customer_id}({demand})', color='black', fontsize=10)
        
        # Display sum of demands of each route at the bottom left corner
        route_demand_text = f'Sum of Demands for {color_names[idx]}: {sum(customer_demands[customer - 1] for customer in route)}'
        titles.append(route_demand_text)
        
    plt.scatter(depot[0], depot[1], color='k', marker='o', label='Depot')
    plt.scatter([loc[0] for loc in customer_locations], [loc[1] for loc in customer_locations], 
                color='b', marker='s', label='Customers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(', '.join(titles))  # Set the title using the list of titles
    plt.legend()
    plt.draw()





def main():

    best_chromosome, best_fitness, best_routes, best_fitnesses = genetic_algorithm()
    print("Best Route:", best_chromosome)
    print("Best Fitness:", best_fitness)
    print(sum(customer_demands))
    plot_routes(best_routes)

    plt.figure()
    plt.plot(best_fitnesses)
    plt.title('Best Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

if __name__=="__main__":
    main()