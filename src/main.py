import numpy as np
import random
import matplotlib.pyplot as plt
import time

''' Defining variables and necessary randomly generated parameters '''
# Define the problem parameters
num_customers = 15
max_vehicles = 20
min_demand = 1
max_demand = 20
# Depot location
depot = (50, 50) 

" Customer locations and demands - randomly generated "
customer_locations = np.random.rand(num_customers, 2) * 100
customer_demands = np.random.randint(min_demand, max_demand, size=num_customers)

" Customer locations and demands - from file "
# Localizations
#np.savetxt('customer_locations.txt', customer_locations)
# customer_locations = np.loadtxt('customer_locations.txt')
# Demands
#np.savetxt('customer_demands.txt', customer_demands)
# customer_demands = np.loadtxt('customer_demands.txt')

" Vehicles capacities - randomly generated "
# Generate random capacities for vehicles
#vehicle_capacities = np.random.randint(50, 150, size=max_vehicles)

" Vehicle capacities - user defined "
# User input - vehicle_capacities [<vehicle1>, <cehicle2>, ...]
vehicle_capacities = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]

# Genetic Algorithm Parameters
population_size = 50
num_generations = 1000
mutation_rate = 0.1
elite_percentage = 0.1

# Distance array creation
distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
# Calculate distances between locations
for i in range(num_customers + 1):
    for j in range(num_customers + 1):
        distance_matrix[i, j] = np.linalg.norm(
            np.array(depot if i == 0 else customer_locations[i - 1]) -
            np.array(depot if j == 0 else customer_locations[j - 1])
        )

''' Population initialization function '''
# Genetic Algorithm Functions
def initialize_population():
    population = []
    for _ in range(population_size):
        chromosome = list(range(1, num_customers + 1))
        # Sorting customers by distance to the depot
        chromosome.sort(key=lambda x: distance_matrix[0][x])
        population.append(chromosome)
    return population

''' Fitness function '''
def evaluate_fitness(chromosome):
    # Initialize empty routes for each vehicle
    routes = [[] for _ in range(max_vehicles)]
    # Initialize remaining capacities for each vehicle
    vehicle_capacities_remaining = list(vehicle_capacities)
    # Iterate through customers in the chromosome
    for customer in chromosome:
        # Iterate through routes and allocate customer to a route if capacity allows
        for idx, route in enumerate(routes):
            if vehicle_capacities_remaining[idx] >= customer_demands[customer - 1]:
                route.append(customer)
                # Update remaining capacity of the vehicle
                vehicle_capacities_remaining[idx] -= customer_demands[customer - 1]
                break
    # Calculate total distance traveled by all vehicles
    total_distance = sum(sum(distance_matrix[i][j] for i, j in zip(route, route[1:]))for route in routes)
    # Return fitness value (reciprocal of total distance) and routes
    return (1 / total_distance) if total_distance > 0 else float('-inf'), routes

''' Selection function - tournament'''
def selection(population):
    # Tournament selection parameters
    tournament_size = 3
    # Initialize list to store selected parents
    selected_parents = []
    # Perform tournament selection twice to select two parents
    for _ in range(2):
        # Randomly select individuals (chromosomes) for the tournament
        tournament = random.sample(population, tournament_size)
        # Choose the best individual (parent) based on fitness
        best_parent = max(tournament, key=lambda x: evaluate_fitness(x)[0])
        # Add the best parent to the list of selected parents
        selected_parents.append(best_parent)
    return selected_parents

''' Elitism function - selection of two best solutions depending on evaluationg_fitness '''
def elitism(population):
    # Selects elite individuals from the population based on their fitness scores
    # this line is to reduce sorting algorithm compiling time
    elite_size = int(population_size * elite_percentage)
    # Sorts the population based on fitness scores and selects top individuals
    elite = sorted(population, key=lambda x: evaluate_fitness(x)[0], reverse=True)[:elite_size]
    return elite

''' Crossover functions '''
def crossover(parent1, parent2):
    crossover_type = random.choice(["OX", "PD", "PMX", "CX"])
    if crossover_type == "OX":  # Order Crossover (OX) 
        return order_crossover(parent1, parent2)
    if crossover_type == "PD":  # Partially Divided Crossover (PD) 
        return pd_crossover(parent1,parent2)
    elif crossover_type == "PMX":  # Partially Mapped Crossover (PMX) 
        return pmx_crossover(parent1, parent2)
    elif crossover_type == "CX":  # Cycle Crossover (CX) 
        return cycle_crossover(parent1, parent2)
    
def order_crossover(parent1, parent2):
    # Order Crossover (OX)
    crossover_point1 = np.random.randint(0, len(parent1))
    crossover_point2 = np.random.randint(crossover_point1 + 1, len(parent1) + 1)
    # Initialize offspring chromosomes
    child1 = [-1] * len(parent1)
    child2 = [-1] * len(parent1)
    # Copy the segment between crossover points from parents to children
    child1[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
    child2[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]
    # Fill in remaining elements from the second parent to the first child
    idx_child1 = crossover_point2
    idx_parent2 = crossover_point2
    while -1 in child1:
        if parent2[idx_parent2 % len(parent2)] not in child1:
            child1[idx_child1 % len(parent1)] = parent2[idx_parent2 % len(parent2)]
            idx_child1 += 1
        idx_parent2 += 1
    # Fill in remaining elements from the first parent to the second child
    idx_child2 = crossover_point2
    idx_parent1 = crossover_point2
    while -1 in child2:
        if parent1[idx_parent1 % len(parent1)] not in child2:
            child2[idx_child2 % len(parent1)] = parent1[idx_parent1 % len(parent1)]
            idx_child2 += 1
        idx_parent1 += 1
    return child1, child2

def pd_crossover(parent1, parent2):
    # Randomly select a crossover point
    crossover_point = random.randint(0, num_customers - 1)
    # Perform crossover to create child1 by combining segments from parent1 and parent2
    child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    # Perform crossover to create child2 by combining segments from parent2 and parent1
    child2 = parent2[crossover_point:] + [gene for gene in parent1 if gene not in parent2[crossover_point:]]
    return child1, child2

def pmx_crossover(parent1, parent2):
    # Partially Mapped Crossover (PMX)
    size = len(parent1)
    point1 = random.randint(0, size)
    point2 = random.randint(0, size)
    if point1 > point2:
        point1, point2 = point2, point1
    # Initialize the children as copies of the parents
    child1 = parent1[:]
    child2 = parent2[:]
    # Copy the segment from parent1 to child2 and from parent2 to child1
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]
    # Update mapping for the segment
    mapping1 = {}
    mapping2 = {}
    for i in range(point1, point2):
        mapping1[parent1[i]] = parent2[i]
        mapping2[parent2[i]] = parent1[i]
    # Apply mapping for the rest of the genes
    for i in range(size):
        if point1 <= i < point2:
            continue
        while child1[i] in mapping1:
            child1[i] = mapping1[child1[i]]
        while child2[i] in mapping2:
            child2[i] = mapping2[child2[i]]
    return child1, child2

def cycle_crossover(parent1, parent2):
    # Cycle Crossover (CX)
    size = len(parent1)
    child1 = [-1] * size
    child2 = [-1] * size
    # Choose a random starting point
    start = np.random.randint(0, size)
    idx = start
    while True:
        # Perform cycle crossover
        child1[idx] = parent1[idx]
        child2[idx] = parent2[idx]
        if parent1[idx] == parent2[start]:
            break
        idx = parent2.index(parent1[idx])
        if idx == start:
            break
    # Swap the elements not yet assigned
    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2[i]
        if child2[i] == -1:
            child2[i] = parent1[i]
    return child1, child2

''' Mutations and mutation function '''
def mutate(chromosome):
    # Check if mutation should occur based on the mutation rate
    if random.random() < mutation_rate:
        # Select a mutation type randomly from swap, reverse, shuffle
        mutation_type = random.choice(["swap", "reverse", "shuffle", "insertion", "displacement", "cim", "creep"])
        # Apply the selected mutation type
        if mutation_type == "swap":
            return swap_mutation(chromosome)
        elif mutation_type == "reverse":
            return reverse_sequence_mutation(chromosome)
        elif mutation_type == "shuffle":
            return shuffle_mutation(chromosome)
        elif mutation_type == "insertion":
            return insertion_mutation(chromosome)
        elif mutation_type == "displacement":
            return displacement_mutation(chromosome)
        elif mutation_type == "cim":
            return cim_mutation(chromosome)
        elif mutation_type == "creep":
            return creep_mutation(chromosome)      
    return chromosome

def swap_mutation(chromosome):
    # Randomly select two distinct indices for mutation
    idx1, idx2 = random.sample(range(num_customers), 2)
    # Swap the genes at the selected indices
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def reverse_sequence_mutation(chromosome):
    # Randomly select start and end indices for the subsequence to be reversed
    start_idx = random.randint(0, num_customers - 2)
    end_idx = random.randint(start_idx + 1, num_customers - 1)
    # Reverse the selected subsequence
    chromosome[start_idx:end_idx + 1] = reversed(chromosome[start_idx:end_idx + 1])
    return chromosome

def shuffle_mutation(chromosome):
    # Randomly select start and end indices for the subsequence to be shuffled
    start_idx = random.randint(0, num_customers - 2)
    end_idx = random.randint(start_idx + 1, num_customers - 1)
    # Extract the selected subsequence
    subset = chromosome[start_idx:end_idx + 1]
    # Shuffle the subsequence
    random.shuffle(subset)
    # Replace the original subsequence with the shuffled one
    chromosome[start_idx:end_idx + 1] = subset
    return chromosome

def insertion_mutation(chromosome):
    # Select a customer to be moved
    customer = random.choice(chromosome)
    chromosome.remove(customer)  # Remove the customer from its current position
    # Choose a random position to insert the customer
    insertion_index = random.randint(0, len(chromosome))
    chromosome.insert(insertion_index, customer)  # Insert the customer at the new position
    return chromosome

def displacement_mutation(chromosome):
    # Select a random subset of customers
    subset_size = random.randint(2, len(chromosome) - 1)  # Ensure at least two customers are selected
    subset_start = random.randint(0, len(chromosome) - subset_size)
    subset_end = subset_start + subset_size
    subset = chromosome[subset_start:subset_end]

    # Remove the subset from the chromosome
    del chromosome[subset_start:subset_end]

    # Choose a random position to insert the subset
    insertion_index = random.randint(0, len(chromosome))
    chromosome[insertion_index:insertion_index] = subset  # Insert the subset at the new position
    return chromosome

def cim_mutation(chromosome):
    # Select a central point in the chromosome
    central_point = len(chromosome) // 2
    # Reverse the sequence of customers around the central point
    chromosome[central_point:] = reversed(chromosome[central_point:])
    return chromosome

def creep_mutation(chromosome):
    # Select a customer
    customer = random.choice(chromosome)
    # Find the index of the customer
    idx = chromosome.index(customer)
    # Determine the direction of movement (left or right)
    direction = random.choice([-1, 1])
    # Perform creep mutation
    if 0 <= idx + direction < len(chromosome):
        chromosome[idx], chromosome[idx + direction] = chromosome[idx + direction], chromosome[idx]
    return chromosome

''' Main genetic algorithm function '''
def genetic_algorithm():
    start_time = time.time()
    best_fitnesses = []
    routes_lengths = []
    # Executes the genetic algorithm to solve the optimization problem
    population = initialize_population()
    route_init = (min(evaluate_fitness(chormosome) for chormosome in population))[1]
    routes_lengths.append(sum(sum(distance_matrix[i][j] for i, j in zip(route, route[1:])) for route in route_init))
    best_fitnesses.append(max([evaluate_fitness(chromosome)[0] for chromosome in population]))
    for generation in range(num_generations):
        new_population = []
        # Creates new individuals through selection, crossover, and mutation
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        # Combines elite individuals with new population to form the next generation
        population = elitism(population) + new_population
        # Records the best fitness value for the current generation
        best_fitnesses.append(max([evaluate_fitness(chromosome)[0] for chromosome in population]))
        # Records routes lengths
        routes_lengths.append(round(1/best_fitnesses[-1], 2))
        # Print actual values of generation
        print(f"Generation {generation + 1}, Best Fitness: {best_fitnesses[generation]}, Route Length: {routes_lengths[generation]}")
    # Identifies the best chromosome in the final population
    best_chromosome = max(population, key=lambda x: evaluate_fitness(x)[0])
    # Calculates fitness and routes for the best chromosome
    best_fitness, best_routes = evaluate_fitness(best_chromosome)
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time}s')
    return best_fitness, best_routes, best_fitnesses, route_init, routes_lengths

''' Ploting function '''
def plot_routes(routes, title):
    plt.figure(num=title)
    # Plots the routes obtained from the genetic algorithm
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'pink', 'olive', 'orange', 'purple', 'brown', 'yellow', 'teal', 'maroon', 'navy', 'turquoise', 'lavender', 'indigo', 'silver', 'gold', 'plum', 'skyblue', 'coral', 'lime', 'violet']
    titles = []
    for idx, route in enumerate(routes):
        # Constructs route points for plotting
        route_points = [depot] + [customer_locations[customer - 1] for customer in route] + [depot]
        # Plots the route.
        plt.plot([point[0] for point in route_points], [point[1] for point in route_points], 
                  color=colors[idx % len(colors)], linewidth = 0.75)
        # Adds customer IDs and demands as annotations
        for i, point in enumerate(route_points):
            if i != 0 and i != len(route_points) - 1:
                customer_id = route[i - 1]
                demand = customer_demands[customer_id - 1]
                plt.text(point[0], point[1], f'{customer_id}({demand})', color='black', fontsize=10)
        # Generates text for summarizing demands of each route if vehicle is used
        if sum(customer_demands[customer - 1] for customer in route) > 0:
            route_demand_text = f'Sum of Demands for {colors[idx]}: {sum(customer_demands[customer - 1] for customer in route)}({vehicle_capacities[idx]})'
            titles.append(route_demand_text)   
    # Adds depot and customer locations to the plot
    plt.scatter(depot[0], depot[1], color='k', marker='s', label='Depot')
    plt.scatter([loc[0] for loc in customer_locations], [loc[1] for loc in customer_locations], color='b', marker='o', label='Customers')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Sets the title based on the list of titles generated
    plt.title(', '.join(titles))
    plt.legend()
    plt.draw()

''' Main function '''
def main():
    # Main function to execute the genetic algorithm and plot the results
    best_fitness, best_routes, best_fitnesses, route_init, routes_lengths = genetic_algorithm()
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'pink', 'olive', 'orange', 'purple', 'brown', 'yellow', 'teal', 'maroon', 'navy', 'turquoise', 'lavender', 'indigo', 'silver', 'gold', 'plum', 'skyblue', 'coral', 'lime', 'violet']
    # Prints the best route and its fitness
    for id, route in enumerate(best_routes):
        if len(route) != 0:
            print(f"Route for {colors[id]}: {route}")
    print("Best Fitness:", best_fitness)

    # Calculate total sum of all demands
    print(f"Sum of all demands: {sum(customer_demands)}")
    
    # Calculate total initial length traveled by all vehicles
    init_length = sum(sum(distance_matrix[i][j] for i, j in zip(route, route[1:])) for route in route_init)
    print("Initial length traveled:", round(init_length, 2))
    
    # Plots the initial routes
    plot_routes(route_init, title=f'Initial Routes, Total Length: {round(init_length, 2)}')
    
    # Calculate total length traveled by all vehicles
    total_distance_traveled = sum(sum(distance_matrix[i][j] for i, j in zip(route, route[1:])) for route in best_routes)
    print("Total length traveled:", round(total_distance_traveled, 2))

    # Plots the best routes
    plot_routes(best_routes, title=f'Best Routes, Total Length: {round(total_distance_traveled, 2)}')
    
    # Plots the evolution of best fitness over generations
    plt.figure('Road length per Generation')
    plt.plot(routes_lengths)
    plt.title('Road length per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Road length')
    plt.draw()

    # Plots the evolution of best fitness over generations
    plt.figure('Best Fitness per Generation')
    plt.plot(best_fitnesses)
    plt.title('Best Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
    
if __name__ == "__main__":
    main()