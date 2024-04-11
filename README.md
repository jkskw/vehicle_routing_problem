# DCVRP Solved With Genetic Algorithm
## Table of contents
- [Intorduction](#introduction)
    - [Vehicle routing problem](#vehicle-routing-problem)
    - [Genetic algorithm](#genetic-algorithm)
- [Program](#program)
    - [Genetic algorithm](#genetic-algorithm-1)
    - [Additional features](#additional-features)
- [Example solution](#example-soution)
- [License](#license)

## Introduction
### Vehicle routing problem
The Vehicle Routing Problem (VRP) is a combinatorial optimization problem where the  goal is to determine the optimal set of routes for a fleet of vehicles to serve a given set of customers or locations, subject to various constraints such as vehicle capacity, time windows, and distance or time limitations. Determining the optimal solution to VRP is NP-hard, so the size of problems that can be optimally solved using mathematical programming or combinatorial optimization may be limited. The objective is typically to minimize the total distance traveled, the total time taken, or some other relevant cost metric, while satisfying all the constraints. Unlike traditional VRP, where customers are typically passive and have fixed demand at known locations, in DCVRP, customers are active and request service dynamically, leading to additional challenges in route planning and optimization. Efficient algorithms, including heuristic methods and metaheuristic approaches like genetic algorithms and simulated annealing, are often used to solve DCVRP and find near-optimal solutions. In DCVRP, the objective is to optimize the allocation of vehicles and their capacities to customer requests in order to minimize various costs, such as total travel time, total distance traveled, or the number of vehicles used, while satisfying all the constraints.

### Genetic algorithm
A genetic algorithm (GA) is an optimization technique inspired by the process of natural selection. It operates by creating a population of potential solutions to a problem, which are then evaluated and evolved over successive generations. Through processes such as selection, crossover, and mutation, the algorithm aims to improve solutions iteratively, mimicking the principles of Darwinian evolution. Genetic algorithms are widely used to solve complex optimization problems in various domains, offering efficient approaches for finding near-optimal solutions in large search spaces. Main features of gengetic algorithm in programming:

**Population Representation**: GAs typically represent potential solutions to the problem being solved as individuals in a population. These individuals are encoded in a way that reflects the problem domain.

**Fitness Function**: A fitness function is used to evaluate how good each individual solution is with respect to the problem's objectives. It assigns a numerical value (fitness score) to each individual, indicating its quality.

**Selection Mechanism**: GAs employ a selection mechanism to choose individuals from the population to serve as parents for the next generation. Common selection methods include roulette wheel selection, tournament selection, and rank-based selection.

**Crossover (Recombination)**: Crossover is the process of combining genetic material from selected parents to create new individuals (offspring) for the next generation. It involves exchanging or recombining parts of the genetic information between parents.

**Mutation**: Mutation introduces random changes into individual solutions to maintain genetic diversity in the population. It helps explore new regions of the search space and prevent premature convergence to suboptimal solutions.

**Replacement Strategy**: After generating offspring through crossover and mutation, a replacement strategy is used to determine which individuals from the current population and offspring population will proceed to the next generation. Strategies may include elitism (keeping the best individuals), generational replacement, or steady-state replacement.

**Termination Criteria**: GAs require termination criteria to determine when to stop the evolution process. Common termination conditions include reaching a maximum number of generations, finding a satisfactory solution, or stagnation (no significant improvement over several generations).

**Parameter Tuning**: GAs often involve several parameters that can influence their performance, such as population size, crossover rate, mutation rate, and selection pressure. Tuning these parameters can significantly impact the algorithm's effectiveness.

## Program
### Genetic algorithm
**Population Initialization**
The *[initilization_population](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L70)* is creating chromosomes randomly, additionaly there is *[optimize_vehicle](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L43)* function that takes user/randomly generated capacities of cars and divide the demands to optimize the weight being put on a car. Without the function the program is going to fully load first car, then will send the next one.

**Fitness function**
The *[evaluate_fitness](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L85)* function is optimizing for the minimum road traveled

**Selection Mechanism**
The *[selection](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L105)* function is a tournament based mechanism that takes *tournament_size* number and create offspring, mainly for chromosomes to be as diverse as possible. More info in [replacement strategy](#replacement-strategy).

**Crossover (Recombination)**
In the code, there's a *[crossover](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L130)* function that randomly selects one of five implemented crossovers: *[SPX](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L143)*, *[OX](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L152)*, *[AEX](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L180)*, *[PMX](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L204)*, *[CX](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L224)* with equal probability. [Click for more info.](https://www.researchgate.net/publication/268043232_Comparison_of_eight_evolutionary_crossover_operators_for_the_vehicle_routing_problem)

**Mutation**
The *[mutate](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L249)* function randomly selects from a variety of mutation types including: *[swap](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L271)*, *[reverse sequence](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L278)*, *[shuffle](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L286)*, *[insertion](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L298)*, *[displacement](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L307)*, *[central inversion mutation](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L322)*, *[creep](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L329)*. [Click for more info.](https://www.linkedin.com/pulse/mutations-genetic-algorithms-ali-karazmoodeh-u94pf/)

<span id="replacement-strategy">**Replacement Strategy**</span>
The replacement strategy is constructed such that a selection of tournament offspring is first created and placed in the *new_population* variable. Then *[eliticism](https://github.com/jkskw/vehicle_routing_problem/blob/main/src/main.py#L121)* function being called and depending on *eliticism_percentage* takes that percentage of actual best chromosomes so that it's imposible to lose greatest solutions.

### Additional features

• Possibility to collect excess from customers in case of negative value in creating demand.

• Entering user values ​​for car capacities, which may be non-uniform.

• (optional) Optimization/distribution of load for the number of cars so that each of them receives a similar value. [Click here to view the figure.](#optimize)

• (optional) Forcing the car to be filled to its maximum capacity before sending another one. [Click here to view the figure.](#maximum)

• Producing a diverse genotype for the next generation through tournament selection, in which the best solution does not always emerge.

• Elitism, i.e. adding the best genes to the next generation.

• If the car cannot be used, it remains in the warehouse.

• When the car's capacity is below the average optimized for each car, the car is used to its maximum capacity and the rest are filled to the optimal value.

## Example solutions

**VRP example**
This section presents an example of a VRP in which the goal is to minimize the longest single route. Below, you'll find a GIF demonstrating the genetic algorithm solving the VRP problem.

<p align="center">
<img src=".github/VRP.gif" width="600" height="600">
</p>


Imagine a situation in which a company needs to effectively visit its customers scattered throughout a city arranged in uniform rectangular blocks. Below is a figure of the city grid, with the company's depot in black and customer locations in blue. Each customer is marked with a number next to which the load is indicated in brackets.

<p align="center">
<img src=".github/start.png" width="600" height="600">
</p>

Below is a figure illustrating the initialization of routes. In the first generation, routes are chosen randomly as part of the genetic algorithm's initialization process. The initial length of all routes is 824.99.

<p align="center">
<img src=".github/initial_routes.png" width="600" height="600">
</p>

Below, you'll find a figure of the best routes identified by the genetic algorithm, total length of all routes is optimized to 212.63 units.

<p align="center">
<img src=".github/best_routes.png" width="600" height="600">
</p>

Below is a figure of the road length per generation. The best solution was found in 95th generation.

<p align="center">
<img src=".github/road_length.png" width="600" height="600">
</p>


The console output serves as a confirmation of the solution's validity, offering key insights and routes details. It also displays the genetic algorithm's progress from the first to the 100th generation.
```
Generation 1, Best Fitness: 0.002242425632031495, Route Length: 824.99
                                .
                                .
                                .
Generation 100, Best Fitness: 0.004702955307627635, Route Length: 212.63
Elapsed time: 1.093742847442627s
Best Fitness: 0.004702955307627634
Route for red: [1, 14, 13, 15]
Route for green: [12, 8, 9, 6]
Route for blue: [10, 7, 5, 4]
Route for cyan: [16, 11, 3, 2]
Initial length traveled: 824.99
Total length traveled: 212.63
Sum of all demands: 166.0
Initial Routes: Sum of Demands for red: 41.0(50), Sum of Demands for green: 41.0(50), Sum of Demands for blue: 42.0(50), Sum of Demands for cyan: 42.0(50)
Best Routes: Sum of Demands for red: 41.0(50), Sum of Demands for green: 42.0(50), Sum of Demands for blue: 41.0(50), Sum of Demands for cyan: 42.0(50)
```

**Load optimization vs. maximum car load**
In this section, two different vehicle load distribution strategies will be presented.

The maximum car load approach focuses on filling each vehicle to its maximum capacity, ensuring efficient use of resources within set constraints. The routes can be found in the figure below.

<p align="center" id="maximum">
<img src=".github/without_optimize.png" width="600" height="600">
</p>

As an option, the system can evenly distribute cargo between vehicles, ensuring that each carries a similar amount of cargo. Below is a graphical representation illustrating the optimized distribution of routes.

<p align="center" id="optimize">
<img src=".github/with_optimize.png" width="600" height="600">
</p>

## License
Distributed under the MIT License. See [`LICENSE`](/LICENSE) for more information.
