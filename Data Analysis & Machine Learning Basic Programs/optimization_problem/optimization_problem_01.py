import array
import random
import numpy

from deap import algorithms ,base,creator,tools

IND_SIZE = 17
OptTour = [15,11,8,4,1,9,10,2,14,13,16,5,7,6,12,3,0]
OptDistance = 2085
distance_map = [
                [321,321,22,12,234,45,342,12,231,56,243,543,34,0,67,6,34],
                [4, 23, 7, 39, 19, 0, 9, 14,32,321,43,21,34,65,54,23,342],
                [321,213,21,21,32,121,12,32,321,21,32,32,12,12,423,4, 23],
                [4, 23, 7, 39, 19, 0, 9, 14,4, 23, 7, 39, 19, 0, 9, 14,100],
                [321,321,22,12,234,45,342,12,231,56,243,543,34,0,67,6,34],
                [4, 23, 7, 39, 19, 0, 9, 14,32,321,43,21,34,65,54,23,342],
                [321,213,21,21,32,121,12,32,321,21,32,32,12,12,423,4, 23],
                [4, 23, 7, 39, 19, 0, 9, 14,4, 23, 7, 39, 19, 0, 9, 14,100],
                [321,321,22,12,234,45,342,12,231,56,243,543,34,0,67,6,34],
                [4, 23, 7, 39, 19, 0, 9, 14,32,321,43,21,34,65,54,23,342],
                [321,213,21,21,32,121,12,32,321,21,32,32,12,12,423,4, 23],
                [4, 23, 7, 39, 19, 0, 9, 14,4, 23, 7, 39, 19, 0, 9, 14,100],
                [321,321,22,12,234,45,342,12,231,56,243,543,34,0,67,6,34],
                [4, 23, 7, 39, 19, 0, 9, 14,32,321,43,21,34,65,54,23,342],
                [321,213,21,21,32,121,12,32,321,21,32,32,12,12,423,4, 23],
                [4, 23, 7, 39, 19, 0, 9, 14,4, 23, 7, 39, 19, 0, 9, 14,100],
                [321,213,21,21,32,121,12,32,321,21,32,32,12,12,423,4, 23]
                ]

creator.create('FitnessMin',base.Fitness,weights=(-1.0,))
creator.create('Individual',array.array,typecode = 'i',fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register('indices',random.sample,range(IND_SIZE),IND_SIZE)
toolbox.register('individual',tools.initIterate,creator.Individual,toolbox.indices)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def TSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for g1,g2  in zip(individual[0:-1],individual[1:]):
        distance += distance_map[g1][g2]
    return distance,

toolbox.register("mate",tools.cxPartialyMatched)
toolbox.register("mutate",tools.mutShuffleIndexes,indpb=0.05)
toolbox.register("select",tools.selTournament,tournsize=3)
toolbox.register("evaluate",TSP)

def main():
    random.seed(169)
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("Min",numpy.min)
    
    algorithms.eaSimple(pop,toolbox,0.7,0.2,ngen=300,stats=stats,halloffame=hof)
    
    best = tools.selBest(pop,1)[0]
    print("best is %s,%s"%(best,best.fitness.values))
    return pop,stats,hof

if __name__ =="__main__":
    main()