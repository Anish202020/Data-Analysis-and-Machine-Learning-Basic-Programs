import random
import numpy
import numpy as np
from deap import algorithms,base,creator,tools

c = np.zeros((2,1))
def cost(x):
    L = len(x)
    for i in range(0,L):
        c[i]=pow(x[i],2)*pow(np.sin(x[i]),2)
    return c

creator.create("FitnessMax",base.Fitness,weights = (1.0,))
creator.create('Individual',numpy.ndarray,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_x",random.uniform,-6,6)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_x,2)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
toolbox.register('evaluate',cost)
toolbox.register('mate',tools.cxTwoPoint)
toolbox.register('mutate',tools.mutFlipBit,indpb=0.05)
toolbox.register('select',tools.selTournament,tournsize=3)

def main():
    random.seed(64)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1,similar = numpy.array_equal)
    
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("Max",numpy.max)
    
    algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=30,stats=stats,halloffame=hof)
    
    best = tools.selBest(pop,1)[0]
    print(best)
    return pop , stats , hof

if __name__ == "__main__":
    main()