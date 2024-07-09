import random
import numpy 
from deap import algorithms,base,creator,tools

def cost(individual):
    return sum(individual)

creator.create('FitnessMax',base.Fitness,weights=(1.0,))
creator.create('Individual',numpy.ndarray , fitness = creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool",random.randint,0,1)
toolbox.register('individual',tools.initRepeat,creator.Individual,toolbox.attr_bool,n=100)
toolbox.register('population',tools.initRepeat,list,toolbox.individual)
toolbox.register('evaluate',cost)
toolbox.register('mate',tools.cxTwoPoint)
toolbox.register('mutate',tools.mutFlipBit,indpb=0.05)
toolbox.register('select',tools.selTournament,tournsize=3)

def main():
    random.seed(64)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1,similar = numpy.array_equal)
    
    stats = tools.Statistics(lambda ind : ind.fitness.values)
    stats.register("Max",numpy.max)
    
    
    algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=50,stats=stats,halloffame=hof)
    
    best = tools.selBest(pop,1)[0]
    print(best)
    return pop , stats,hof

if __name__ == "__main__":
    main()