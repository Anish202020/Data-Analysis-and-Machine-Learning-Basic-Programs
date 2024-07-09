import numpy as np
import operator
import random
from deap import base,creator,tools

creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
creator.create("Particle",list,fitness=creator.FitnessMin,speed=list,smin=None,smax=None,best=None)

def cost(x):
    return x*pow(np.sin(x),2)

def generate(size,pmin,pmax,smin,smax):
    part = creator.Particle(random.uniform(pmin,pmax) for _ in range(size))
    part.speed = [random.uniform(smin,smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part,best,phi1,phi2):
    u1 = (random.uniform(0,phi1) for _ in range(len(part)))
    u2 = (random.uniform(0,phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul,u1,map(operator.sub,part.best,part))
    v_u2 = map(operator.mul,u2,map(operator.sub,best,part))
    part.speed = list(map(operator.add,part.speed,map(operator.add , v_u1,v_u2)))
    
    for i , speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add,part,part.speed))

toolbox = base.Toolbox()
toolbox.register("particle",generate,size=1,pmin=-5,pmax=+5 ,smin=-3,smax=+3)
toolbox.register("population",tools.initRepeat,list,toolbox.particle)
toolbox.register("update",updateParticle,phi1=2.0,phi2=2.0)
toolbox.register("evaluate",cost)

def main():
    random.seed(64)
    pop = toolbox.population(n=5)
    
    GEN = 1000
    best = None
    
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
    for part in pop:
        toolbox.update(part,best)
        
    print(" %s Best= %s , %s"%(g, best,best.fitness))
    
if __name__ == "__main__":
    main()