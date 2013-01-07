from __future__ import division
#Alice vs Bob in table game

import random
max_simulations = 1e6

simulation = 0
wins_alice = 0
wins_bob = 0


while simulation < max_simulations:
    #draw random p
    p = random.random()
    #draw eight trials
    Alice_wins = sum( [ random.random() < p for i in range(8) ] )
    if Alice_wins == 5:
        simulation += 1
        #This is case of 5vs3 in 8th round, lets check who wins by drawing three more
        if any( [ random.random() < p for i in range(3) ] ):
            wins_alice +=1
        else:
            wins_bob +=1
            

print "Proportion of Alice wins: %.3f."%( wins_alice/max_simulations ) 