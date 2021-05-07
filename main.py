from belabot.engine.belot import Belot
from belabot.engine.player import AiPlayer, BigBrain, RandomPlayer
import time
import torch

big_brain = BigBrain()
big_brain.model.eval()
players = [
    AiPlayer("0", big_brain),
    RandomPlayer("1"),
    AiPlayer("2", big_brain),
    RandomPlayer("3"),
]
belot = Belot(players)
#belot.play()
since = time.time()
for i in range(1_000):
    print(i, end='\t')
    belot.round(i % 4)
    new_time = time.time()
    #log.debug(new_time - since)
    since = new_time
big_brain.save()
