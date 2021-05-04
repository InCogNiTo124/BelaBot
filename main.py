from belabot.engine.belot import Belot
from belabot.engine.player import AiPlayer, BigBrain, RandomPlayer

big_brain = BigBrain()
players = [
    AiPlayer("0", big_brain),
    RandomPlayer("1"),
    AiPlayer("2", big_brain),
    RandomPlayer("3"),
]
belot = Belot(players)
#belot.play()
for i in range(1000):
    print(i, end='\t')
    belot.round(i % 4)
