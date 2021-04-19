from belabot.engine.belot import Belot
from belabot.engine.player import RandomPlayer

players = [RandomPlayer(f"{i}") for i in range(4)]
belot = Belot(players)
belot.play()
