from belabot.engine.belot import Belot, Player

players = [Player() for _ in range(4)]
belot = Belot(players)
belot.play()
