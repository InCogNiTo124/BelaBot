from belabot.engine.belot import *
from belabot.engine.card import *
from belabot.engine.declarations import *
import random

while True:
    card_ints = random.sample(range(32), k=8)
    cards = list(map(Card.from_int, card_ints))
    declarations = get_player_declarations(cards)
    if len(declarations) > 1:
        print(declarations)
