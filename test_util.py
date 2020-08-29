from belabot.engine.card import Card
from belabot.engine.util import sort_cards_by_suit

print(sort_cards_by_suit([Card.from_int(i) for i in range( 0,  8)]))
print(sort_cards_by_suit([Card.from_int(i) for i in range( 8, 16)]))
print(sort_cards_by_suit([Card.from_int(i) for i in range(16, 24)]))
print(sort_cards_by_suit([Card.from_int(i) for i in range(24, 32)]))
