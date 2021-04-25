import itertools as it

from belabot.engine.card import Card, Rank, Suit


def test_card():
    for i, (s, r) in enumerate(it.product(Suit, Rank)):
        card = Card(r, s)
        assert i == card.to_int()
        assert Card.from_int(i) == card
    return
