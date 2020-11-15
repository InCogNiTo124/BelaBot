from belabot.engine.declarations import (
    get_player_declarations,
    SuitDeclaration,
)
from belabot.engine.card import Card, Suit, Rank


def test_declarations_19_vulgaris():
    # 7,8,9 in HEARTS
    cards = [Card.from_int(i) for i in range(3)]
    declarations = get_player_declarations(cards)
    assert len(declarations) == 1
    assert declarations[0] == SuitDeclaration(Rank.IX, Suit.HEARTS, 3)
    return


def test_declarations_god():
    # A K Q in DIAMONDS
    cards = [Card.from_int(i) for i in range(15, 12, -1)]
    declarations = get_player_declarations(cards)
    assert len(declarations) == 1
    assert declarations[0] == SuitDeclaration(Rank.ACE, Suit.DIAMONDS, 3)
    return
