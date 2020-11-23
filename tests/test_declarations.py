from belabot.engine.declarations import (
    get_player_declarations,
    RankDeclaration,
    SuitDeclaration,
)
from belabot.engine.card import Card, Suit, Rank
import itertools as it
import more_itertools as mit


def test_single_suit_declaration():
    for run_length in range(3, 9):
        for suit in Suit:
            for ranks in mit.windowed(Rank, run_length):
                cards = [Card(rank=rank, suit=suit) for rank in ranks]
                assert len(cards) == len(ranks)
                declarations = get_player_declarations(cards)
                assert len(declarations) == 1
                assert SuitDeclaration(ranks[-1], suit, run_length) in declarations
    return


def test_multiple_suit_one_declarations():
    for run_length1, run_length2 in filter(
        lambda x: x[0] + x[1] <= 8, it.combinations(range(3, 9), 2)
    ):
        for suit1, suit2 in it.combinations(Suit, 2):
            # testing across the suits
            assert suit1 != suit2
            for ranks1, ranks2 in it.product(
                mit.windowed(Rank, run_length1), mit.windowed(Rank, run_length2)
            ):
                cards = [Card(rank=rank, suit=suit1) for rank in ranks1] + [
                    Card(rank=rank, suit=suit2) for rank in ranks2
                ]
                assert len(cards) == run_length1 + run_length2
                declarations = get_player_declarations(cards)
                assert len(declarations) == 2
                assert SuitDeclaration(ranks1[-1], suit1, run_length1) in declarations
                assert SuitDeclaration(ranks2[-1], suit2, run_length2) in declarations
    return


def test_one_suit_multiple_declarations():
    for suit in Suit:
        for run_length1, run_length2 in filter(
            lambda t: t[0] + t[1] < 8, it.product(range(3, 5), range(3, 5))
        ):
            ranks_gen1 = mit.windowed(Rank, run_length1)
            ranks_gen2 = mit.windowed(Rank, run_length2)
            product = it.product(ranks_gen1, ranks_gen2)
            # generated ranks should not be neighbours
            # this means run length of 3 and 3 must not make
            # one continuous length of 6
            # hence the first two conditions
            compare = lambda pair: (
                abs(max(pair[0]).value - min(pair[1]).value) > 1
                and abs(max(pair[1]).value - min(pair[0]).value) > 1
                and len(set(pair[0]) & set(pair[1])) == 0
            )
            for ranks1, ranks2 in filter(compare, product):
                cards = [Card(rank=rank, suit=suit) for rank in ranks1] + [
                    Card(rank=rank, suit=suit) for rank in ranks2
                ]
                assert len(cards) == (run_length1 + run_length2)
                declarations = get_player_declarations(cards)
                assert len(declarations) == 2, print(f"{declarations}, {cards}")
                assert SuitDeclaration(ranks1[-1], suit, run_length1) in declarations
                assert SuitDeclaration(ranks2[-1], suit, run_length2) in declarations
    return
