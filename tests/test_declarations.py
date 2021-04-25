from belabot.engine.declarations import (
    get_player_declarations,
    RankDeclaration,
    SuitDeclaration,
    VALUES_RANK,
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
    def compare(pair):
        return (
            abs(max(pair[0]).value - min(pair[1]).value) > 1
            and abs(max(pair[1]).value - min(pair[0]).value) > 1
            and len(set(pair[0]) & set(pair[1])) == 0
        )

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


def test_single_rank_declaration():
    for rank in VALUES_RANK:
        cards = [Card(rank=rank, suit=suit) for suit in Suit]
        assert len(cards) == 4
        declarations = get_player_declarations(cards)
        assert len(declarations) == 1
        assert RankDeclaration(rank) in declarations
    return


def test_two_rank_declarations():
    for rank1, rank2 in it.combinations(VALUES_RANK, 2):
        cards = [Card(rank=rank1, suit=suit) for suit in Suit] + [
            Card(rank=rank2, suit=suit) for suit in Suit
        ]
        assert len(cards) == 8
        declarations = get_player_declarations(cards)
        assert len(declarations) == 2
        assert RankDeclaration(rank1) in declarations
        assert RankDeclaration(rank2) in declarations
    return


def test_rank_suit3():
    run_length = 3
    for suit in Suit:
        for ranks in mit.windowed(Rank, run_length):
            ranks = set(ranks)
            cards = set(Card(suit=suit, rank=rank) for rank in ranks)
            possible_ranks = set(VALUES_RANK.keys()) & ranks
            for the_rank in possible_ranks:
                round_cards = cards | set(
                    Card(rank=the_rank, suit=suit) for suit in Suit
                )
                declarations = get_player_declarations(list(round_cards))
                assert len(declarations) == 1
                assert RankDeclaration(the_rank) in declarations
    return


def test_rank_suit4():
    run_length = 4
    for suit in Suit:
        for ranks in mit.windowed(Rank, run_length):
            cards = set(Card(suit=suit, rank=rank) for rank in ranks)
            # restrict the inputs to disallow possible sub declarations
            # eg 7 8 9 X X X X can technically be 2 different declarations
            # we want to avoid that
            ranks = set(ranks[1:-1])
            possible_ranks = set(VALUES_RANK.keys()) & ranks
            for the_rank in possible_ranks:
                round_cards = cards | set(
                    Card(rank=the_rank, suit=suit) for suit in Suit
                )
                declarations = get_player_declarations(list(round_cards))
                assert len(declarations) == 1
                assert RankDeclaration(the_rank) in declarations
    return


def test_rank_suit5():
    run_length = 5
    for suit in Suit:
        for ranks in mit.windowed(Rank, run_length):
            cards = set(Card(suit=suit, rank=rank) for rank in ranks)
            # restrict the inputs to disallow possible sub declarations
            # eg 7 8 9 X J J J J can technically be 2 different declarations
            # we want to avoid that
            ranks = set(ranks[2:-2])
            possible_ranks = set(VALUES_RANK.keys()) & ranks
            for the_rank in possible_ranks:
                round_cards = cards | set(
                    Card(rank=the_rank, suit=suit) for suit in Suit
                )
                declarations = get_player_declarations(list(round_cards))
                assert len(declarations) == 1
                assert RankDeclaration(the_rank) in declarations
    return


def test_declaration_order_rank():
    for rank1 in VALUES_RANK:
        for rank2 in VALUES_RANK:
            assert (rank1.points(adut=True) > rank2.points(adut=True)) == (
                RankDeclaration(rank1) > RankDeclaration(rank2)
            )
    return


def test_declaration_order_suit():
    for length1 in range(3, 8 + 1):
        ranks1 = list(Rank)[length1-1:]
        for length2 in range(3, 8 + 1):
            ranks2 = list(Rank)[length2-1:]
            for (rank1, rank2) in it.product(ranks1, ranks2):
                for (suit1, suit2) in it.product(Suit, Suit):
                    decl1 = SuitDeclaration(rank1, suit1, length1)
                    decl2 = SuitDeclaration(rank2, suit2, length2)
                    assert (decl1 > decl2) == (
                        (length1 > length2) or (length1 == length2 and rank1 > rank2)
                    )
    return


def test_declarations_order_both():
    for suit_length in range(3, 8 + 1):
        suit_ranks = list(Rank)[suit_length-1:]
        for rank_rank in VALUES_RANK:
            rank_decl = RankDeclaration(rank_rank)
            for suit_rank in suit_ranks:
                for suit_suit in Suit:
                    suit_decl = SuitDeclaration(suit_rank, suit_suit, suit_length)
                    assert suit_decl != rank_decl
                    assert (
                        suit_decl.value() > rank_decl.value() and suit_decl > rank_decl
                    ) != (
                        rank_decl.value() >= suit_decl.value() and rank_decl > suit_decl
                    )
    return
