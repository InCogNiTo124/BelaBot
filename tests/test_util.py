from belabot.engine.util import calculate_points, get_valid_moves, get_winner
from belabot.engine.card import Card, Suit, Rank


def test_calculate_points_no_decl():
    assert calculate_points(82, True, 0, 0) == (82, 80)
    assert calculate_points(81, True, 0, 0) == (0, 162)
    assert calculate_points(80, True, 0, 0) == (0, 162)

    assert calculate_points(82, False, 0, 0) == (162, 0)
    assert calculate_points(81, False, 0, 0) == (162, 0)
    assert calculate_points(80, False, 0, 0) == (80, 82)
    return


def test_calculate_points_decl():
    assert calculate_points(81, True, 50, 0) == (131, 81)
    assert calculate_points(81, True, 20, 0) == (101, 81)
    assert calculate_points(90, True, 0, 20) == (0, 182)

    assert calculate_points(82, False, 0, 20) == (82, 100)
    assert calculate_points(82, False, 0, 50) == (82, 130)
    assert calculate_points(80, False, 20, 0) == (182, 0)
    return


def test_calculate_points_stiglja():
    assert calculate_points(162, True, 0, 0) == (252, 0)
    assert calculate_points(162, False, 0, 0) == (252, 0)
    assert calculate_points(162, True, 20, 0) == (272, 0)
    assert calculate_points(162, False, 0, 50) == (302, 0)
    assert calculate_points(162, True, 20, 50) == (322, 0)
    assert calculate_points(162, False, 20, 50) == (322, 0)

    assert calculate_points(0, True, 0, 0) == (0, 252)
    assert calculate_points(0, False, 0, 0) == (0, 252)
    assert calculate_points(0, True, 20, 0) == (0, 272)
    assert calculate_points(0, False, 0, 50) == (0, 302)
    assert calculate_points(0, True, 20, 50) == (0, 322)
    assert calculate_points(0, False, 20, 50) == (0, 322)


def test_valid_moves():
    player_cards = [
        Card(Rank.IX, Suit.HEARTS),
        Card(Rank.X, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.VII, Suit.SPADES),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.VIII, Suit.CLUBS),
    ]
    for suit in Suit:
        assert get_valid_moves([], player_cards, suit) == player_cards

    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
        assert get_valid_moves([Card(Rank.VIII, Suit.SPADES)], player_cards, suit) == [
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.ACE, Suit.SPADES),
        ]
        assert get_valid_moves([Card(Rank.KING, Suit.SPADES)], player_cards, suit) == [
            Card(Rank.ACE, Suit.SPADES)
        ]

    assert get_valid_moves([Card(Rank.VII, Suit.CLUBS)], player_cards, Suit.CLUBS) == [
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.VIII, Suit.CLUBS),
    ]
    assert get_valid_moves(
        [Card(Rank.QUEEN, Suit.CLUBS)], player_cards, Suit.CLUBS
    ) == [Card(Rank.KING, Suit.CLUBS)]
    assert get_valid_moves([Card(Rank.IX, Suit.CLUBS)], player_cards, Suit.CLUBS) == [
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.VIII, Suit.CLUBS),
    ]

    for rank in Rank:
        assert get_valid_moves(
            [Card(rank, Suit.DIAMONDS)], player_cards, Suit.HEARTS
        ) == [
            Card(Rank.IX, Suit.HEARTS),
            Card(Rank.X, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
        ]
        assert get_valid_moves(
            [Card(rank, Suit.DIAMONDS)], player_cards, Suit.SPADES
        ) == [
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.VII, Suit.SPADES),
        ]
        assert get_valid_moves(
            [Card(rank, Suit.DIAMONDS)], player_cards, Suit.CLUBS
        ) == [Card(Rank.KING, Suit.CLUBS), Card(Rank.VIII, Suit.CLUBS)]

        assert (
            get_valid_moves([Card(rank, Suit.DIAMONDS)], player_cards, Suit.DIAMONDS)
            == player_cards
        )

    # two cards
    # uber cards without adut
    for adut_suit in [Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS]:
        assert (
            get_valid_moves(
                [Card(Rank.JACK, Suit.HEARTS), Card(Rank.VIII, Suit.HEARTS)],
                player_cards,
                adut_suit,
            )
            == [Card(Rank.X, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)]
        )
        assert (
            get_valid_moves(
                [Card(Rank.JACK, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)],
                player_cards,
                adut_suit,
            )
            == [Card(Rank.X, Suit.HEARTS)]
        )

    # cutting
    for adut_suit in [Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS]:
        assert get_valid_moves(
            [Card(Rank.JACK, Suit.HEARTS), Card(Rank.JACK, adut_suit)],
            player_cards,
            adut_suit,
        ) == [
            Card(Rank.IX, Suit.HEARTS),
            Card(Rank.X, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
        ]

    # no color
    assert get_valid_moves(
        [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.IX, Suit.DIAMONDS)],
        player_cards,
        Suit.SPADES,
    ) == [
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.VII, Suit.SPADES),
    ]

    # second card is a filter for aduts
    assert (
        get_valid_moves(
            [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.VIII, Suit.SPADES)],
            player_cards,
            Suit.SPADES,
        )
        == [Card(Rank.QUEEN, Suit.SPADES), Card(Rank.ACE, Suit.SPADES)]
    )
    assert (
        get_valid_moves(
            [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.X, Suit.SPADES)],
            player_cards,
            Suit.SPADES,
        )
        == [Card(Rank.ACE, Suit.SPADES)]
    )
    assert get_valid_moves(
        [Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.IX, Suit.SPADES)],
        player_cards,
        Suit.SPADES,
    ) == [
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.VII, Suit.SPADES),
    ]

    return


def test_get_winner():
    # all cards same suit, suit not adut
    for suit in [Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS]:
        assert (
            get_winner(
                [
                    Card(Rank.QUEEN, Suit.HEARTS),
                    Card(Rank.KING, Suit.HEARTS),
                    Card(Rank.ACE, Suit.HEARTS),
                    Card(Rank.JACK, Suit.HEARTS),
                ],
                suit,
            )
            == 2
        )

    # all cards same, suit is adut
    assert (
        get_winner(
            [
                Card(Rank.QUEEN, Suit.HEARTS),
                Card(Rank.KING, Suit.CLUBS),
                Card(Rank.ACE, Suit.CLUBS),
                Card(Rank.JACK, Suit.CLUBS),
            ],
            Suit.SPADES,
        )
        == 0
    )

    # adut
    assert (
        get_winner(
            [
                Card(Rank.QUEEN, Suit.HEARTS),
                Card(Rank.VII, Suit.SPADES),
                Card(Rank.ACE, Suit.CLUBS),
                Card(Rank.ACE, Suit.HEARTS),
            ],
            Suit.SPADES,
        )
        == 1
    )
    assert (
        get_winner(
            [
                Card(Rank.QUEEN, Suit.HEARTS),
                Card(Rank.VII, Suit.SPADES),
                Card(Rank.VIII, Suit.SPADES),
                Card(Rank.ACE, Suit.HEARTS),
            ],
            Suit.SPADES,
        )
        == 2
    )

    return
