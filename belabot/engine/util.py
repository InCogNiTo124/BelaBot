from typing import Tuple, List
from .card import Card, Suit
from .player import Player

DECK_TOTAL = 162
STIGLJA_PENALTY = 90


def calculate_points(
    mi_points_raw: int, mi_bid: bool, mi_declarations: int, vi_declarations: int
) -> Tuple[int, int]:
    if mi_points_raw == 0:
        return 0, DECK_TOTAL + STIGLJA_PENALTY + mi_declarations + vi_declarations
    elif mi_points_raw == DECK_TOTAL:
        return DECK_TOTAL + STIGLJA_PENALTY + mi_declarations + vi_declarations, 0

    mi_points = mi_points_raw + mi_declarations
    vi_points = (DECK_TOTAL - mi_points_raw) + vi_declarations
    if mi_bid:
        if mi_points > vi_points:
            return mi_points, vi_points
        else:
            return 0, mi_points + vi_points
    else:
        if vi_points > mi_points:
            return mi_points, vi_points
        else:
            return mi_points + vi_points, 0


def get_valid_moves(
    turn_cards: List[Card], player_cards: List[Card], adut: Suit
) -> List[Card]:
    if len(turn_cards) == 0:
        # first card is always a valid move
        return player_cards
    # Player must always respect the Suit of the first card, if possible
    first_card = turn_cards[0]
    cards_in_suit = [card for card in player_cards if card.suit == first_card.suit]
    if len(cards_in_suit) > 0:
        # The player has at least one card matching the suit of the first card
        # It is, then, determined if a player has a card that is of greater value
        strongest_card = max(
            card for card in turn_cards if card.suit == first_card.suit
        )
        uber_cards = [
            card
            for card in cards_in_suit
            if card.points(adut) >= strongest_card.points(adut)
            and card > strongest_card
        ]
        return uber_cards if len(uber_cards) > 0 else cards_in_suit
        # TODO if adut is played, the player doesn't have to play uber card

    # The player doesn't have any cards which match the first card's suit.
    # Therefore, the player must play an adut, if possible
    player_aduts = [card for card in player_cards if card.suit == adut]
    if len(player_aduts) > 0:
        # the player has aduts
        turn_aduts = [card for card in turn_cards if card.suit == adut]
        if len(turn_aduts) > 0:
            # some aduts have been played. The player must play an uber adut, if possible
            strongest_adut = max(turn_aduts)
            uber_aduts = [
                card
                for card in player_aduts
                if card.points(adut) >= strongest_adut.points(adut)
                and card > strongest_adut
            ]
            return uber_aduts if len(uber_aduts) > 0 else player_aduts

        # There are no aduts played. Player can play whatever adut
        return player_aduts
    else:
        # The player does not have neither matching suit nor an adut.
        # Therefore, the player can choose whatever card
        return player_cards
