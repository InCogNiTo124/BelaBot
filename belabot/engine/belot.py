from .card import Card, Suit

from enum import Enum
import logging
import os
import random
import sys
from typing import List, Tuple, Optional

random_gen = random.SystemRandom()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())


class Adut(Enum):
    HEARTS = Suit.HEARTS.value
    DIAMONDS = Suit.DIAMONDS.value
    SPADES = Suit.SPADES.value
    CLUBS = Suit.CLUBS.value
    NEXT = 5


class Player:
    def __init__(self: "Player") -> None:
        self.cards: List[Card] = []
        return

    def set_cards(self: "Player", cards: List[int]) -> None:
        self.cards = [Card.from_int(t) for t in cards]
        return

    def get_adut(self: "Player", is_muss: bool) -> Adut:
        return Adut(random_gen.choice(range(4 if is_muss else 5)) + 1)


class Belot:
    def __init__(self: "Belot", players: List[Player]):
        self.players = players
        self.deck = range(32)
        return

    def play(self: "Belot") -> None:
        current_dealer_index = 0
        mi, vi = 0, 0
        while mi <= 1000 and vi <= 1000 or mi == vi:
            round_mi, round_vi = self.round(current_dealer_index)
            current_dealer_index = (current_dealer_index + 1) % 4
            mi += round_mi
            vi += round_vi
        log.info(f"MI {mi} \t {vi} VI")
        return

    def round(self: "Belot", dealer_index: int) -> Tuple[int, int]:
        adut = self.get_adut(dealer_index)
        log.debug(adut)
        x = random_gen.randint(0, 162)
        return x, 162 - x

    def shuffle(self: "Belot") -> None:
        deck = random_gen.sample(self.deck, len(self.deck))
        cards_per_player = len(self.deck) // len(self.players)
        for i, player in enumerate(self.players):
            player.set_cards(deck[i * cards_per_player: (i + 1) * cards_per_player])
        return

    def get_adut(self: "Belot", dealer_index: int) -> Optional[Suit]:
        for i in range(1, 5):  # the dealer calls last
            player_index = (dealer_index + i) % len(self.players)
            log.debug(player_index)
            player = self.players[player_index]
            adut = player.get_adut(is_muss=(i == 4))
            if adut != Adut.NEXT:
                return Suit(adut.value)
        return None
