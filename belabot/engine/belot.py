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
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class Adut(Enum):
    HEARTS = Suit.HEARTS.value
    DIAMONDS = Suit.DIAMONDS.value
    SPADES = Suit.SPADES.value
    CLUBS = Suit.CLUBS.value
    NEXT = 5


class Player:
    def __init__(self) -> None:
        self.cards: List[Card] = []
        return

    def add_cards(self, cards: List[int]) -> None:
        self.cards.extend([Card.from_int(t) for t in cards])
        return

    def clear_cards(self) -> None:
        self.cards.clear()
        return

    def get_adut(self, is_muss: bool) -> Adut:
        print('\t', self.cards)
        return Adut(random_gen.choice(range(4 if is_muss else 5)) + 1)


class Belot:
    def __init__(self, players: List[Player]):
        self.players = players
        self.deck = range(32)
        return

    def play(self) -> None:
        current_dealer_index = 0
        mi, vi = 0, 0
        while mi <= 1000 and vi <= 1000 or mi == vi:
            round_mi, round_vi = self.round(current_dealer_index)
            current_dealer_index = (current_dealer_index + 1) % 4
            mi += round_mi
            vi += round_vi
        log.info(f"MI {mi} \t {vi} VI")
        return

    def round(self, dealer_index: int) -> Tuple[int, int]:
        first_6, talons = self.shuffle()
        for cards, player in zip(first_6, self.players):
            player.add_cards(cards)
        adut = self.get_adut(dealer_index)
        log.debug(adut)
        for talon, player in zip(talons, self.players):
            player.add_cards(talon)
        x = random_gen.randint(0, 162)
        for player in self.players:
            player.clear_cards()
        return x, 162 - x

    def shuffle(self) -> Tuple[List[List[int]], List[List[int]]]:
        # Usually, cards in Bela are dealed in a particular order
        # this kinda makes sense in a real world where not all
        # permutations are equally likely
        # however, I'm using OS provided source of randomness
        # which is guaranteed to have enough entropy
        # therefore, with simple math, one concludes that there
        # is no advantage of simulating the deal rule in bela
        # as opposed to just allocating cards contiguously
        deck = random_gen.sample(self.deck, len(self.deck))
        cards_per_player = len(self.deck) // len(self.players)
        adut_cards = []
        talons = []
        for i in range(0, len(deck), cards_per_player):
            player_cards_end = i + cards_per_player
            talon_start = player_cards_end - 2
            adut_cards.append(deck[i:talon_start])
            talons.append(deck[talon_start:player_cards_end])
        return adut_cards, talons

    def get_adut(self, dealer_index: int) -> Optional[Suit]:
        for i in range(1, 1+len(self.players)):  # the dealer calls last
            player_index = (dealer_index + i) % len(self.players)
            log.debug(player_index)
            player = self.players[player_index]
            adut = player.get_adut(is_muss=(i == 4))
            if adut != Adut.NEXT:
                return Suit(adut.value)
        return None
