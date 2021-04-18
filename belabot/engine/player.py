from .card import Card, Adut
from typing import List
import random
import sys
import os
import logging
import random

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class Player:
    def __init__(self) -> None:
        self.cards: List[Card] = []
        self.played: List[Card] = []
        return

    def add_cards(self, cards: List[int]) -> None:
        self.cards.extend([Card.from_int(t) for t in cards])
        return

    def clear_cards(self) -> None:
        self.cards.clear()
        return

    def get_adut(self, is_muss: bool) -> Adut:
        log.debug("\t" + repr(self.cards))
        return Adut(random.choice(range(4 if is_muss else 5)) + 1)

    def play_card(self, turn_cards: List[Card]) -> Card:
        card = self.cards.pop(random.randrange(len(self.cards)))
        return card

    def notify_played(self, card: Card) -> None:
        self.played.append(card)
        return
