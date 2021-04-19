from .card import Card, Adut, Suit
from .util import get_valid_moves
from typing import List, Optional
import abc
import random
import sys
import os
import logging
import random

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class Player(abc.ABC):
    def __init__(self, name: Optional[str]) -> None:
        self.name: Optional[str] = name
        self.cards: List[Card] = []
        self.played: List[Card] = []
        return

    def add_cards(self, cards: List[int]) -> None:
        self.cards.extend([Card.from_int(t) for t in cards])
        return

    def clear_cards(self) -> None:
        self.cards.clear()
        return

    def notify_played(self, card: Card) -> None:
        self.played.append(card)
        self.card_played(card)
        return

    def card_accepted(self, card: Card) -> None:
        self.cards.remove(card)
        return

    def card_played(self, card: Card) -> None:
        return

    @abc.abstractmethod
    def get_adut(self, is_muss: bool) -> Adut:
        pass

    @abc.abstractmethod
    def play_card(self, turn_cards: List[Card]) -> Card:
        pass

class RandomPlayer(Player):
    def get_adut(self, is_muss: bool) -> Adut:
        log.debug("\t" + repr(self.cards))
        return Adut(random.choice(range(4 if is_muss else 5)) + 1)

    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        possible_cards = get_valid_moves(turn_cards, self.cards, adut_suit)
        random_index = random.randrange(len(possible_cards))
        return possible_cards[random_index]
