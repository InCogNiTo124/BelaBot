from .card import Card, Adut, Suit
from .declarations import Declaration
from .util import get_valid_moves
from typing import List, Optional, Dict
from collections import defaultdict
import abc
import random
import sys
import os
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class Player(abc.ABC):
    def __init__(self, name: Optional[str]) -> None:
        self.name: Optional[str] = name
        self.cards: List[Card] = []
        self.played: Dict[Player, List[Card]] = defaultdict(list)
        self.points: List[int] = []
        self.turn_declarations: Dict[int, List[Declaration]] = dict()
        return

    def __hash__(self) -> int:
        return hash(self.name)

    def add_cards(self, cards: List[int]) -> None:
        self.cards.extend([Card.from_int(t) for t in cards])
        return

    def clear_cards(self) -> None:
        self.cards.clear()
        return

    def notify_played(self, player: 'Player', card: Card) -> None:
        self.played[player].append(card)
        self.card_played(card)
        return

    def notify_pregame(
        self,
        declarations: Dict[int, List[Declaration]],
        adut: Suit,
        adut_caller: "Player",
    ) -> None:
        self.turn_declarations = declarations
        self.turn_adut = adut
        self.turn_adut_called = adut_caller
        return

    def card_accepted(self, card: Card) -> None:
        self.cards.remove(card)
        self.notify_played(self, card)
        return

    def card_played(self, card: Card) -> None:
        return

    def notify_turn_points(self, points: int) -> None:
        self.points.append(points)
        return

    def team_setup(self, teammate: "Player", left: "Player", right: "Player") -> None:
        self.teammte = teammate
        self.left = left
        self.rigth = right
        return

    @abc.abstractmethod
    def get_adut(self, is_muss: bool) -> Adut:
        pass

    @abc.abstractmethod
    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        pass


class RandomPlayer(Player):
    def get_adut(self, is_muss: bool) -> Adut:
        log.debug("\t" + repr(self.cards))
        return Adut(random.choice(range(4 if is_muss else 5)) + 1)

    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        possible_cards = get_valid_moves(turn_cards, self.cards, adut_suit)
        random_index = random.randrange(len(possible_cards))
        return possible_cards[random_index]
