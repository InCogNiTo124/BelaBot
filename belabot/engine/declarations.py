import abc
from .card import Card, Rank, Suit
from bisect import insort
from typing import Dict, List, Set, Tuple
from enum import Enum
import more_itertools as mit


VALUES_SEQUENCE = {
    0: 0,
    1: 0,
    2: 0,
    3: 20,
    4: 50,
    5: 100,
    6: 100,
    7: 100,
    8: 1000
}

VALUES_RANK = {
    Rank.VII: 0,
    Rank.VIII: 0,
    Rank.IX: 150,
    Rank.X: 100,
    Rank.JACK: 200,
    Rank.QUEEN: 100,
    Rank.KING: 100,
    Rank.ACE: 100
}

class DeclarationDetector(abc.ABC):
    def __init__(self):
        return

    @abc.abstractmethod
    def value(self):
        pass


class SuitDeclarationDetector(DeclarationDetector):
    def __init__(self, length):
        self.sequence_length = length
        return

    def __call__(self, player_cards: set):
        declarations = []
        for suit in Suit:
            for sequence in mit.windowed(reversed(Rank), self.sequence_length):
                high_card = sequence[0]
                cards = set(map(lambda rank: Card(rank=rank, suit=suit).to_int(), sequence))
                if cards <= player_cards:
                    declarations.append((high_card, suit, self.sequence_length))
                    player_cards -= cards
        return declarations

    def value(self):
        return VALUES_SEQUENCE[self.sequence_length]

class RankDeclarationDetector(DeclarationDetector):
    def __init__(self, rank):
        self.declaration_rank = rank
        return

    def __call__(self, player_cards: set):
        cards = set(map(lambda suit: Card(rank=self.declaration_rank, suit=suit).to_int(), Suit))
        if cards <= player_cards:
            return [(self.declaration_rank, self.value())]
        else:
            return []

    def value(self):
        return VALUES_RANK[self.declaration_rank]

def get_player_declarations(cards: List[Card]): # -> Tuple[List[RankDeclaration], Dict[Suit, SuitDeclaration]]:
    declarations = []
    player_cards = set(map(Card.to_int, cards))
    rank_detectors = [RankDeclarationDetector(rank) for rank, value in VALUES_RANK.items() if value > 0]
    suit_detectors = [SuitDeclarationDetector(length) for length, value in VALUES_SEQUENCE.items() if value > 0]
    # TODO: fix sorting subtleties.
    for detector in sorted(rank_detectors+suit_detectors, key=lambda detector: (detector.value(), isinstance(detector, RankDeclarationDetector)), reverse=True):
        declarations.extend(detector(player_cards))
        if len(player_cards) <= 2:
            break 
    return declarations
