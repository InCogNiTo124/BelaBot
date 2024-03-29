import abc
import dataclasses
from functools import total_ordering
from typing import List, MutableSequence, Sequence, Set, Tuple, cast

import more_itertools as mit

from .card import Card, Rank, Suit

VALUES_SEQUENCE = {0: 0, 1: 0, 2: 0, 3: 20, 4: 50, 5: 100, 6: 100, 7: 100, 8: 1000}

VALUES_RANK = {
    # Rank.VII: 0,
    # Rank.VIII: 0,
    Rank.IX: 150,
    Rank.X: 100,
    Rank.JACK: 200,
    Rank.QUEEN: 100,
    Rank.KING: 100,
    Rank.ACE: 100,
}


@total_ordering
class Declaration:
    def value(self) -> int:
        pass

    def __lt__(self, other: "Declaration") -> bool:
        pass

    def __eq__(self, other: object) -> bool:
        pass

    def cards(self) -> List[Card]:
        return []


@dataclasses.dataclass
class RankDeclaration(Declaration):
    rank: Rank

    def value(self) -> int:
        return VALUES_RANK[self.rank]

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Declaration)
        if self.value() < other.value():
            return True
        elif isinstance(other, SuitDeclaration):
            return other.value() == 8
        elif isinstance(other, RankDeclaration):
            return self.rank.points(adut=True) < other.rank.points(adut=True)
        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RankDeclaration):
            return self.rank == other.rank
        return False

    def cards(self) -> List[Card]:
        return [Card(suit=suit, rank=self.rank) for suit in Suit]


@dataclasses.dataclass
class SuitDeclaration(Declaration):
    high_rank: Rank
    suit: Suit
    length: int

    def value(self) -> int:
        return VALUES_SEQUENCE[self.length]

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Declaration)
        if self.value() < other.value():
            return True
        elif isinstance(other, RankDeclaration):
            return self.length < 8
        elif isinstance(other, SuitDeclaration):
            return self.length < other.length or (
                self.length == other.length and self.high_rank < other.high_rank
            )
        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SuitDeclaration):
            return self.length == other.length and self.high_rank == other.high_rank
        return False

    def cards(self) -> List[Card]:
        return [
            Card(suit=self.suit, rank=Rank(i))
            for i in range(self.high_rank.value, self.high_rank.value - self.length, -1)
        ]


class DeclarationDetector(abc.ABC):
    def __init__(self) -> None:
        return

    @abc.abstractmethod
    def value(self) -> int:
        pass

    @abc.abstractmethod
    def __call__(self, cards: Set[int]) -> Sequence[Declaration]:
        pass


class SuitDeclarationDetector(DeclarationDetector):
    def __init__(self, length: int) -> None:
        self.sequence_length = length
        return

    def __call__(self, player_cards: Set[int]) -> Sequence[Declaration]:
        declarations: MutableSequence[SuitDeclaration] = []
        for suit in Suit:
            for sequence in mit.windowed(reversed(Rank), self.sequence_length):
                sequence = cast(Tuple[Rank], sequence)
                high_card = sequence[0]
                # mapping: Callable[[Rank], int] = lambda rank: Card(rank=rank, suit=suit).to_int()
                cards = set(
                    map(lambda rank: Card(rank=rank, suit=suit).to_int(), sequence)
                )
                if cards <= player_cards:
                    declarations.append(
                        SuitDeclaration(high_card, suit, self.sequence_length)
                    )
                    player_cards -= cards
        return declarations

    def value(self) -> int:
        return VALUES_SEQUENCE[self.sequence_length]


class RankDeclarationDetector(DeclarationDetector):
    def __init__(self, rank: Rank) -> None:
        self.declaration_rank = rank
        return

    def __call__(self, player_cards: Set[int]) -> Sequence[RankDeclaration]:
        cards = set(
            map(lambda suit: Card(rank=self.declaration_rank, suit=suit).to_int(), Suit)
        )
        if cards <= player_cards:
            player_cards -= cards
            return [RankDeclaration(self.declaration_rank)]
        else:
            return []

    def value(self) -> int:
        return VALUES_RANK[self.declaration_rank]


def get_player_declarations(cards: List[Card]) -> List[Declaration]:
    declarations: List[Declaration] = []
    player_cards = set(map(Card.to_int, cards))
    rank_detectors: List[DeclarationDetector] = [
        RankDeclarationDetector(rank)
        for rank, value in sorted(
            VALUES_RANK.items(), key=lambda pair: pair[0].points(), reverse=True
        )
        if value > 0
    ]
    suit_detectors: List[DeclarationDetector] = [
        SuitDeclarationDetector(length)
        for length, value in sorted(VALUES_SEQUENCE.items(), key=lambda item: -item[0])
        if value > 0
    ]
    detectors = sorted(
        suit_detectors + rank_detectors,
        key=lambda detector: (
            detector.value(),
            isinstance(detector, RankDeclarationDetector),
        ),
        reverse=True,
    )

    # please note that the variable player_cards is mutated by the Detector classes
    # this is by design to disallow using one card in multiple declarations
    for detector in detectors:
        new_declarations = detector(player_cards)
        declarations.extend(new_declarations)
        if len(player_cards) <= 2:
            break
    return declarations
