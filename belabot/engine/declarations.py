import abc
from .card import Card, Rank, Suit
from typing import cast, List, MutableSequence, Sequence, Set, Tuple
import more_itertools as mit
import dataclasses


VALUES_SEQUENCE = {0: 0, 1: 0, 2: 0, 3: 20, 4: 50, 5: 100, 6: 100, 7: 100, 8: 1000}

VALUES_RANK = {
    Rank.VII: 0,
    Rank.VIII: 0,
    Rank.IX: 150,
    Rank.X: 100,
    Rank.JACK: 200,
    Rank.QUEEN: 100,
    Rank.KING: 100,
    Rank.ACE: 100,
}


class Declaration:
    pass


@dataclasses.dataclass
class RankDeclaration(Declaration):
    rank: Rank
    value: int


@dataclasses.dataclass
class SuitDeclaration(Declaration):
    high_rank: Rank
    suit: Suit
    length: int


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
            return [RankDeclaration(self.declaration_rank, self.value())]
        else:
            return []

    def value(self) -> int:
        return VALUES_RANK[self.declaration_rank]


def get_player_declarations(cards: List[Card]) -> Sequence[Declaration]:
    declarations: MutableSequence[Declaration] = []
    player_cards = set(map(Card.to_int, cards))
    rank_detectors: List[DeclarationDetector] = [
        RankDeclarationDetector(rank)
        for rank, value in VALUES_RANK.items()
        if value > 0
    ]
    suit_detectors: List[DeclarationDetector] = [
        SuitDeclarationDetector(length)
        for length, value in sorted(VALUES_SEQUENCE.items(), key=lambda item: -item[0])
        if value > 0
    ]
    # TODO: fix sorting subtleties.
    for detector in sorted(
        suit_detectors + rank_detectors,
        key=lambda detector: (
            detector.value(),
            isinstance(detector, RankDeclarationDetector),
        ),
        reverse=True,
    ):
        declarations.extend(detector(player_cards))
        if len(player_cards) <= 2:
            break
    return declarations
