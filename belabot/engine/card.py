from enum import Enum, auto
from functools import total_ordering


@total_ordering
class Suit(Enum):
    HEARTS = auto()
    DIAMONDS = auto()
    SPADES = auto()
    CLUBS = auto()

    def __repr__(self: "Suit") -> str:
        if self is Suit.HEARTS:
            return " <3"
        elif self is Suit.DIAMONDS:
            return "(o)"
        elif self is Suit.SPADES:
            return "-\u0190>"
        elif self is Suit.CLUBS:
            return "-\u0190\u22fb"
        else:
            raise ValueError(f"Not a Suit: {repr(self)}")

    def __lt__(self: "Suit", other: object) -> bool:
        return isinstance(other, Suit) and self.value < other.value

    def __eq__(self: "Suit", other: object) -> bool:
        return isinstance(other, Suit) and self.value == other.value

    def __hash__(self: "Suit") -> int:
        return hash(self.value)


@total_ordering
class Rank(Enum):
    VII = auto()
    VIII = auto()
    IX = auto()
    X = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    ACE = auto()

    def __repr__(self: "Rank") -> str:
        if self is Rank.VII:
            return "7"
        elif self is Rank.VIII:
            return "8"
        elif self is Rank.IX:
            return "9"
        elif self is Rank.X:
            return "X"
        elif self is Rank.JACK:
            return "J"
        elif self is Rank.QUEEN:
            return "Q"
        elif self is Rank.KING:
            return "K"
        elif self is Rank.ACE:
            return "A"
        else:
            raise ValueError("Not a Rank: {repr(self}")

    def __lt__(self: "Rank", other: object) -> bool:
        return isinstance(other, Rank) and self.value < other.value

    def __eq__(self: "Rank", other: object) -> bool:
        return isinstance(other, Rank) and self.value == other.value

    def __hash__(self: "Rank") -> int:
        return hash(self.value)


@total_ordering
class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
        return

    @classmethod
    def from_int(cls, number: int) -> "Card":
        suit = number // len(Rank) + 1
        rank = number % len(Rank) + 1
        return cls(Rank(rank), Suit(suit))

    def to_int(self: "Card") -> int:
        return (self.suit.value - 1) * len(Rank) + self.rank.value - 1

    def __repr__(self: "Card") -> str:
        return f"{repr(self.rank)} {repr(self.suit)}"

    def __lt__(self: "Card", other: object) -> bool:
        return isinstance(other, Card) and self.to_int() < other.to_int()

    def __eq__(self: "Card", other: object) -> bool:
        return isinstance(other, Card) and self.to_int() == other.to_int()
