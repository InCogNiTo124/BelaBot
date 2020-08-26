from typing import List
from enum import Enum, auto

class Suit(Enum):
    HEARTS = auto()
    DIAMONDS = auto()
    SPADES = auto()
    CLUBS = auto()

    def __repr__(self):
        if self is Suit.HEARTS:
            return "<3 "
        elif self is Suit.DIAMONDS:
            return "(o)"
        elif self is Suit.SPADES:
            return "-\u0190>"
        elif self is Suit.CLUBS:
            return "-\u0190\u22fb"
        else:
            raise ValueError(f"Not a Suit: {repr(self)}")

class Rank(Enum):
    VII = auto()
    VIII = auto()
    IX = auto()
    X = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    ACE = auto()

    def __repr__(self):
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


class Player():
    def __init__(self):
        return

    def get_adut(self):
        return

class Belot():
    def __init__(self, players: List[Player]):
        self.players = players
        return

    def play(self):
        self.shuffle()
        self.adut = self.get_adut()
        return

    def get_adut(self):
        for i, player in enumerate(self.players):
            adut = player.get_adut()
            return

    def shuffle(self):

        return

