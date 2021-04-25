import abc
import enum
import itertools as it
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

# from .player import 'Player'
from .card import Adut, Card, Suit
from .declarations import Declaration
from .util import get_valid_moves, one_hot_encode

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class CardState(enum.Enum):
    UNKNOWN = 0  # there is no way of knowing where the card is
    LEFT = 1  # the card is in the left player's hands (deduced from declarations)
    TEAMMATE = 2  # the card is in the teammate's hands (deduced from declarations)
    RIGHT = 3  # the card is in the right player's hands (deduced from declarations)
    ME = 4  # the card is in the players hands (deduced from actually having the cards)
    TURN_LEFT = 5  # the card was played in this turn by left
    TURN_TEAMMATE = 6  # the card was played in this turn by the teammate
    TURN_RIGHT = 7  # the card was played in this turn by right
    PLAYED_LEFT = 8  # the card was played in the past by left
    PLAYED_TEAMMATE = 9  # the card was played in the past by the teammate
    PLAYED_RIGHT = 10  # the card was played in the past by right
    PLAYED_ME = 11  # the card was played in the past by me


class Brain(abc.ABC):
    @abc.abstractmethod
    def get_state_representation(
        self,
        current_player: "Player",
        cards_played: Dict["Player", List[Card]],
        cards_turn: Dict["Player", Card],
        player_cards: List[Card],
        declarations: Dict["Player", List[Declaration]],
        adut_suit: Suit,
        adut_caller: "Player",
    ) -> List[int]:
        pass


class BigBrain(Brain):
    def get_state_representation(
        self,
        current_player: "Player",
        cards_played: Dict["Player", List[Card]],
        cards_turn: Dict["Player", Card],
        player_cards: List[Card],
        declarations: Dict["Player", List[Declaration]],
        adut_suit: Suit,
        adut_caller: "Player",
    ) -> List[int]:
        assert (len(cards_turn) == 0) ^ (current_player.left in cards_turn.keys())
        # print(declarations)
        cards_state = {
            card: CardState.UNKNOWN for card in map(Card.from_int, range(32))
        }  # GOD I love Python
        declaration_map = {
            current_player: CardState.ME,
            current_player.left: CardState.LEFT,
            current_player.right: CardState.RIGHT,
            current_player.teammate: CardState.TEAMMATE,
        }
        played_map = {
            current_player: CardState.PLAYED_ME,
            current_player.left: CardState.PLAYED_LEFT,
            current_player.right: CardState.PLAYED_RIGHT,
            current_player.teammate: CardState.PLAYED_TEAMMATE,
        }

        turn_map = {
            current_player.left: CardState.TURN_LEFT,
            current_player.right: CardState.TURN_RIGHT,
            current_player.teammate: CardState.TURN_TEAMMATE,
        }
        adut_map = {
            current_player: 1,
            current_player.right: 2,
            current_player.teammate: 3,
            current_player.left: 4,
        }

        # handle deducing card locations from declarations
        for player, declaration_list in declarations.items():
            for declaration in declaration_list:
                for card in declaration.cards():
                    cards_state[card] = declaration_map[player]

        # handle player cards
        for card in player_cards:
            cards_state[card] = CardState.ME

        # handle played cards
        for player, cards in cards_played.items():
            for card in cards:
                cards_state[card] = played_map[player]

        # handle turn cards
        for player, card in cards_turn.items():
            cards_state[card] = turn_map[player]
        cards_encoded = list(
            one_hot_encode(t.value, len(CardState)) for t in cards_state.values()
        )
        adut_encoded = one_hot_encode(adut_suit.value + 1, len(Suit) + 1)
        adut_player_encoded = one_hot_encode(adut_map[adut_caller], 5)
        return list(
            it.chain.from_iterable(
                cards_encoded + [adut_encoded] + [adut_player_encoded]
            )
        )


class Player(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name: Optional[str] = name
        self.cards: List[Card] = []
        self.played: Dict[Player, List[Card]] = defaultdict(list)
        self.points: List[int] = []
        self.round_declarations: Dict[Player, List[Declaration]] = dict()
        return

    def __hash__(self) -> int:
        return hash(self.name)

    def add_cards(self, cards: List[int]) -> None:
        self.cards.extend([Card.from_int(t) for t in cards])
        return

    def clear_cards(self) -> None:
        self.cards.clear()
        return

    def notify_played(self, player: "Player", card: Card) -> None:
        self.played[player].append(card)
        self.card_played(card)
        return

    def notify_pregame(
        self,
        declarations: Dict["Player", List[Declaration]],
        adut: Suit,
        adut_caller: "Player",
    ) -> None:
        self.round_declarations = declarations
        self.round_adut = adut
        self.round_adut_caller = adut_caller
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
        self.teammate = teammate
        self.left = left
        self.right = right
        return

    @abc.abstractmethod
    def get_adut(self, is_muss: bool) -> Adut:
        pass

    @abc.abstractmethod
    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        pass


class RandomPlayer(Player):
    def get_adut(self, is_muss: bool) -> Adut:
        # log.debug("\t" + repr(self.cards))
        return Adut(random.choice(range(4 if is_muss else 5)) + 1)

    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        possible_cards = get_valid_moves(turn_cards, self.cards, adut_suit)
        random_index = random.randrange(len(possible_cards))
        return possible_cards[random_index]


class AiPlayer(Player):
    def __init__(self, name: str, brain: Brain) -> None:
        super().__init__(name)
        self.brain = brain
        return

    def get_adut(self, is_muss: bool) -> Adut:
        return Adut(random.choice(range(4 if is_muss else 5)) + 1)

    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        state_encoding = self.brain.get_state_representation(
            self,
            self.played,
            {
                player: card
                for player, card in zip(
                    [self.left, self.teammate, self.right], reversed(turn_cards)
                )
            },
            self.cards,
            self.round_declarations,
            self.round_adut,
            self.round_adut_caller,
        )
        assert len(state_encoding) == 360
        possible_cards = get_valid_moves(turn_cards, self.cards, adut_suit)
        random_index = random.randrange(len(possible_cards))
        return possible_cards[random_index]
