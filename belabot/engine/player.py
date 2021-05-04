import abc
import enum
import itertools as it
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_discounted_cumsum import discounted_cumsum_right

# from .player import 'Player'
from .card import Adut, Card, Suit
from .declarations import Declaration
from .util import get_valid_moves, one_hot_encode

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


def indices_to_mask(indices, num_of_cards=32):
    mask = torch.zeros(num_of_cards, dtype=float)
    mask[indices] = 1.0
    return mask


class PolicyGradientLoss(nn.Module):
    def forward(self, log_action_probabilities, discounted_rewards):
        losses = - discounted_rewards * log_action_probabilities
        loss = losses.mean()
        #print(loss.requires_grad)
        return loss

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


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.card_conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, stride=11)
        self.player_adut = nn.Linear(8, 8)
        self.fc = nn.Linear(136, 32)
        return

    def forward(self, x):
        assert x.shape == (1, 1, 360)
        card_features = self.card_conv(x[..., :352])
        assert card_features.shape == (1, 4, 32)
        player_adut_features = self.player_adut(x[..., 352:])
        assert player_adut_features.shape == (1, 1, 8)
        cat = torch.cat((card_features.view(1, -1), player_adut_features.view(1, -1)), dim=-1)    # (1, 136)
        assert cat.shape == (1, 136)
        x = torch.tanh(cat)
        logits = self.fc(x)
        assert logits.shape == (1, 32)
        return logits.view(-1)

class Brain(abc.ABC):
    def __init__(self, input_size):
        self.model = Model(input_size)
        self.model.train()
        self.logprobs_per_player = defaultdict(list)
        self.rewards_per_player = defaultdict(list)
        self.loss = PolicyGradientLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9)
        self.optimizer.zero_grad()
        self.gamma = 1.0
        return

    def set_rewards(self, player, rewards_list):
        self.rewards_per_player[player].extend(rewards_list)
        return

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

    def make_move(self, player, state, allowed_cards):
        allowed_indices = torch.tensor([card.to_int() for card in allowed_cards])
        #print(allowed_indices)
        mask = indices_to_mask(allowed_indices)
        logits = self.model(torch.FloatTensor(state).view(1, 1, -1))
        c = Categorical(logits=logits)
        c.probs *= mask
        c = Categorical(probs=c.probs)  # lowkey hacky xd
        card_idx = c.sample()
        log_prob = c.log_prob(card_idx)
        self.logprobs_per_player[player].append(log_prob)
        return card_idx.item()

    def train(self):
        assert self.rewards_per_player.keys() == self.logprobs_per_player.keys()
        # adam n shit
        logprobs = torch.cat(
            tuple(torch.tensor(lp, dtype=float, requires_grad=True) for lp in self.logprobs_per_player.values()),
            dim=-1
        )
        assert logprobs.requires_grad
        rewards = torch.cat(
            tuple(
                discounted_cumsum_right(torch.tensor(r, dtype=float).view(1, -1), self.gamma)
                for r in self.rewards_per_player.values()),
            dim=-1).view(-1)
        assert not rewards.requires_grad
        #print(logprobs)
        #print(rewards)
        loss = self.loss(logprobs, rewards)
        log.info(f"LOSS: \t{loss}")
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        for player in self.rewards_per_player.keys():
            self.rewards_per_player[player].clear()
            self.logprobs_per_player[player].clear()
        return


class BigBrain(Brain):
    def __init__(self):
        return super().__init__(360)

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
        self._brain = None
        return

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def brain(self):
        return self._brain

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

    def notify_rewards(self):
        if self.brain is not None:
            self.brain.set_rewards(self, self.points)
            self.points.clear()
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
        self._brain = brain
        return

    def get_adut(self, is_muss: bool) -> Adut:
        return Adut(random.choice(range(4 if is_muss else 5)) + 1)

    def play_card(self, turn_cards: List[Card], adut_suit: Suit) -> Card:
        possible_cards = get_valid_moves(turn_cards, self.cards, adut_suit)
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
        assert len(state_encoding) == 32 * (len(CardState) - 1) + 8
        card_id = self._brain.make_move(self, state_encoding, possible_cards)
        return Card.from_int(card_id)
