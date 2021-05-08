import abc
import enum
import itertools as it
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def cumsum_reverse(x, dim=-1):
    return x + torch.sum(x, dim=dim, keepdims=True) - torch.cumsum(x, dim=dim)


def indices_to_mask(indices, num_of_cards=32, device=torch.device('cuda')):
    mask = torch.zeros(num_of_cards, dtype=torch.float32, device=device, requires_grad=False)
    mask[indices] = 1.0
    assert not mask.requires_grad
    return mask


class PolicyGradientLoss(nn.Module):
    def forward(self, log_action_probabilities, discounted_rewards):
        losses = -discounted_rewards * log_action_probabilities
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
    def __init__(self):
        super().__init__()
        self.card_conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, stride=11)
        self.player_adut = nn.Linear(8, 8)
        self.fc = nn.Linear(136, 32)
        return

    def forward(self, x):
        assert not x.requires_grad
        assert x.ndim == 3
        N_p, N_t, N_d = x.size()   # players, turns, dims
        assert N_d == 360
        since = time.time()
        x = x.view(N_p*N_t, 1, N_d)
        #card_features, player_adut_features = torch.split(x, 352, dim=-1)
        card_features = self.card_conv(x)
        assert card_features.shape == (N_p*N_t, 4, 32)
        player_adut_features = self.player_adut(x[..., 352:])
        assert player_adut_features.shape == (N_p*N_t, 1, 8)
        x = torch.cat((card_features.view(N_p*N_t, -1), player_adut_features.view(N_p*N_t, -1)), dim=-1)
        assert x.shape == (N_p*N_t, 128+8)
        x = torch.tanh(x)
        logits = self.fc(x)
        assert logits.shape == (N_p*N_t, 32)
        #print(f'\tModel time: {time.time() - since}')
        #del cat, player_adut_features, card_features
        return logits.view(N_p, N_t, -1)


class AdutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, stride=8)
        self.fc = nn.Linear(16, 5)
        return

    def forward(self, x):
        assert x.ndim == 3
        N_p, N_c, N_d = x.shape
        assert N_c == 1
        assert N_d == 32
        features = self.conv(x).view(N_p, -1)
        features = torch.tanh(features)
        return self.fc(features)


class Brain(abc.ABC):
    def __init__(self, checkpoint=None):
        self.input_states_per_player = defaultdict(list)
        self.masks_per_player = defaultdict(list)
        self.samples_per_player = defaultdict(list)
        self.rewards_per_player = defaultdict(list)
        self.adut_cards_per_player = defaultdict(list)
        self.adut_points_per_player = dict()
        self.muss_per_player = dict()
        self.adut_sampled_per_player = dict()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        log.info(f"Device: {self.device}")
        self.model = Model().to(self.device)
        self.model.eval()
        self.loss = PolicyGradientLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)
        self.optimizer.zero_grad()

        self.adut_model = AdutModel()
        self.adut_model.eval().to(self.device)
        self.adut_loss = PolicyGradientLoss().to(self.device)
        self.adut_optimizer = optim.Adam(self.adut_model.parameters(), lr=0.01)
        self.adut_optimizer.zero_grad()

        if checkpoint is not None:
            with checkpoint as file: #this will close it
                save_dict = torch.load(file)
                self.model.load_state_dict(save_dict['model']),
                self.adut_model.load_state_dict(save_dict['adut_model'])
                self.optimizer.load_state_dict(save_dict['optimizer'])
                self.adut_optimizer.load_state_dict(save_dict['adut_optimizer'])
            assert checkpoint.closed
            log.info(f"Loaded checkpoint {checkpoint.name}")
        return

    def save(self):
        import tempfile
        with tempfile.NamedTemporaryFile(prefix='rl-model-', suffix='.pth', dir='.', delete=False) as file:
            save_dict = {
                'model': self.model.state_dict(),
                'adut_model': self.adut_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'adut_optimizer': self.adut_optimizer.state_dict()
            }
            torch.save(save_dict, file)
        log.info(f"Saved model @ {file.name}")
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
        since = time.time()
        with torch.no_grad():
            allowed_indices = torch.tensor(
                [card.to_int() for card in allowed_cards],
                requires_grad=False,
                device=self.device
            )
            assert not allowed_indices.requires_grad
            mask = indices_to_mask(allowed_indices, device=self.device)
            x = torch.tensor(state, dtype=torch.float32, requires_grad=False, device=self.device).view(1, 1, -1)
            self.input_states_per_player[player].append(state)
            self.masks_per_player[player].append(mask.tolist())     # due to memory bugs, I store lists
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            assert not probs.requires_grad
            probs2 = probs * mask
            try:
                c = Categorical(probs=probs2)  # lowkey hacky xd
            except ValueError as ex:
                print(allowed_indices)
                print(probs)
                print(probs2)
                print(logits)
                raise ex
            card_idx = c.sample().item()
            del probs
            self.samples_per_player[player].append(card_idx)
        #print(f'\tMove time: {time.time() - since}')
        return card_idx

    def train_model(self):
        self.model.train()
        since = time.time()

        masks = torch.tensor(
            list(self.masks_per_player.values()),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device
        )
        assert masks.shape == (len(self.masks_per_player), 8, 32)

        inputs = torch.tensor(
            list(self.input_states_per_player.values()),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device)
        assert inputs.shape == (len(self.input_states_per_player), 8, 360)

        samples = torch.tensor(
            list(self.samples_per_player.values()),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device
        )
        assert samples.shape == (len(self.samples_per_player), 8)

        rewards = torch.tensor(
            list(self.rewards_per_player.values()),
            requires_grad=False,
            dtype=torch.float32,
            device=self.device
        )
        assert rewards.shape == samples.shape
        discounted_rewards = cumsum_reverse(rewards, dim=-1)

        logits = self.model(inputs)
        assert logits.shape == masks.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs*masks
        try:
            c = Categorical(probs=probs)
        except ValueError as ex:
            print(probs)
            raise ex
        logprobs = c.log_prob(samples)
        assert logprobs.shape == rewards.shape
        ##print(rewards)
        loss = self.loss(logprobs, discounted_rewards)
        log.info(f"PLAY LOSS: \t{loss}")
        loss.backward()
        #
        self.optimizer.step()
        self.optimizer.zero_grad()
        keys = list(self.rewards_per_player.keys())
        for player in keys:
            assert len(self.rewards_per_player[player]) == 8
            assert all(isinstance(t, int) for t in self.rewards_per_player[player])
            del self.rewards_per_player[player]
            assert len(self.input_states_per_player[player]) == 8
            assert all(isinstance(t, list) for t in self.input_states_per_player[player])
            del self.input_states_per_player[player]
            assert len(self.masks_per_player[player]) == 8
            assert all(isinstance(t, list) for t in self.masks_per_player[player])
            del self.masks_per_player[player]
            assert len(self.samples_per_player[player]) == 8
            assert all(isinstance(t, int) for t in self.samples_per_player[player])
            del self.samples_per_player[player]
            #del self.logprobs_per_player[player]# .clear()
        #del rewards, logprobs, loss
        torch.cuda.empty_cache()
        #print(f'\tTrain time: {time.time() - since}')
        self.model.eval()
        return

    def train_adut_model(self):
        assert list(self.adut_cards_per_player.keys()) == list(self.muss_per_player.keys())
        assert list(self.adut_cards_per_player.keys()) == list(self.adut_sampled_per_player.keys())
        assert list(self.adut_cards_per_player.keys()) == list(self.adut_points_per_player.keys())
        if len(self.adut_cards_per_player) > 0:
            # if ai players had a chance to decide adut at all
            self.adut_model.train()
            inputs = torch.tensor(
                list(self.adut_cards_per_player.values()),
                requires_grad=False,
                device=self.device,
                dtype=torch.float32
            )
            sampled = torch.tensor(
                list(self.adut_sampled_per_player.values()),
                requires_grad=False,
                device=self.device,
                dtype=torch.float32
            )
            points = torch.tensor(
                list(self.adut_points_per_player.values()),
                requires_grad=False,
                device=self.device,
                dtype=torch.float32
            )
            #print(inputs, inputs.shape)
            logits = self.adut_model(inputs)
            probs = F.softmax(logits, dim=-1)
            for muss, i in zip(self.muss_per_player.values(), range(len(probs))):
                if muss:
                    probs[i, -1] = 0.0
            c = Categorical(probs=probs)
            logprobs = c.log_prob(sampled)
            assert logprobs.shape == points.shape, f"{logprobs} {points}"
            loss = self.adut_loss(logprobs, points)
            log.info(f'ADUT LOSS: {loss}')
            loss.backward()
            self.adut_optimizer.step()
            self.adut_optimizer.zero_grad()

            self.adut_cards_per_player.clear()  # = defaultdict(list)
            self.adut_points_per_player.clear()
            self.muss_per_player.clear()
            self.adut_sampled_per_player.clear()
            self.adut_model.eval()
        return

    def train(self):
        self.train_model()
        self.train_adut_model()
        return

    def get_adut(self, player, card_idx_list, is_muss):
        with torch.no_grad():
            cards = torch.tensor(card_idx_list, device=self.device)
            model_input = indices_to_mask(cards)
            self.adut_cards_per_player[player].append(model_input.tolist())
            logits = self.adut_model(model_input.view(1, 1, -1))
            probs = F.softmax(logits, dim=-1)
            if is_muss:
                probs[..., -1] = 0.0
            c = Categorical(probs=probs)
            suit_idx = c.sample().item()
        self.muss_per_player[player] = is_muss
        self.adut_sampled_per_player[player] = suit_idx
        return suit_idx

    def adut_feedback(self, player, points):
        if player in self.adut_sampled_per_player:
            self.adut_points_per_player[player] = points
        return

class BigBrain(Brain):
    def __init__(self, checkpoint=None):
        return super().__init__(checkpoint)

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

    def notify_rewards(self, final_points):
        if self.brain is not None:
            self.brain.set_rewards(self, self.points)
            self.brain.adut_feedback(self, final_points)
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
        if not is_muss:
            options = [1, 2, 3, 4, 5]
            weights = [0.125, 0.125, 0.125, 0.125, 0.5]
        else:
            options = [1, 2, 3, 4]
            weights = None
        return Adut(random.choices(options, weights=weights, k=1)[0])

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
        suit_idx = self._brain.get_adut(self, list(map(Card.to_int, self.cards)), is_muss)
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
