import logging
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple, Set

import more_itertools as mit

from .card import Adut, Card, Suit
from .declarations import Declaration, get_player_declarations
from .player import Player, Brain
from .util import calculate_points, get_valid_moves, get_winner

random_gen = random.SystemRandom()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class Belot:
    def __init__(self, players: Sequence[Player]):
        assert len(set(players)) == len(
            players
        ), "The players must all have different names"
        self.players = players
        self.deck = range(32)
        self.cards_played: List[Card] = []
        self.brains: Set[Brain] = set()
        for player, right, teammate, left in mit.circular_shifts(self.players):
            player.team_setup(teammate, left, right)
            self.brains.add(player.brain)
        self.brains -= {None}
        return

    def play(self) -> None:
        # mi is implicitly the name of the team made out of players at indices 0 and 2
        # vi team is made out of players 1 and 3
        current_dealer_index = 0
        mi, vi = 0, 0
        while mi <= 1000 and vi <= 1000 or mi == vi:
            round_mi, round_vi = self.round(current_dealer_index)
            current_dealer_index = (current_dealer_index + 1) % 4
            mi += round_mi
            vi += round_vi
        log.info(f"MI {mi} \t {vi} VI")
        return

    def round(self, dealer_index: int) -> Tuple[int, int]:
        # BIDDING PHASE
        first_6, talons = self.shuffle()
        self.deal_cards(first_6)
        adut, adut_caller_index = self.get_adut(dealer_index)
        mi_bid = (adut_caller_index % 2) == 0
        log.info(
            f'{self.players[adut_caller_index].name} ({"MI" if mi_bid else "VI"}) have bid {repr(adut)} for adut'
        )
        self.deal_cards(talons)
        all_declarations = self.compute_declarations(
            [player.cards for player in self.players], dealer_index
        )
        mi_declarations = all_declarations[0] + all_declarations[2]
        vi_declarations = all_declarations[1] + all_declarations[3]
        log.debug("MI_DECL: {}".format(mi_declarations))
        log.debug("VI_DECL: {}".format(vi_declarations))
        total_points = 162 + sum(t.value() for t in mi_declarations + vi_declarations)
        log.debug("Total points: {}".format(total_points))

        self.notify_pregame(
            {self.players[key]: value for key, value in all_declarations.items()},
            adut,
            adut_caller_index,
        )

        # MAIN PHASE
        # mi_points = random_gen.randint(0, 162)
        mi_points, vi_points = 0, 0
        # next player starts first
        start_player_index = (dealer_index + 1) % len(self.players)
        turn_cards: List[Card] = []
        for turn in range(8):
            for i in range(4):
                player_index = (start_player_index + i) % 4
                player = self.players[player_index]
                card = player.play_card(turn_cards, adut)  # reinforcement learning step
                assert card in get_valid_moves(turn_cards, player.cards, adut)
                log.debug(f"\t{player.name} plays {repr(card)}")
                turn_cards.append(card)
                self.notify_played(player, card)
            turn_winner = get_winner(turn_cards, adut)
            log.debug(f"{turn_winner} wins the turn.")
            if (start_player_index + turn_winner) % 2 == 0:
                mi_turn = sum(card.points(adut) for card in turn_cards) + (
                    10 if turn == 7 else 0
                )
                vi_turn = 0
            else:
                mi_turn = 0
                vi_turn = sum(card.points(adut) for card in turn_cards) + (
                    10 if turn == 7 else 0
                )
            log.debug(f"mi_turn: {mi_turn}\tvi_turn: {vi_turn}")
            self.notify_turn_points(mi_turn, vi_turn)
            mi_points += mi_turn
            vi_points += vi_turn
            start_player_index = (start_player_index + turn_winner) % len(self.players)
            turn_cards.clear()
        assert (mi_points + vi_points) == 162
        log.debug(f"MI won {repr(mi_points)}, VI won {repr(162 - mi_points)} in game.")
        mi_points, vi_points = calculate_points(
            mi_points,
            mi_bid,
            sum(t.value() for t in mi_declarations),
            sum(t.value() for t in vi_declarations),
        )
        log.debug(f"MI won {repr(mi_points)}, VI won {repr(vi_points)} in total.")
        assert len(self.brains) > 0
        for player in self.players:
            assert len(player.cards) == 0
        self.notify_rewards(mi_points, vi_points)
        for brain in self.brains:
            brain.train()   # haha sounds funny
        return mi_points, vi_points

    def notify_rewards(self, mi_points, vi_points):
        for i, player in enumerate(self.players):
            player.notify_rewards(mi_points if i % 2 == 0 else vi_points)
        return

    def deal_cards(self, cards_list: List[List[int]]) -> None:
        for cards, player in zip(cards_list, self.players):
            player.add_cards(cards)
        return

    def notify_played(self, player: Player, card: Card) -> None:
        for other_player in self.players:
            if other_player == player:
                player.card_accepted(card)
            else:
                other_player.notify_played(player, card)
        return

    def notify_turn_points(self, mi_points: int, vi_points: int) -> None:
        for i, player in enumerate(self.players):
            player.notify_turn_points(mi_points if i % 2 == 0 else vi_points)
        return

    def notify_pregame(
        self,
        declarations: Dict[Player, List[Declaration]],
        adut: Suit,
        adut_caller: int,
    ) -> None:
        for player in self.players:
            player.notify_pregame(declarations, adut, self.players[adut_caller])
        return

    def shuffle(self) -> Tuple[List[List[int]], List[List[int]]]:
        # Usually, cards in Bela are dealed in a particular order
        # this kinda makes sense in a real world where not all
        # permutations are equally likely
        # however, I'm using OS provided source of randomness
        # which is guaranteed to have enough entropy
        # therefore, with simple math, one concludes that there
        # is no advantage of simulating the deal rule in bela
        # as opposed to just allocating cards contiguously
        deck = random_gen.sample(self.deck, len(self.deck))
        cards_per_player = len(self.deck) // len(self.players)
        adut_cards = []
        talons = []
        for i in range(0, len(deck), cards_per_player):
            player_cards_end = i + cards_per_player
            talon_start = player_cards_end - 2
            adut_cards.append(deck[i:talon_start])
            talons.append(deck[talon_start:player_cards_end])
        return adut_cards, talons

    def get_adut(self, dealer_index: int) -> Tuple[Suit, int]:
        for i in range(1, 1 + len(self.players)):  # the dealer calls last
            player_index = (dealer_index + i) % len(self.players)
            log.debug(f"Player {player_index} bids...")
            player = self.players[player_index]
            adut = player.get_adut(is_muss=(i == 4))
            if adut != Adut.NEXT:
                to_return = (Suit(adut.value), player_index)
                break
        return to_return

    def compute_declarations(
        self, player_cards: List[List[Card]], dealer_index: int
    ) -> Dict[int, List[Declaration]]:
        declarations_per_player: Dict[int, List[Declaration]] = {
            0: [],
            1: [],
            2: [],
            3: [],
        }  # absolute indices
        for i in range(1, 1 + len(self.players)):
            player_index = (dealer_index + i) % len(self.players)
            player = self.players[player_index]
            cards = player_cards[player_index]
            log.debug(f"\t{player.name} {cards}")
            player_declarations = get_player_declarations(cards)
            declarations_per_player[player_index].extend(player_declarations)

        # Here's the trick: we are searching for the best declaration (in terms of points
        # and general ordering of declarations). But if it so happens that the two players
        # have the exact same declarations (e.g. both players have a sequence of three cards,
        # both with highest rank of Ace, but one player has it in, say, Hearts, and the other
        # in, say, Clubs (the Suit obviously doesn't matter)), then we break the ties with
        # respect to the order of declaring - the advantage is given to the player that
        # declares sooner.
        # PS nested for loop list comprehensions never truly made sense to me
        all_declarations = [
            (declaration, -i)
            for i in range(1, 1 + len(self.players))
            for declaration in declarations_per_player[
                (dealer_index + i) % len(self.players)
            ]
        ]
        log.debug(all_declarations)
        if len(all_declarations) == 0:
            return declarations_per_player
        best_relative_index = -max(all_declarations)[1]
        best_index = (best_relative_index + dealer_index) % len(self.players)
        teammate_index = (best_index + 2) % len(self.players)
        final_declarations = {
            best_index: declarations_per_player[best_index],
            teammate_index: declarations_per_player[teammate_index],
        }
        for i in set(range(len(self.players))) - set([best_index, teammate_index]):
            final_declarations[i] = []
        return final_declarations
