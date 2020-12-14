from .card import Suit, Adut
from .player import Player
from .declarations import get_player_declarations

import logging
import os
import random
import sys
from typing import List, Tuple, Optional

random_gen = random.SystemRandom()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(os.environ.get("BB_LOGLEVEL", "INFO").upper())


class Belot:
    def __init__(self, players: List[Player]):
        self.players = players
        self.deck = range(32)
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
        first_6, talons = self.shuffle()
        self.deal_cards(first_6)
        adut = self.get_adut(dealer_index)
        log.debug("Adut is " + repr(adut))
        self.deal_cards(talons)
        mi_declarations, vi_declarations = self.compute_declarations(
            [player.cards for player in self.players], dealer_index
        )
        log.debug("MI_DECL: {}".format(mi_declarations))
        log.debug("VI_DECL: {}".format(vi_declarations))
        total_points = 162 + sum(t.value() for t in mi_declarations + vi_declarations)
        log.debug("Total points: {}".format(total_points))
        x = random_gen.randint(0, 162)
        for player in self.players:
            player.clear_cards()
        return x + sum(t.value() for t in mi_declarations), total_points - x

    def deal_cards(self, cards_list: List[List[int]]) -> None:
        for cards, player in zip(cards_list, self.players):
            player.add_cards(cards)
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

    def get_adut(self, dealer_index: int) -> Optional[Suit]:
        for i in range(1, 1 + len(self.players)):  # the dealer calls last
            player_index = (dealer_index + i) % len(self.players)
            log.debug(player_index)
            player = self.players[player_index]
            adut = player.get_adut(is_muss=(i == 4))
            if adut != Adut.NEXT:
                return Suit(adut.value)
        return None

    def compute_declarations(
        self, player_cards: List[List], dealer_index: int
    ) -> Tuple[List, List]:
        declarations_per_player = []
        for i in range(1, 1 + len(self.players)):  # the dealer calls last
            player_index = (dealer_index + i) % len(self.players)
            player_declarations = get_player_declarations(player_cards[player_index])
            declarations_per_player.append((player_declarations, -(i - 1)))
        # Here's a trick: we are searching for the best declaration (in terms of points
        # and general ordering of declarations). But if it so happens that the two players
        # have the exact same declarations (e.g. both players have a sequence of three cards,
        # both with highest rank of Ace, but one player has it in, say, Hearts, and the other
        # in, say, Clubs (the Suit obviously doesn't matter)), then we break the ties with
        # respect to the order of declaring - the advantage is given to the player that
        # declares sooner.
        # PS nested for loop list comprehensions never truly made sense to me
        all_declarations = [
            (declaration, player_index)
            for player_declarations, player_index in declarations_per_player
            for declaration in player_declarations
        ]
        if len(all_declarations) == 0:
            return [], []
        best_index = -max(all_declarations)[1]
        teammate_index = (best_index + 2) % len(declarations_per_player)
        final_declarations = (
            declarations_per_player[best_index][0]
            + declarations_per_player[teammate_index][0]
        )
        # if the dealer_index is an even number, the dealer belongs to the team MI
        # else the dealer belongs to the team VI
        # in both cases, the dealer's team has a disadvantage which manifests itself
        # as lack of precedence (explained earlier)
        if dealer_index % 2 == 0:
            return [], final_declarations
        else:
            return final_declarations, []
