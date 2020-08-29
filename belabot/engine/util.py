from .card import Card, Rank, Suit
from bisect import insort
from typing import List, Dict


def sort_cards_by_suit(cards: List[Card]) -> Dict[Suit, List[Card]]:
    sorted_cards = dict()
    for card in cards:
        if card.suit not in sorted_cards:
            sorted_cards[card.suit] = [card]
        else:
            insort(sorted_cards[card.suit], card)
    return sorted_cards


def sort_cards_by_rank(cards: List[Card]) -> Dict[Rank, List[Card]]:
    sorted_cards = dict()
    for card in cards:
        if card.rank not in sorted_cards:
            sorted_cards[card.rank] = [card]
        else:
            insort(sorted_cards[card.rank], card)
    return sorted_cards


def get_declarations(cards: List[Card]) -> None:
    return
