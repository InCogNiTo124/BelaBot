from typing import Tuple

DECK_TOTAL = 162
STIGLJA_PENALTY = 90


def calculate_points(
    mi_points_raw: int, mi_bid: bool, mi_declarations: int, vi_declarations: int
) -> Tuple[int, int]:
    if mi_points_raw == 0:
        return 0, DECK_TOTAL + STIGLJA_PENALTY + mi_declarations + vi_declarations
    elif mi_points_raw == DECK_TOTAL:
        return DECK_TOTAL + STIGLJA_PENALTY + mi_declarations + vi_declarations, 0

    mi_points = mi_points_raw + mi_declarations
    vi_points = (DECK_TOTAL - mi_points_raw) + vi_declarations
    if mi_bid:
        if mi_points > vi_points:
            return mi_points, vi_points
        else:
            return 0, mi_points + vi_points
    else:
        if vi_points > mi_points:
            return mi_points, vi_points
        else:
            return mi_points + vi_points, 0
