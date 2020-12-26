from belabot.engine.util import calculate_points


def test_calculate_points_no_decl():
    assert calculate_points(82, True, 0, 0) == (82, 80)
    assert calculate_points(81, True, 0, 0) == (0, 162)
    assert calculate_points(80, True, 0, 0) == (0, 162)

    assert calculate_points(82, False, 0, 0) == (162, 0)
    assert calculate_points(81, False, 0, 0) == (162, 0)
    assert calculate_points(80, False, 0, 0) == (80, 82)
    return


def test_calculate_points_decl():
    assert calculate_points(81, True, 50, 0) == (131, 81)
    assert calculate_points(81, True, 20, 0) == (101, 81)
    assert calculate_points(90, True, 0, 20) == (0, 182)

    assert calculate_points(82, False, 0, 20) == (82, 100)
    assert calculate_points(82, False, 0, 50) == (82, 130)
    assert calculate_points(80, False, 20, 0) == (182, 0)
    return


def test_calculate_points_stiglja():
    assert calculate_points(162, True, 0, 0) == (252, 0)
    assert calculate_points(162, False, 0, 0) == (252, 0)
    assert calculate_points(162, True, 20, 0) == (272, 0)
    assert calculate_points(162, False, 0, 50) == (302, 0)
    assert calculate_points(162, True, 20, 50) == (322, 0)
    assert calculate_points(162, False, 20, 50) == (322, 0)

    assert calculate_points(0, True, 0, 0) == (0, 252)
    assert calculate_points(0, False, 0, 0) == (0, 252)
    assert calculate_points(0, True, 20, 0) == (0, 272)
    assert calculate_points(0, False, 0, 50) == (0, 302)
    assert calculate_points(0, True, 20, 50) == (0, 322)
    assert calculate_points(0, False, 20, 50) == (0, 322)
