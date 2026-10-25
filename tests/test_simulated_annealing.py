import random

import pytest

from simulated_annealing import (
    count_contacts,
    coordinates_to_moves,
    propose_move,
    search,
    validate_sequence,
)


def test_contact_count_excludes_bonded_pairs():
    sequence = "HHHH"
    square_fold = ((0, 0), (1, 0), (1, 1), (0, 1))

    assert count_contacts(sequence, square_fold) == 1


def test_solver_returns_valid_self_avoiding_fold():
    sequence = "HPHPPHHPHH"

    contacts, coords = search(sequence, iterations=2_000, seed=21)

    assert len(coords) == len(sequence)
    assert coords[0] == (0, 0)
    assert len(set(coords)) == len(coords)
    assert all(
        abs(start[0] - end[0]) + abs(start[1] - end[1]) == 1
        for start, end in zip(coords, coords[1:])
    )
    assert contacts == count_contacts(sequence, coords)


def test_solver_finds_expected_contacts_for_six_hydrophobic_residues():
    contacts, _ = search("HHHHHH", iterations=5_000, seed=7)

    assert contacts == 2


def test_search_is_reproducible_for_a_fixed_seed():
    first = search("HHHHHH", iterations=1_000, seed=5)
    second = search("HHHHHH", iterations=1_000, seed=5)

    assert first == second


def test_move_proposals_preserve_self_avoiding_bonded_walk():
    coords = tuple((index, 0) for index in range(10))
    rng = random.Random(15)
    for _ in range(500):
        candidate = propose_move(coords, rng)
        assert len(candidate) == len(coords)
        assert len(set(candidate)) == len(candidate)
        assert all(
            abs(start[0] - end[0]) + abs(start[1] - end[1]) == 1
            for start, end in zip(candidate, candidate[1:])
        )


@pytest.mark.parametrize("sequence", ["", "HHX", "123"])
def test_sequence_validation_rejects_invalid_input(sequence):
    with pytest.raises(ValueError, match="H/P"):
        validate_sequence(sequence)


def test_move_conversion_uses_gym_lattice_action_labels():
    coords = ((0, 0), (-1, 0), (-1, 1), (0, 1), (0, 0))

    assert coordinates_to_moves(coords) == "LDRU"


def test_move_conversion_rejects_non_unit_steps():
    with pytest.raises(ValueError, match="unit-step"):
        coordinates_to_moves(((0, 0), (2, 0)))
