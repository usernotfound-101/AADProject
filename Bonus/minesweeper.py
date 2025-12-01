#!/usr/bin/env python3
import argparse
import itertools
import logging
import sys
import fileinput
import typing
import os
import numpy as np
from pysat.card import CardEnc
from pysat.solvers import Solver as SATEngine

CELL_ID = {'flag': 9, 'mine': 10, 'unknown': 11}


class UnsolvableBoard(Exception):
    pass


class EmptyInput(Exception):
    pass

def neighbors_box(arr, center, radius=1):
    return arr[max(0, center[0] - radius):center[0] + radius + 1,
               max(0, center[1] - radius):center[1] + radius + 1]


def parse_input(path):
    mines_remaining = None
    first_step = None

    with fileinput.input(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("#mines "):
                mines_remaining = int(line[len("#mines "):].rstrip())
            elif line.startswith("#first_bloc "):
                raw = line[len("#first_bloc "):].rstrip()
                a, _, b = raw.partition(',')
                first_step = (int(a), int(b))
        else:
            raise EmptyInput

        f = itertools.chain([line], f)
        board = np.loadtxt(f, delimiter=",", dtype=int)

    if not board.size:
        raise EmptyInput

    return board, mines_remaining, first_step


class CardinalityEncoder:
    def __init__(self, total_vars):
        self.maxid = total_vars

    def __call__(self, vars_, k):
        cnf = CardEnc.equals(vars_, k, top_id=self.maxid)
        self.maxid = max(self.maxid, cnf.nv)
        return cnf.clauses


def encode_board(board, mines_remaining=None):
    var_ids = np.arange(board.size).reshape(board.shape)
    clauses = set()

    candidates = []
    for x, y in zip(*np.nonzero(board == CELL_ID["unknown"])):
        if mines_remaining is None:
            around = neighbors_box(board, (x, y))
            if np.any((around >= 1) & (around <= 8)):
                candidates.append(int(var_ids[x, y]))
        else:
            candidates.append(int(var_ids[x, y]))

    # if no usable SAT variables, return empty constraints now
    if not candidates:
        return [], []

    candidates.sort()
    map_to_vid = {v: i for i, v in enumerate(candidates, start=1)}
    encoder = CardinalityEncoder(len(candidates))

    for x, y in zip(*np.nonzero((board >= 1) & (board <= 8))):
        nb = neighbors_box(board, (x, y))
        nb_var = neighbors_box(var_ids, (x, y))
        if np.any(nb == CELL_ID["unknown"]):
            vars_ = sorted(nb_var[nb == CELL_ID["unknown"]].tolist())
            mapped = [map_to_vid[v] for v in vars_ if v in map_to_vid]
            if mapped:  # skip empty lists
                k = board[x, y] - np.sum((nb == CELL_ID["mine"]) | (nb == CELL_ID["flag"]))
                clauses.update(map(tuple, encoder(mapped, k)))

    if mines_remaining is not None:
        mapped_all = [map_to_vid[v] for v in candidates]
        if mapped_all:
            clauses.update(map(tuple, encoder(mapped_all, mines_remaining)))

    return candidates, list(map(list, clauses))


def enumerate_solutions(clauses, solver_name="minisat22", limit=10000):
    with SATEngine(name=solver_name, bootstrap_with=clauses) as s:
        models = list(itertools.islice(s.enum_models(), limit + 1))
    if models in ([], [[]]):
        raise UnsolvableBoard
    return np.array(models, dtype=np.int64)


def sat_infer(board, mines_remaining=None):
    qvars, clauses = encode_board(board, mines_remaining)
    sols = enumerate_solutions(clauses)
    sols = np.sign(sols[:, :len(qvars)])
    confidence = np.abs(np.sum(sols, 0)) / sols.shape[0]
    is_mine = np.sign(np.sum(sols, 0)) > 0

    coords = np.stack(np.unravel_index(np.array(qvars), board.shape), axis=1)
    output = np.concatenate((coords, is_mine[:, None]), axis=1)
    return output, confidence


def sat_solve(board, mines_remaining, first_step=None):
    if np.all(board == CELL_ID["unknown"]):
        if first_step:
            return np.array([[first_step[0], first_step[1], 0]])
        unknown = np.stack(np.nonzero(board == CELL_ID["unknown"]), axis=1)
        chosen = unknown[np.random.randint(len(unknown))]
        return np.array([[chosen[0], chosen[1], 0]])

    try:
        result, conf = sat_infer(board, mines_remaining)
        if np.max(conf) > 1 - 1e-6:
            return result[conf > 1 - 1e-6]
        return result[np.argmax(conf)][None]

    except UnsolvableBoard:
        unknown = np.stack(np.nonzero(board == CELL_ID["unknown"]), axis=1)
        r = unknown[np.random.randint(len(unknown))]
        return np.array([[r[0], r[1], 0]])


def main():
    input_dir = "boards"
    output_dir = "results"

    # Create output directory if missing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each CSV file in data/
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".csv"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            board, mines_remaining, first_step = parse_input(input_path)
            move = sat_solve(board, mines_remaining, first_step)

            # save results
            np.savetxt(output_path, move, delimiter=",", fmt="%d")

            print(f"Processed {filename} -> {output_path}")

        except EmptyInput:
            print(f"Skipped empty file: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
