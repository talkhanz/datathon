"""A small, transparent genetic algorithm (numpy only). Used twice in `modeling.py`:

1. wrapper **feature selection** (binary chromosome = feature mask), and
2. **inverse design** of the "optimal provider profile" (mixed binary / integer / continuous genes) that maximises a
   trained model's score while staying close to real, observed facilities.

Operators: tournament selection, uniform crossover, per-gene mutation (bit flip / Gaussian step), elitism, an
optional `repair` hook that projects children back into the feasible set. Fully seeded, so runs are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Gene:
    name: str
    kind: str            # "binary" | "int" | "float"
    low: float = 0.0
    high: float = 1.0


@dataclass
class GAResult:
    best_x: np.ndarray
    best_fitness: float
    history: list = field(default_factory=list)   # [{"generation", "best", "mean", "evaluations"}]
    population: Optional[np.ndarray] = None
    fitness: Optional[np.ndarray] = None


def _sample(genes: list[Gene], n: int, rng) -> np.ndarray:
    X = np.empty((n, len(genes)))
    for j, g in enumerate(genes):
        if g.kind == "binary":
            X[:, j] = rng.integers(0, 2, n)
        elif g.kind == "int":
            X[:, j] = rng.integers(int(g.low), int(g.high) + 1, n)
        else:
            X[:, j] = rng.uniform(g.low, g.high, n)
    return X


def _clip(X: np.ndarray, genes: list[Gene]) -> np.ndarray:
    for j, g in enumerate(genes):
        X[:, j] = np.clip(X[:, j], g.low, g.high)
        if g.kind in ("binary", "int"):
            X[:, j] = np.round(X[:, j])
    return X


def run_ga(fitness: Callable[[np.ndarray], np.ndarray], genes: list[Gene], pop_size: int = 40, n_gen: int = 40,
           seed: int = 20261121, elite: int = 2, tournament: int = 3, crossover_rate: float = 0.9,
           mutation_rate: Optional[float] = None, sigma: float = 0.15,
           repair: Optional[Callable[[np.ndarray], np.ndarray]] = None,
           init: Optional[np.ndarray] = None, patience: Optional[int] = None) -> GAResult:
    """Maximise `fitness` (a vectorised function: population matrix -> 1-D fitness array)."""
    rng = np.random.default_rng(seed)
    d = len(genes)
    mutation_rate = mutation_rate if mutation_rate is not None else max(1.0 / d, 0.02)
    span = np.array([g.high - g.low for g in genes], dtype=float)
    is_bin = np.array([g.kind == "binary" for g in genes])

    P = _sample(genes, pop_size, rng)
    if init is not None:
        P[: len(init)] = init[:pop_size]
    P = _clip(P, genes)
    if repair:
        P = repair(P)
    F = np.asarray(fitness(P), dtype=float)
    evals = len(P)
    history, best_so_far, stale = [], -np.inf, 0

    for gen in range(n_gen):
        history.append({"generation": gen, "best": float(F.max()), "mean": float(F.mean()), "evaluations": evals})
        if F.max() > best_so_far + 1e-9:
            best_so_far, stale = F.max(), 0
        else:
            stale += 1
            if patience and stale >= patience:
                break
        order = np.argsort(-F)
        children = [P[i].copy() for i in order[:elite]]
        while len(children) < pop_size:
            a, b = (P[max(rng.choice(pop_size, tournament, replace=False), key=lambda i: F[i])] for _ in range(2))
            child = np.where(rng.random(d) < 0.5, a, b) if rng.random() < crossover_rate else a.copy()
            m = rng.random(d) < mutation_rate
            child[m & is_bin] = 1 - child[m & is_bin]
            cont = m & ~is_bin
            child[cont] += rng.normal(0, sigma, cont.sum()) * span[cont]
            children.append(child)
        P = _clip(np.array(children), genes)
        if repair:
            P = repair(P)
        F = np.asarray(fitness(P), dtype=float)
        evals += len(P)
    history.append({"generation": len(history), "best": float(F.max()), "mean": float(F.mean()), "evaluations": evals})
    i = int(np.argmax(F))
    return GAResult(P[i].copy(), float(F[i]), history, P, F)
