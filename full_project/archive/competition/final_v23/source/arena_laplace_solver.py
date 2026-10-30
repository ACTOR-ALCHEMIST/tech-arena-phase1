"""Damped Newton MAP and local curvature for linear slate-choice models.

Each user's candidate score is offset + design @ delta; outside has zero
design. A positive diagonal prior makes the objective strictly convex.
This numerical module has no data-loading or temporal-split side effects.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def terms(delta, design, offsets, target, precision, derivatives=True):
    events, candidates, dimension = design.shape
    value = .5*np.sum(precision*delta*delta)
    gradient = precision*delta
    hessian = np.diag(precision)
    scores = np.empty(candidates+1, np.float64)
    mean = np.empty(dimension, np.float64)
    probability = np.empty(candidates, np.float64)
    for event in range(events):
        scores[candidates] = offsets[event, candidates]
        for slot in range(candidates):
            scores[slot] = offsets[event, slot]
            for k in range(dimension):
                scores[slot] += design[event, slot, k]*delta[k]
        top = np.max(scores)
        normalizer = np.sum(np.exp(scores-top))
        value += np.log(normalizer)+top-scores[target[event]]
        if not derivatives:
            continue
        mean[:] = 0.
        for slot in range(candidates):
            probability[slot] = np.exp(scores[slot]-top)/normalizer
            for k in range(dimension):
                mean[k] += probability[slot]*design[event, slot, k]
        for k in range(dimension):
            gradient[k] += mean[k]
            if target[event] < candidates:
                gradient[k] -= design[event, target[event], k]
            for l in range(dimension):
                hessian[k, l] -= mean[k]*mean[l]
                for slot in range(candidates):
                    hessian[k, l] += probability[slot]*design[event, slot, k]*design[event, slot, l]
    return value, gradient, hessian


@njit(cache=True)
def solve_user(design, offsets, target, precision, max_steps=12):
    dimension = design.shape[2]
    delta = np.zeros(dimension, np.float64)
    steps = 0
    for iteration in range(max_steps):
        value, gradient, hessian = terms(delta, design, offsets, target, precision)
        direction = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(direction)) < 1e-8:
            break
        decrease = np.dot(gradient, direction)
        rate, accepted = 1., False
        for attempt in range(16):
            candidate = delta-rate*direction
            next_value, _, _ = terms(candidate, design, offsets, target, precision, False)
            if np.isfinite(next_value) and next_value <= value-1e-4*rate*decrease:
                delta = candidate
                accepted = True
                break
            rate *= .5
        if not accepted:
            break
        steps += 1
    value, gradient, hessian = terms(delta, design, offsets, target, precision)
    covariance = np.linalg.inv(hessian)
    return delta, covariance, value, np.max(np.abs(gradient)), steps


@njit(parallel=True, cache=True)
def solve_users(indptr, candidate_items, offsets, target, vectors, scale, precision, max_steps=12):
    users, dimension = len(indptr)-1, vectors.shape[1]+1
    delta = np.zeros((users, dimension), np.float32)
    covariance = np.zeros((users, dimension, dimension), np.float32)
    loss = np.zeros(users, np.float64)
    gradient = np.zeros(users, np.float64)
    steps = np.zeros(users, np.int16)
    for user in prange(users):
        start, stop = indptr[user], indptr[user+1]
        design = np.ones((stop-start, candidate_items.shape[1], dimension), np.float64)
        for event in range(stop-start):
            for slot in range(candidate_items.shape[1]):
                iid = candidate_items[start+event, slot]
                for k in range(dimension-1):
                    design[event, slot, k] = scale*vectors[iid, k]
        d, c, value, g, n = solve_user(design, offsets[start:stop], target[start:stop], precision[user], max_steps)
        delta[user], covariance[user] = d, c
        loss[user], gradient[user], steps[user] = value, g, n
    return delta, covariance, loss, gradient, steps
