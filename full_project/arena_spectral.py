"""Sparse normalized co-click graph embeddings; no dense item-square matrix."""
import fcntl
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits
from arena_core import snapshot
from arena_ood import write_json


def model(cutoff, width=16):
    folder = snapshot(cutoff)/('spectral_v1_d'+str(width)+'_shrink5')
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        start = time.monotonic()
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz').astype(np.float64)
        degree = np.asarray(x.sum(1)).ravel()
        pop = np.asarray(x.sum(0)).ravel()
        # Degree normalization prevents prolific users from dominating pair counts.
        normalized = sparse.diags(1/np.sqrt(degree.clip(1)))@x@sparse.diags(1/np.sqrt(pop+5))
        graph = (normalized.T@normalized).tocsr()
        with threadpool_limits(limits=1):
            value, vectors = eigsh(graph, k=width+1, which='LA', tol=1e-5, maxiter=500,
                                   v0=np.random.default_rng(901).normal(size=graph.shape[0]))
        order = np.argsort(value)[::-1]
        value, vectors = value[order], vectors[:, order]
        # Remove the leading global popularity/activity direction.
        v = (vectors[:, 1:width+1]*np.sqrt(value[1:width+1].clip(0))).astype(np.float32)
        v[pop == 0] = 0
        v[0] = 0
        u = (x@v/degree.clip(1)[:, None]).astype(np.float32)
        u[0] = 0
        np.savez(folder/'factors.npz', users=u, items=v, bias=np.log1p(pop).astype('f'))
        write_json(folder/'result.json', dict(cutoff=cutoff, dimensions=width, item_degree_shrink=5,
                   eigenvalues=value.tolist(), seconds=time.monotonic()-start,
                   source='Past-only unique user-item clicks, degree-normalized sparse graph'))
        print({'stage': 'spectral_embedding', 'cutoff': cutoff, 'seconds': time.monotonic()-start}, flush=True)
    return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config=None, width=16):
    from arena_coclick import describe
    rep = model(cutoff, width)
    pieces, names = [], []
    for size in ([8, 16] if width == 16 else [32, 64]):
        part = dict(rep, users=rep['users'][:, :size], items=rep['items'][:, :size])
        x, cols = describe(part, data, indices)
        pieces.append(x)
        names += [c.replace('coclick_', 'spectral'+str(size)+'_', 1) for c in cols]
    return np.concatenate(pieces, axis=1), names
