"""A separate item graph using shrunk positive pointwise mutual information."""
import fcntl
import json
import os
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits
from arena_core import snapshot, clear_diagonal
from arena_ood import write_json

RECIPE = dict(version=3, dimensions=32, shift=5., pair_shrink=3., degree_shrink=20., seed=8761,
              counts='recomputed_symmetric_with_explicit_diagonal_mask')


def ppmi_graph(counts, popularity, users, shift=5., shrink=3., degree_shrink=20.):
    """Keep sparsity; a zero co-occurrence never creates an edge."""
    graph = counts.astype('float64').copy()
    columns = graph.indices
    for start in range(0, graph.shape[0], 2048):
        end = min(start+2048, graph.shape[0])
        a, b = graph.indptr[start], graph.indptr[end]
        row = np.repeat(np.arange(start, end), np.diff(graph.indptr[start:end+1]))
        observed = graph.data[a:b].copy()
        expected = popularity[row]*popularity[columns[a:b]]/max(users, 1)
        graph.data[a:b] = np.maximum(np.log(observed/np.maximum(expected*shift, 1e-12)), 0)*observed/(observed+shrink)
    graph = clear_diagonal(graph)
    strength = np.asarray(graph.sum(1)).ravel()
    # Rare, isolated cliques otherwise contribute eigenvalue-one directions
    # and displace the useful large-population components.
    scale = 1/np.sqrt((strength+degree_shrink).clip(1e-12))
    graph = (sparse.diags(scale)@graph@sparse.diags(scale)).tocsr()
    return graph


def model(cutoff):
    folder = snapshot(cutoff)/'ppmi_v3_d32_shift5_shrink3_degree20'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe'] == RECIPE
            return dict(np.load(folder/'factors.npz'))
        tick = time.monotonic()
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        # Legacy cached counts were affected by scipy.setdiag. Do not modify
        # those artifacts: rebuild this view directly from the binary matrix.
        counts = clear_diagonal(x.astype('float64').T@x)
        assert (counts-counts.T).nnz == 0
        degree = np.diff(x.indptr)
        pop = np.asarray(x.sum(0), dtype='float64').ravel()
        graph = ppmi_graph(counts, pop, int((degree > 0).sum()))
        asymmetry = graph-graph.T
        assert not asymmetry.nnz or np.max(np.abs(asymmetry.data)) < 1e-12
        del counts
        with threadpool_limits(limits=1):
            value, vectors = eigsh(graph, k=33, which='LA', tol=1e-5, maxiter=800,
                                   v0=np.random.default_rng(RECIPE['seed']).normal(size=graph.shape[0]))
        order = np.argsort(value)[::-1]
        value, vectors = value[order], vectors[:, order]
        v = (vectors[:, 1:33]*np.sqrt(value[1:33].clip(0))).astype('float32')
        v[pop == 0] = 0
        v[0] = 0
        u = (x@v/degree.clip(1)[:, None]).astype('float32')
        u[0] = 0
        np.savez(folder/'factors.tmp.npz', users=u, items=v, bias=np.log1p(pop).astype('float32'))
        os.replace(folder/'factors.tmp.npz', folder/'factors.npz')
        write_json(folder/'result.json', dict(recipe=RECIPE, cutoff=cutoff, eigenvalues=value.tolist(),
                   edges=graph.nnz, seconds=time.monotonic()-tick,
                   source='Strictly past-only binary user/item click matrix; PMI debiases item popularity, count shrinkage reduces singleton influence.'))
        print(json.dumps(dict(stage='ppmi', cutoff=cutoff, seconds=time.monotonic()-tick)), flush=True)
        return dict(users=u, items=v, bias=np.log1p(pop).astype('float32'))


def extra_features(data, indices, cutoff, config=None):
    from arena_coclick import describe
    from arena_core import co_features
    rep = model(cutoff)
    pieces, names = [], []
    for size in [16, 32]:
        x, cols = describe(dict(rep, users=rep['users'][:, :size], items=rep['items'][:, :size]), data, indices)
        pieces.append(x)
        names += [c.replace('coclick_', 'ppmi'+str(size)+'_', 1) for c in cols]
    # Supply the corrected exact graph alongside the new low-rank graph. Old
    # canonical numeric caches are kept intact, so archived heads stay stable.
    past = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
    counts = clear_diagonal(past.T@past)
    pop = np.asarray(past.sum(0)).ravel().astype('float32')
    exact = co_features(counts.indptr, counts.indices, counts.data, pop,
                        np.asarray(data['items'][indices]), np.asarray(data['history'][indices]),
                        np.asarray(data['age_hours'][indices]))
    pieces.append(exact.reshape(-1, 9))
    names += ['corrected_cf_'+n for n in ['cos_sum', 'cos_max', 'recent_12h', 'recent_3d', 'recent_30d',
               'shrink3', 'log_count', 'conditional', 'last']]
    return np.concatenate(pieces, 1), names
