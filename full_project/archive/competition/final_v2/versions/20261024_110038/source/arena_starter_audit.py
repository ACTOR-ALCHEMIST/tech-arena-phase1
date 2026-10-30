"""FP64 audit of the supplied starter formula on the same temporal cohort."""
import math
import time
import numpy as np
from numba import njit,prange
from scipy import sparse
from arena_core import snapshot,evaluate
from arena_report import LAB,note,write_json
from ctr_data import load_cache,temporal_split
from run import timestamp


@njit(parallel=True,cache=True)
def cosine_sum64(indptr,columns,counts,pop,items,history):
    out=np.zeros(items.shape,dtype=np.float64)
    for e in prange(len(items)):
        for j in range(10):
            item=items[e,j];a,b=indptr[item],indptr[item+1]
            for h in history[e]:
                if h==0:continue
                lo,hi=a,b
                while lo<hi:
                    mid=(lo+hi)//2
                    if columns[mid]<h:lo=mid+1
                    else:hi=mid
                if lo<b and columns[lo]==h and pop[item]>0 and pop[h]>0:
                    out[e,j]+=float(counts[lo])/(math.sqrt(float(pop[item]))*math.sqrt(float(pop[h])))
    return out


def main():
    started=time.monotonic();path=snapshot('2025-01-26');stats=np.load(path/'stats.npz')
    _,data=load_cache('data/cache_v1','train');_,indices=temporal_split(data,timestamp('2025-01-26'))
    counts=sparse.load_npz(path/'co_counts.npz');ctr=float(stats['ctr'])
    def logit(p):
        p=min(max(p,1e-6),1-1e-6);return math.log(p/(1-p))
    def rates(clicks,impressions):
        return np.array([logit((float(c)+50*ctr)/(float(n)+50)) if n else logit(ctr) for c,n in zip(clicks,impressions)])
    user=rates(stats['uc'],stats['ui']);item=rates(stats['ic'],stats['ii'])
    items=np.array(data['items'][indices]);history=np.array(data['history'][indices])
    cf=cosine_sum64(counts.indptr,counts.indices,counts.data,stats['pop'],items,history)
    z=item[items]+user[data['users'][indices],None]-logit(ctr)+25*cf
    p=np.array([1/(1+math.exp(-float(value))) for value in z.ravel()]).reshape(z.shape)
    folder=LAB/'experiments/cf_26_fp64_audit';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'validation.npz',event_indices=indices,labels=data['labels'][indices],predictions=p,starter=p)
    result={'name':folder.name,'method':'sparse_item_cf_fp64','cutoff':'2025-01-26',**evaluate(data['labels'][indices],p),
            'config':{'smoothing':50,'cf_weight':25,'precision':'FP64, Python math.exp sigmoid, supplied starter formula'},
            'seconds':time.monotonic()-started,'saturated_probability_rows':int((p==1).sum())}
    write_json(folder/'result.json',result)
    note(f"公平基线复核：原单精度优化实现AUC0.615649；按官方starter双精度公式重新计算AUC{result['auc']:.6f}。概率接近1的舍入并列造成微小差异，最终提升与置信区间使用双精度参考。",'audit')
    print(result)


if __name__=='__main__':main()
