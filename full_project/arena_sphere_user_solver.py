"""Per-user conditional MAP on a unit sphere, with fixed item vectors.

Damped projected Gauss-Newton steps use the exact spherical objective and an
Armijo line search. This isolates the extra information in actual displayed
negatives while retaining the pretrained item geometry.
"""
import numpy as np
from numba import njit,prange


@njit(cache=True)
def terms(u,initial,design,offsets,target,precision,derivatives=True):
    d=len(u);value=.5*precision*np.sum((u-initial)**2)
    gradient=precision*(u-initial);hessian=precision*np.eye(d)
    for event in range(len(target)):
        score=offsets[event]+design[event]@u
        top=np.max(score);ex=np.exp(score-top);p=ex/ex.sum()
        value+=np.log(ex.sum())+top-score[target[event]]
        if not derivatives:continue
        mean=np.zeros(d)
        for slot in range(len(score)):mean+=p[slot]*design[event,slot]
        gradient+=mean-design[event,target[event]]
        for k in range(d):
            for l in range(d):
                hessian[k,l]-=mean[k]*mean[l]
                for slot in range(len(score)):
                    hessian[k,l]+=p[slot]*design[event,slot,k]*design[event,slot,l]
    return value,gradient,hessian


@njit(cache=True)
def solve_user(initial,design,offsets,target,precision,max_steps=32):
    u=initial.copy();dimension=len(u);steps=0
    if np.linalg.norm(u)<1e-8 or len(target)==0:
        loss,_,_=terms(u,initial,design,offsets,target,precision)
        return u,loss,0.,steps
    for iteration in range(max_steps):
        value,g,h=terms(u,initial,design,offsets,target,precision)
        projection=np.eye(dimension)-np.outer(u,u)
        tangent=projection@g
        if np.max(np.abs(tangent))<1e-7:break
        # The normalization/retraction contributes radial curvature. Without
        # this term a Gauss-Newton approximation can oscillate on users whose
        # conditional likelihood strongly favors increasing the vector norm.
        radial=np.dot(u,g)
        system=projection@(h-radial*np.eye(dimension))@projection+np.outer(u,u)
        minimum=np.linalg.eigvalsh(system)[0]
        if minimum<1e-5:system+=(1e-5-minimum)*projection
        direction=np.linalg.solve(system,tangent)
        norm=np.linalg.norm(direction)
        if norm>.5:direction*=.5/norm
        decrease=np.dot(tangent,direction)
        rate=1.;accepted=False
        for attempt in range(20):
            candidate=u-rate*direction;candidate/=np.linalg.norm(candidate)
            next_value,_,_=terms(candidate,initial,design,offsets,target,precision,False)
            if np.isfinite(next_value) and next_value<=value-1e-4*rate*decrease:
                u=candidate;accepted=True;steps+=1;break
            rate*=.5
        if not accepted:break
    value,g,_=terms(u,initial,design,offsets,target,precision)
    tangent=g-u*np.dot(u,g)
    return u,value,np.max(np.abs(tangent)),steps


@njit(parallel=True,cache=True)
def solve_users(ptr,items,offsets,target,vectors,initial,temperature,precision,max_steps=32):
    n_users,dimension=initial.shape
    result=initial.copy();loss=np.zeros(n_users);gradient=np.zeros(n_users);steps=np.zeros(n_users,np.int16)
    for user in prange(n_users):
        a,b=ptr[user],ptr[user+1]
        design=np.empty((b-a,items.shape[1],dimension))
        for event in range(b-a):
            for slot in range(items.shape[1]):
                design[event,slot]=temperature*vectors[items[a+event,slot]]
        u,value,g,n=solve_user(initial[user],design,offsets[a:b],target[a:b],precision[user],max_steps)
        result[user]=u;loss[user]=value;gradient[user]=g;steps[user]=n
    return result,loss,gradient,steps
