import unittest
import numpy as np
from arena_sphere_user_solver import terms,solve_user,solve_users


class UserSphereSolverTests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(4);self.initial=r.normal(size=4);self.initial/=np.linalg.norm(self.initial)
        self.design=r.normal(size=(6,10,4));self.offset=r.normal(size=(6,10));self.target=r.integers(0,10,6)

    def test_gradient_on_sphere_matches_finite_difference(self):
        u=self.initial;_,g,_=terms(u,u,self.design,self.offset,self.target,3.)
        d=np.random.default_rng(5).normal(size=4);d-=u*np.dot(u,d);d/=np.linalg.norm(d)
        values=[]
        for rate in [-1e-5,1e-5]:
            q=u+rate*d;q/=np.linalg.norm(q)
            values.append(terms(q,u,self.design,self.offset,self.target,3.,False)[0])
        self.assertAlmostEqual((values[1]-values[0])/2e-5,np.dot(g,d),places=7)

    def test_objective_decrease_stationarity_and_norm(self):
        value=terms(self.initial,self.initial,self.design,self.offset,self.target,3.)[0]
        q,loss,g,n=solve_user(self.initial,self.design,self.offset,self.target,3.,100)
        self.assertLess(loss,value);self.assertLess(g,1e-5);self.assertGreater(n,0)
        self.assertAlmostEqual(np.linalg.norm(q),1.,places=12)

    def test_spherical_hessian_contains_retraction_curvature(self):
        u=self.initial;f,g,h=terms(u,u,self.design,self.offset,self.target,3.)
        d=np.random.default_rng(9).normal(size=4);d-=u*np.dot(d,u);d/=np.linalg.norm(d)
        eps=1e-4;values=[]
        for rate in [-eps,eps]:
            q=u+rate*d;q/=np.linalg.norm(q)
            values.append(terms(q,u,self.design,self.offset,self.target,3.,False)[0])
        actual=(sum(values)-2*f)/eps**2
        expected=d@h@d-np.dot(u,g)
        self.assertAlmostEqual(actual,expected,places=5)

    def test_rotation_and_candidate_permutation(self):
        a=solve_user(self.initial,self.design,self.offset,self.target,3.,100)[0]
        r=np.linalg.qr(np.random.default_rng(6).normal(size=(4,4)))[0]
        b=solve_user(self.initial@r,self.design@r,self.offset,self.target,3.,100)[0]
        np.testing.assert_allclose(b,a@r,atol=1e-7)
        p=np.random.default_rng(7).permutation(10);inv=np.argsort(p)
        c=solve_user(self.initial,self.design[:,p],self.offset[:,p],inv[self.target],3.,100)[0]
        np.testing.assert_allclose(c,a,atol=1e-7)

    def test_no_evidence_and_cold_are_unchanged(self):
        q,_,g,n=solve_user(self.initial,np.empty((0,10,4)),np.empty((0,10)),np.empty(0,dtype='i'),3.)
        np.testing.assert_array_equal(q,self.initial);self.assertEqual((g,n),(0.,0))
        q,_,_,_=solve_user(np.zeros(4),self.design,self.offset,self.target,3.)
        np.testing.assert_array_equal(q,0)

    def test_parallel_wrapper_does_not_mix_users(self):
        initial=np.stack([self.initial,self.initial]);vectors=np.eye(4)
        ptr=np.array([0,1,2]);items=np.array([[0,1,2],[1,2,3]])
        offset=np.zeros((2,3));target=np.array([0,2]);precision=np.array([3.,3.])
        u,loss,g,n=solve_users(ptr,items,offset,target,vectors,initial,2.,precision,100)
        expected=np.stack([solve_user(initial[k],2*vectors[items[k:k+1]],offset[k:k+1],target[k:k+1],3.,100)[0] for k in range(2)])
        np.testing.assert_allclose(u,expected,atol=1e-12)


if __name__=='__main__':unittest.main()
