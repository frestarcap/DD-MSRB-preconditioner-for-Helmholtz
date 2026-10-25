"""Independent dense algebra check; not a Helmholtz performance experiment."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(915)
n=96; ns=4; blocks=np.array_split(np.arange(n),6)
A0=np.diag(2.5+.2j+.1*rng.random(n))+np.diag(-np.ones(n-1),1)+np.diag(-.8*np.ones(n-1),-1)
D=np.diag(rng.uniform(.2,2,n));mus=np.linspace(.2,1.4,ns)
As=[A0+mu*D for mu in mus]
b=rng.normal(size=n)+1j*rng.normal(size=n);b=b/np.linalg.norm(b)
S=np.column_stack([np.linalg.solve(A,b) for A in As])

def basis(s,rank):
    vs=[]
    for idx in blocks:
        u,_,_=np.linalg.svd(s[idx],full_matrices=False)
        v=np.zeros((n,min(rank,u.shape[1])),complex);v[idx]=u[:,:v.shape[1]];vs.append(v)
    return np.column_stack(vs)

V0=basis(S,ns)
rows=[]
for identity_term in [False,True]:
    Q=[b[:,None].copy() for _ in mus]
    for j in range(3):
        # Independent direct solutions, NOT the inverse-action recurrence.
        E=np.column_stack([np.linalg.solve(A,Q[k][:,j]) for k,A in enumerate(As)])
        leak=np.linalg.norm(E-V0@(V0.conj().T@E))/np.linalg.norm(E)
        rows.append(dict(identity_term=identity_term,layer=j+1,relative_leakage=float(leak)))
        V=basis(E,2)
        if j<2:
            for k,A in enumerate(As):
                q=Q[k][:,j]
                z=V@np.linalg.solve(V.conj().T@A@V,V.conj().T@q)
                if identity_term:z=z+q
                v=A@z
                for _ in range(2):v=v-Q[k]@(Q[k].conj().T@v)
                Q[k]=np.column_stack([Q[k],v/np.linalg.norm(v)])
assert max(x['relative_leakage'] for x in rows if not x['identity_term'])<1e-10
assert max(x['relative_leakage'] for x in rows if x['identity_term'])>1e-3
A=A0+.73*D
C=V0@np.linalg.solve(V0.conj().T@A@V0,V0.conj().T)
idem=float(np.linalg.norm(C@A@C-C)/np.linalg.norm(C))
# An independent RHS avoids a nearly vanishing Arnoldi vector: the training
# RHS has a 1.5e-5 remainder, which amplifies cancellation in this relative test.
t=rng.normal(size=n)+1j*rng.normal(size=n);t=t/np.linalg.norm(t)
z1=C@t;w=A@z1;q2=w-t*(t.conj()@w);q2=q2/np.linalg.norm(q2);z2=C@q2
collinear=float(np.linalg.norm(z2-z1*(z1.conj()@z2)/(z1.conj()@z1))/np.linalg.norm(z2))
assert idem<1e-12 and collinear<1e-10
out=Path(__file__).resolve().parents[1]/'outputs/dd_parameter_mechanism_20260915/span_identity_toy.json'
out.write_text(json.dumps(dict(passed=True,seed=915,n=n,samples=ns,blocks=len(blocks),rows=rows,
                             CAC_minus_C_relative=idem,fixed_second_action_outside_first=collinear),indent=2))
print(out.read_text())
