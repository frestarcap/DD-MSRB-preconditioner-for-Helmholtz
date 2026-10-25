"""Exact frozen-spectrum rank allocation across the current JCP experiments.

All decisions use integer energies converted from binary64 input spectra.
Lagrangian bounds prune only states that cannot improve the greedy solution.
"""
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from itertools import product
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import random
import time
from audit_ecc_greedy_dp import integer_energies, tails_of, greedy, cost_dp, enumeration_three

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/greedy_dp_paper_20260921'


def bounded_dp(a,n,tau,order=None):
    tails=tails_of(a)
    rg,last=greedy(a,n,tau)
    upper=sum(nb*r for nb,r in zip(n,rg))
    if last is None:
        return 0,rg,dict(frontiers=[1]*len(n),candidate_counts=[1]*len(n))
    last_b,den=last;num=n[last_b]
    # lambda=num/den. P_b(r)=den*c_b(r)+num*T_b(r).
    # Every feasible allocation of cost <= upper has sum(P_b-min(P_b)) <= W.
    p=[[den*nb*r+num*t for r,t in enumerate(tb)] for nb,tb in zip(n,tails)]
    mins=[min(pb) for pb in p]
    slack=den*upper+num*tau-sum(mins)
    assert slack>=0
    candidates=[[(r,tb[r],pb[r]-mn) for r in range(len(tb))
                 if tb[r]<=tau and pb[r]-mn<=slack]
                for tb,pb,mn in zip(tails,p,mins)]
    assert all(candidates)
    if order is None:
        order=sorted(range(len(n)),key=lambda b:(len(candidates[b]),-n[b]))
    min_cost=[min(n[b]*r for r,_,_ in cb) for b,cb in enumerate(candidates)]
    min_tail=[min(t for _,t,_ in cb) for cb in candidates]
    remaining_cost=sum(min_cost);remaining_tail=sum(min_tail)
    partial_min=0
    g=reduce(math.gcd,n)
    states={0:(0,())};counts=[]
    for b in order:
        remaining_cost-=min_cost[b];remaining_tail-=min_tail[b];partial_min+=mins[b]
        new={}
        for c,(value,ranks) in states.items():
            for r,tail,_ in candidates[b]:
                nc=c+n[b]//g*r
                if nc*g+remaining_cost>upper:
                    continue
                nt=value+tail
                if nt+remaining_tail>tau:
                    continue
                if den*nc*g+num*nt-partial_min>slack:
                    continue
                if nc not in new or nt<new[nc][0]:
                    new[nc]=(nt,ranks+(r,))
        states={};best=None
        for c in sorted(new):
            if best is None or new[c][0]<best:
                states[c]=new[c];best=new[c][0]
        assert states
        counts.append(len(states))
    c=min(states);ranks=[0]*len(n)
    for b,r in zip(order,states[c][1]):ranks[b]=r
    assert sum(tails[b][r] for b,r in enumerate(ranks))<=tau
    return c*g,ranks,dict(frontiers=counts,candidate_counts=list(map(len,candidates)))


def self_check():
    rng=random.Random(20260921)
    cases=[([[5],[3]],[3,2],5),([[10**50,1],[2]],[7,2],0),([[0],[0]],[1,3],0)]
    for _ in range(200):
        b=rng.randint(1,7)
        a=[sorted([rng.randrange(1,50) for _ in range(rng.randint(1,4))],reverse=True) for _ in range(b)]
        cases.append((a,[rng.randint(1,30) for _ in range(b)],rng.randrange(sum(map(sum,a))+1)))
    for a,n,tau in cases:
        tails=tails_of(a)
        exhaustive=min(sum(x*y for x,y in zip(n,r))
            for r in product(*(range(len(v)+1) for v in a))
            if sum(tails[b][k] for b,k in enumerate(r))<=tau)
        s,_,_=bounded_dp(a,n,tau)
        assert s==exhaustive
        assert branch_verify(a,n,tau)[0]==exhaustive
    print('EXACT_DP_SELF_CHECK_PASS cases=203',flush=True)


def branch_verify(a,n,tau):
    """Independent exact branch enumeration; no cost-state table/dominance.

    Uses the same valid LP dual bound to exclude impossible improvements,
    then visits rank combinations with nonnegative-cost and tail bounds.
    """
    tails=tails_of(a);rg,last=greedy(a,n,tau)
    best=sum(x*y for x,y in zip(n,rg))
    if last is None:return best,1
    b,den=last;num=n[b]
    scores=[[den*nb*r+num*t for r,t in enumerate(tb)] for nb,tb in zip(n,tails)]
    mins=[min(s) for s in scores]
    slack=den*best+num*tau-sum(mins)
    choices=[[(nb*r,t,scores[b][r]-mins[b]) for r,t in enumerate(tb)
              if t<=tau and scores[b][r]-mins[b]<=slack]
             for b,(nb,tb) in enumerate(zip(n,tails))]
    choices.sort(key=lambda opts:(len(opts),-max(c for c,t,d in opts)))
    count=len(choices);sc=[0]*(count+1);st=[0]*(count+1);sd=[0]*(count+1)
    for j in range(count-1,-1,-1):
        sc[j]=sc[j+1]+min(c for c,t,d in choices[j])
        st[j]=st[j+1]+min(t for c,t,d in choices[j])
        sd[j]=sd[j+1]+min(d for c,t,d in choices[j])
        choices[j].sort(key=lambda x:x[2])
    nodes=0
    def visit(j,c,t,d):
        nonlocal nodes,best
        nodes+=1
        if c+sc[j]>=best or t+st[j]>tau or d+sd[j]>slack:return
        if j==count:
            best=c;return
        for dc,dt,dd in choices[j]:visit(j+1,c+dc,t+dt,d+dd)
    visit(0,0,0,0)
    return best,nodes


def analyze(record):
    a,tau=integer_energies(record);n=list(map(int,record['n']));tails=tails_of(a)
    total=sum(map(sum,a));rg,last=greedy(a,n,tau)
    assert rg==record['saved_ranks'],(record['tag'],record['layer'],'saved greedy mismatch',rg,record['saved_ranks'])
    sg=sum(x*y for x,y in zip(n,rg));dg=sum(t[r] for t,r in zip(tails,rg))
    t0=time.perf_counter();sd,rd,stats=bounded_dp(a,n,tau);seconds=time.perf_counter()-t0
    # Verify against the original unbounded cost DP for up to 16 blocks;
    # 37-block problems use reversed block order with the same exact bounds.
    if len(n)<=16:
        ref,_,_=cost_dp(tails,n,tau,sg);check='unbounded exact-cost DP'
    else:
        ref,_,_=bounded_dp(a,n,tau,list(reversed(range(len(n)))));check='reverse-order exact-cost DP'
    assert ref==sd
    verification_nodes=None
    if len(n)>16:
        ref,verification_nodes=branch_verify(a,n,tau)
        assert ref==sd
        check+=' and independent exact branch enumeration'
    if len(n)==3:
        assert enumeration_three(tails,n,tau)==sd;check+=' and rank enumeration'
    dd=sum(t[r] for t,r in zip(tails,rd));assert dd<=tau
    if last:
        b,energy=last;bound=Fraction(n[b]*(tau-dg),energy);slp=sg-bound
        assert 0<=sg-sd<=bound<n[b]
    else:bound=Fraction(0);slp=Fraction(0)
    g=reduce(math.gcd,n);lb=g*math.ceil(slp/g);assert lb<=sd<=sg
    return dict(tag=record['tag'],group=record['group'],layer=record['layer'],blocks=len(n),
        eta=record['eta'],n=n,greedy_ranks=rg,dp_ranks=rd,greedy_rank=sum(rg),dp_rank=sum(rd),
        greedy_entries=sg,dp_entries=sd,saved_entries=sg-sd,
        savings_percent=100*(sg-sd)/sg if sg else 0,
        greedy_tail=math.sqrt(Fraction(dg,total)) if total else 0,
        dp_tail=math.sqrt(Fraction(dd,total)) if total else 0,
        dp_budget_fraction=float(Fraction(dd,tau)) if tau else 0,
        lp_lower_bound=float(slp),gcd_lower_bound=lb,gap_bound_entries=float(bound),
        dp_seconds=seconds,verification=check,verification_nodes=verification_nodes,dp_stats=stats,
        spectrum_origin=record['spectrum_origin'],source=record['source'])


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--suffix',default='')
    args=p.parse_args();self_check();records=[];hashes={}
    for name in args.inputs:
        src=Path(name);records+=json.loads(src.read_text())['records'];hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    results=[]
    dest=OUT/f'results{args.suffix}.json'
    for r in records:
        result=analyze(r);results.append(result)
        print(f"{r['tag']} L{r['layer']}: {result['greedy_entries']} -> {result['dp_entries']} "
              f"({result['savings_percent']:.4f}%) DP={result['dp_seconds']:.4f}s states={max(result['dp_stats']['frontiers'])}",flush=True)
        dest.write_text(json.dumps(dict(input_sha256=hashes,results=results),indent=2)+'\n')
    groups=defaultdict(list)
    for r in results:groups[r['tag']].append(r)
    summary=[]
    for tag,rs in groups.items():
        sg=sum(r['greedy_entries'] for r in rs);sd=sum(r['dp_entries'] for r in rs)
        summary.append(dict(tag=tag,group=rs[0]['group'],layers=len(rs),blocks=rs[0]['blocks'],
            greedy_entries=sg,dp_entries=sd,greedy_basis_MiB=sg/65536,dp_basis_MiB=sd/65536,
            saved_entries=sg-sd,savings_percent=100*(sg-sd)/sg,
            max_layer_savings_percent=max(r['savings_percent'] for r in rs),
            greedy_layer_ranks=[r['greedy_rank'] for r in rs],dp_layer_ranks=[r['dp_rank'] for r in rs]))
    write_csv(OUT/f'layers{args.suffix}.csv',results);write_csv(OUT/f'models{args.suffix}.csv',summary)
    print(f'PAPER_EXACT_DP_PASS models={len(summary)} layers={len(results)}',flush=True)


if __name__=='__main__':main()
