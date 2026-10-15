import numpy as np
from peercheck.keyframes import select, quota_order
from peercheck.effort import effort


def test_nested_and_shared_starts():
    frames=[{'sequence':s,'frame_index':i} for s,n in [('a',11),('b',33),('c',22)] for i in range(n)]
    e=np.random.default_rng(4).normal(size=(66,16))
    a=select(frames,20,'uniform',0)
    b=select(frames,20,'coverage-aware',0,e)
    assert a[:3]==b[:3]
    assert select(frames,10,'uniform',0)==a[:7]
    assert len(a)==len(set(a))==14
    assert [sum(frames[i]['sequence']==s for i in a) for s in ['a','b','c']]==[sum(frames[i]['sequence']==s for i in b) for s in ['a','b','c']]


def test_effort():
    assert effort(100,5,10)['seconds']==5*157+10*7
    e=effort(100,5,10,4)
    assert e['seconds']==(5+4)*157+10*7
    assert abs(e['relative_to_full_annotation']-(9+10*7/157)/100)<1e-12
