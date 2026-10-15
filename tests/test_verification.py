import numpy as np
import pytest
from peercheck.verification import pooled_foreground_iou, agreement, intersection_labels, route
from peercheck.metrics import evaluate


def test_semantic_pooled_iou():
    a=np.array([[1,1,2,0]],dtype=np.uint8)
    b=np.array([[1,2,2,0]],dtype=np.uint8)
    assert pooled_foreground_iou(a,b,0)==pytest.approx(2/4)
    assert pooled_foreground_iou(a,a)==1


def test_symmetry():
    rng=np.random.default_rng(42)
    a=rng.choice([0,1,2,3,4,255],(75,100)).astype(np.uint8)
    b=rng.choice([0,1,2,3,4,255],(75,100)).astype(np.uint8)
    assert agreement(a,b)==agreement(b,a)


def test_intersection_and_ignore():
    a=np.array([[1,2,0,3]],dtype=np.uint8)
    b=np.array([[1,0,0,4]],dtype=np.uint8)
    v=intersection_labels(a,b)
    np.testing.assert_array_equal(v,[[1,255,0,255]])
    assert pooled_foreground_iou(a,v,0)==pytest.approx(1/3)
    assert evaluate(a,v)['labeled_precision']==1
    assert evaluate(a,v)['ignore_fraction']==.5


def test_area_tolerance_and_strict_boundary():
    empty=np.zeros((100,100),dtype=np.uint8)
    small=empty.copy();small.flat[:4]=1
    assert agreement(empty,small)==1
    assert agreement(empty,small,0)==0
    small.flat[4]=1
    assert agreement(empty,small)==0
    assert agreement(small,empty)==0
    assert np.count_nonzero(small)==5


def test_total_foreground_not_per_class():
    a=np.zeros((100,100),dtype=np.uint8)
    b=a.copy();a.flat[:3]=1;a.flat[3:6]=2
    assert agreement(a,b)==0


def test_labeled_precision_is_foreground_only():
    gt=np.array([[1,1,0,2]],dtype=np.uint8)
    pred=np.array([[1,255,2,0]],dtype=np.uint8)
    assert evaluate(gt,pred)['labeled_precision']==.5


def test_empty_and_routing():
    a=np.zeros((4,4),dtype=np.uint8)
    assert agreement(a,a)==1
    assert route(.85)=='accept'
    assert route(.849)=='review'
    assert route(1,is_keyframe=True)=='keyframe'
    with pytest.raises(ValueError):route(float('nan'))
    with pytest.raises(ValueError):agreement(a,np.ones((3,3)))
    with pytest.raises(ValueError):agreement(a,np.full_like(a,7))
