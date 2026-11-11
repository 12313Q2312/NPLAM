import numpy as np
import pytest
import torch
from nplam.model import NPLAM
from nplam.data import synthetic, components, split_scale, TrainMinMax
from nplam.metrics import recovery
from nplam.trainer import seed_everything, fit_neural, predict


@pytest.mark.parametrize("mode,slope", [("paper",0),("legacy",.2)])
def test_forward_against_independent_scalar_equation(mode,slope):
    m=NPLAM(2,2,mode).double()
    with torch.no_grad():
        m.v.copy_(torch.tensor([[1.,-2.],[3.,4.]]))
        m.b.copy_(torch.tensor([[-.3,.5],[.2,-.7]]))
        m.g1.copy_(torch.tensor([-.4,.8]))
        m.g2.copy_(torch.tensor([1.3,-.2]))
        m.w.copy_(torch.tensor([2.,-1.]))
        m.a.copy_(torch.tensor([[.5,-.8],[1.2,.3]]))
        m.beta.fill_(.4)
    x=torch.tensor([[.7,-.2],[1.,.3]],dtype=torch.float64)
    expected=[]
    with torch.no_grad():
        for row in x.tolist():
            total = float(m.beta)
            for j in range(2):
                total += row[j] * float(m.g1[j]) * float(m.w[j])
                for t in range(2):
                    z = row[j] * float(m.g1[j]) * float(m.g2[j]) * float(m.v[j, t]) + float(m.b[j, t])
                    total += float(m.a[j, t]) * max(z, slope * z)
            expected.append(total)
    np.testing.assert_allclose(m(x).detach(),expected,rtol=1e-12)
    np.testing.assert_allclose(m.effective_linear_weights().detach(),m.g1.detach()*m.w.detach())


def test_zero_gate_removes_dependence_but_not_constant():
    m=NPLAM(2,4)
    with torch.no_grad():
        m.g1[0]=0
        m.g2[1]=0
        m.a.fill_(1)
        m.b.fill_(1)
    x=torch.tensor([[1.,2.],[100.,4.]])
    c=m.contributions(x)
    assert c[0,0]==c[1,0]==4
    assert c[1,1]-c[0,1]==2*m.g1[1]*m.w[1]


def test_l1_semantics_and_zero_subgradient():
    m=NPLAM(3,5)
    with torch.no_grad():
        m.g1.fill_(2); m.g2.fill_(-3); m.a.fill_(4)
    assert m.regularization([.1,.2,.3]).item()==pytest.approx(.1*6+.2*9+.3*60)
    m.mode="legacy"
    assert m.regularization([.1,.2,.3]).item()==pytest.approx(.1*2+.2*3+.3*4)
    z=torch.tensor(0.,requires_grad=True)
    z.abs().backward()
    assert z.grad==0


def test_frozen_random_features_and_gate_gradient():
    seed_everything(1)
    m=NPLAM(2,4).double()
    with torch.no_grad(): m.a.fill_(.2)
    assert "v" in dict(m.named_buffers()) and "v" not in dict(m.named_parameters())
    x=torch.tensor([[.37,-.26]],dtype=torch.float64)
    m(x).sum().backward()
    analytic=m.g2.grad[0].item()
    with torch.no_grad():
        eps=1e-6
        m.g2[0]+=eps; high=m(x).item()
        m.g2[0]-=2*eps; low=m(x).item()
        m.g2[0]+=eps
    assert analytic==pytest.approx((high-low)/(2*eps),abs=1e-7)


@pytest.mark.parametrize("task",["mixed30","sparse300","linear30","nonlinear30","dense30"])
def test_synthetic_truth_and_determinism(task):
    x,y,t=synthetic(task,100,9,noise=0)
    x2,y2,t2=synthetic(task,100,9,noise=0)
    np.testing.assert_array_equal(x,x2)
    np.testing.assert_array_equal(y,y2)
    np.testing.assert_allclose(y,components(x,task).sum(1))
    assert (t>0).sum()==(30 if task=="dense30" else 5)
    assert np.all(x>=-2.5) and np.all(x<=2.5)


def test_train_only_preprocessing_and_disjoint_partition():
    x=np.arange(100.).reshape(50,2); y=np.arange(50.)
    a=split_scale(x,y,3)
    changed=x.copy(); changed[a.indices[2]]+=1e8
    b=split_scale(changed,y,3)
    np.testing.assert_array_equal(a.x_scaler.center,b.x_scaler.center)
    np.testing.assert_array_equal(a.x[0],b.x[0])
    assert len(set(np.concatenate(a.indices)))==50
    assert set(a.indices[0]).isdisjoint(a.indices[1])
    assert a.x_scaler.transform(changed[a.indices[2]]).max()>1
    const=TrainMinMax().fit(np.ones((10,2)))
    assert np.isfinite(const.transform(np.ones((2,2)))).all()


def test_recovery_denominators_include_missing_features():
    r=recovery([2,1,2,0],[1,2,0,2])
    assert (r["tp"],r["fp"],r["tn"],r["fn"])==(2,1,0,1)
    assert r["f1"]==pytest.approx(2/3)
    assert r["cf"]==0
    assert r["uf"]+r["of"]+r["missed_structure"]==1


def test_training_checkpoint_and_deterministic_replay(tmp_path):
    x,y,_=synthetic("linear30",200,0)
    data=split_scale(x,y,0)
    results=[]
    for _ in range(2):
        seed_everything(0)
        m=NPLAM(30,10)
        initial=np.mean((predict(m,data.x[1])-data.y[1])**2)
        v=m.v.clone()
        m,info=fit_neural(m,data,0,60,.02,[0,0,0],64,20)
        np.testing.assert_array_equal(v,m.v)
        assert np.mean((predict(m,data.x[1])-data.y[1])**2)==pytest.approx(info["validation_mse"])
        assert info["validation_mse"]<initial*.1
        results.append(predict(m,data.x[2]))
    np.testing.assert_array_equal(*results)
    p=tmp_path/"model.pt"
    torch.save(m.state_dict(),p)
    m2=NPLAM(30,10)
    m2.load_state_dict(torch.load(p,weights_only=True))
    np.testing.assert_array_equal(predict(m,data.x[2]),predict(m2,data.x[2]))


@pytest.mark.parametrize("bad",["invalid", "complementary"])
def test_invalid_mode(bad):
    with pytest.raises(ValueError): NPLAM(3,mode=bad)
