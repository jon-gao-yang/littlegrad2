import torch
from littlegrad2.engine import Tensor

def my_test():
    #a = Tensor([2, 3])
    #b = Tensor([5, 10])
    #c = 4

    a = -5
    b = 3
    c = 1

    aVal = Tensor(a)
    bVal = Tensor(b)
    print("a:      | Passed == ", type(aVal) == Tensor)
    print("b:      | Passed == ", type(bVal) == Tensor)
    print("a.data: | Passed == ", aVal.data == a)
    print("b.data: | Passed == ", bVal.data == b)
    print("a+c:    | Passed == ", (aVal+c).data == a+c)
    print("c+a:    | Passed == ", (c+aVal).data == c+a)
    print("a-c:    | Passed == ", (aVal-c).data == a-c)
    print("c-a:    | Passed == ", (c-aVal).data == c-a)
    print("a*c:    | Passed == ", (aVal*c).data == a*c)
    print("c*a:    | Passed == ", (c*aVal).data == c*a)
    print("-a:     | Passed == ", (-aVal).data == -a)
    print("a/c:    | Passed == ", (aVal/c).data == a/c)
    print("c/a:    | Passed == ", (c/aVal).data == c/a)
    print("a**c:   | Passed == ", (aVal**c).data == a**c)
    print("c**a:   | Passed == ", (c**aVal).data == c**a)
    print()

# from karpathy/micrograd test_engine.py:
def test_sanity_check():

    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backprop()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    #print('xmg.grad: ', xmg.grad)
    #print('xpt.grad.item(): ', xpt.grad.item())
    assert xmg.grad == xpt.grad.item()
    print('Karpathy #1: Passed == True')

def test_more_ops():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print('Karpathy #2: Passed == True')
    print()

#############################################################################################

my_test()
test_sanity_check()
test_more_ops()