from includes import *

y = Variable(dtype(np.array([21])))
x = Variable(dtype(np.array([9])), requires_grad=True)

z = x*y
z.backward(x, retain_graph=True)
print('z: {}'.format(z))
print('x.grad: {}'.format(x.grad))

gx = Variable(x.grad, requires_grad=True)
m = z*gx

z.retain_grad()
gx.retain_grad()
m.backward()

print('gx.grad: {}'.format(gx.grad))
print('x.grad: {}'.format(x.grad))
