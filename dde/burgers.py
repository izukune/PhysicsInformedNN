import deepxde as dde
import numpy as np

# 自動微分による微分項を求める関数
def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

# 空間領域と時間領域
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 境界条件
bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
# 初期条件
ic = dde.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

# 学習データの登録(コロケーションポイント / 境界条件 / 初期条件)
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
)
# ニューラルネットの構築(構造 / 活性化関数 / パラメータの初期化)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
# インスタンスの生成
model = dde.Model(data, net)
# パラメータ更新アルゴリズム
model.compile("adam", lr=1e-3)
# 学習のエポック数
model.train(epochs=15000)
model.compile("L-BFGS")
# 損失関数の遷移
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
