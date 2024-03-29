"""
author: いずくね
group: 東京理科大
Date: 2021/1月15日
explanation: 深層学習で(2+1)次元の非線形偏微分方程式を解く
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

#ランダム生成値の固定---------------------------------
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:

    #インスタンスが生成されたときに実行される関数
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.t_u = X_u[:,0:1]
        self.x_u = X_u[:,1:2]
        self.y_u = X_u[:,2:3]
        self.t_f = X_f[:,0:1]
        self.x_f = X_f[:,1:2]
        self.y_f = X_f[:,2:3]
        self.u = u
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])  
        self.y_u_tf = tf.placeholder(tf.float32, shape=[None, self.y_u.shape[1]])       
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf, self.y_u_tf) 
        self.f_pred = self.net_f(self.t_f_tf, self.x_f_tf, self.y_f_tf)  
        
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        init = tf.global_variables_initializer()
        self.sess.run(init)

    #初期のweightとbiasの生成
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    #切断正規分布に従うweightのランダム初期値を生成
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    #ニューラルネットワークの構築
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        return tf.add(tf.matmul(H, W), b)

    #出力uのニューラルネットワーク
    def net_u(self, t, x, y):
        return self.neural_net(tf.concat([t,x, y],1), self.weights, self.biases)
    
    #出力fのニューラルネットワーク(自動微分)
    def net_f(self, t, x, y):
        u = self.net_u(t, x, y)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xy = tf.gradients(u_x, y)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxy = tf.gradients(u_xx, y)[0]
        u_y = tf.gradients(u, y)[0]
        u_yx = tf.gradients(u_y, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_yyx = tf.gradients(u_yy, x)[0]
        u_yyy = tf.gradients(u_yy, y)[0]
        #>>>>>>>>解きたい方程式::::::::::::::::::::::::::::::::
        return u_t - u_xx - u_yy

    #コールバック関数の定義
    def callback(self, loss):
        print('Loss:', loss)

    #ニューラルネットワークのトレーニング
    def train(self):
        tf_dict = {self.x_u_tf: self.x_u,
                   self.t_u_tf: self.t_u,
                   self.y_u_tf: self.y_u,
                   self.u_tf: self.u,
                   self.x_f_tf: self.x_f,
                   self.t_f_tf: self.t_f,
                   self.y_f_tf: self.y_f}
        self.optimizer.minimize(self.sess, feed_dict = tf_dict,
                                fetches = [self.loss], loss_callback = self.callback)

    #出力されたuの格子点(X_star)の値を変数に代入
    def predict(self, X_star):
        return self.sess.run(self.u_pred, {self.t_u_tf: X_star[:,0:1], 
                                           self.x_u_tf: X_star[:,1:2], 
                                           self.y_u_tf: X_star[:,2:3]})

if __name__ == "__main__":

    #>>>>>>>設定--------------------------------------------------
    N_u = 1000   #初期条件と境界条件の学習データ数
    N_f = 30000  #コロケーションポイントの数
    x0 = -5      #xの始点
    x1 = 5       #xの終点
    y0 = -5      #yの始点
    y1 = 5       #yの終点
    t0 = 0       #tの始点
    t1 = 1       #tの終点
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1] #NNの構造
    N = 200      #x,yの分割数
    Nt = 100     #tの分割数
    #>>>>>>>設定--------------------------------------------------

    #初期条件------------------------------------------------------
    xi = np.linspace(x0, x1, N)
    yi = np.linspace(y0, y1, N)
    Xi, Yi = np.meshgrid(xi, yi)
    ui = np.exp(-(Xi**2 + Yi**2))

    tti = np.zeros(N*N)[:,None]
    xxi = Xi.flatten()[:,None]
    yyi = Yi.flatten()[:,None]
    uui = ui.flatten()[:,None]

    X_u_train = np.hstack([tti, xxi, yyi])
    u_train = uui

    #初期条件データをNuの数だけ抽出する----------------------------------
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    #三次元座標格子点の生成---------------------------------------------
    x = np.linspace(x0, x1, N)
    y = np.linspace(y0, y1, N)
    t = np.linspace(t0, t1, Nt)
    X, Y, T = np.meshgrid(x, y, t)
    
    tt = T.flatten()[:,None]
    xx = X.flatten()[:,None]
    yy = Y.flatten()[:,None]
    X_star = np.hstack([tt, xx, yy])
    
    #コロケーションポイントの生成----------------------------------------
    lb = np.array([t0, x0, y0])  #<lower bound>
    ub = np.array([t1, x1, y1])  #<upper bound>
    X_f_train = lb + (ub-lb)*lhs(3, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    #PhysicsInformedNNクラスにデータを渡す------------------------------
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    
    #演算の実行-------------------------------------------------------
    start_time = time.time()
    model.train()
    run_time = time.time() - start_time
    print('Training time:', run_time)
    
    #演算結果を変数に格納--------------------------------------------
    u_pred = model.predict(X_star)

    #Plotting style-----------------------------------------------
    ims = []    #gifアニメーションの格納箱
    L = Nt / 100
    fig = plt.figure()
    for j in range(100):
        U = []
        for i in range(N*N):
            U = np.append(U, u_pred[Nt*i + L*j])
        U = U.reshape(N, N)
        im = plt.imshow(U,interpolation='nearest', extent=[x0, x1, y0, y1],
                        cmap='seismic', vmin=0, vmax=1)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50)
    plt.colorbar()
    ani.save("output.gif", writer="imagemagick")
    plt.show()
