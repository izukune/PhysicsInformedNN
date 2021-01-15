import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

#ランダム生成値の固定---------------------------------
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
      
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)         
        
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        init = tf.global_variables_initializer()
        self.sess.run(init)
         
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        f = u_t + 6*u*u_x + u_xxx
        return f
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    def predict(self, X_star):    
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})    
        return u_star, f_star
    
if __name__ == "__main__":       

    #>>>>>>>設定欄-----------------------------------------------------------------
    N_u = 300    #初期条件と境界条件の学習データ数
    N_f = 10000  #コロケーションポイントの数
    x0 = -5      #xの始点
    x1 = 5       #xの終点
    t0 = 0       #tの始点
    t1 = 1       #tの終点
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1] #NNの構造
    #>>>>>>>設定欄-----------------------------------------------------------------

    t = np.linspace(t0, t1, 100*(t1-t0))[:,None]
    x = np.linspace(x0, x1, 128*(x1 - x0))[:,None]

    #>>>>>>>初期条件-----------------------------------------------------------------
    c = 7.0      #KdVの伝播速度
    c0 = 0.0     #KdVの初期位置
    ini_func = (c / 2.0) / (np.cosh((np.sqrt(c) / 2.0) * (x - c0)) ** 2)  #初期条件
    #>>>>>>>初期条件-----------------------------------------------------------------

    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))     

    lb = np.array([x0, t0])
    ub = np.array([x1, t1])  

    zero_time = np.zeros(128*(x1 - x0))[:,None]
    bound_data = x0 * np.ones(100*(t1-t0))[:,None]
    zero_u = np.zeros(100*(t1-t0))[:,None]

    #初期条件の座標-----------------------------------------------
    xx1 = np.hstack([x, zero_time])
    uu1 = ini_func
    #境界条件1の座標----------------------------------------------
    xx2 = np.hstack([bound_data, t])
    uu2 = zero_u
    #境界条件2の座標----------------------------------------------
    xx3 = np.hstack([-bound_data, t])
    uu3 = zero_u

    #学習データを一つにまとめる-------------------------------------
    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])
    
    #コロケーションポイントの生成-----------------------------------
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    
    #学習データをNuの数だけ抽出する----------------------------------
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
        
    #PhysicsInformedNNクラスにデータを渡す---------------------------
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    
    #演算の実行----------------------------------------------------
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    #演算結果を変数に格納--------------------------------------------
    u_pred, f_pred = model.predict(X_star)                 
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    np.savetxt('./data/myfile.txt', U_pred)
    

    #plotting style----------------------------------------------------------
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')
     
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u$', fontsize = 10)
    

    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(top=1-0.5, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, :])

    #gifアニメーションの作成----------------------------------------------------------
    ims = []
    for i in range(100*(t1-t0)):
        im = ax.plot(x, U_pred[i,:], 'b-', linewidth = 2, label = 'Prediction')
        ims.append(im)
        
    ani = animation.ArtistAnimation(fig, ims, interval=50)    
    ani.save("output.gif", writer="imagemagick")
    #====>>output.gifに作ったgifアニメーションが格納される--------------------------------

    plt.show()