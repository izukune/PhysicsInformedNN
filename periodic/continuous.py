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


class PhysicsInformedNN:
    def __init__(self, x0, u0, tb, X_f, layers, lb, ub):
        
        X0 = np.concatenate((x0, 0*x0), 1)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) 
        
        self.lb = lb
        self.ub = ub       
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]
        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]
        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u0 = u0
        
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
      
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u0_pred, _  = self.net_u(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.net_u(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred = self.net_f_u(self.x_f_tf, self.t_f_tf)
        
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred))
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
                
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
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
        X = tf.concat([x,t],1)
        u = self.neural_net(X, self.weights, self.biases)[:,0:1]
        u_x = tf.gradients(u, x)[0]

        return u, u_x

    def net_f_u(self, x, t):
        u, u_x = self.net_u(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]

        f_u = u_t - u_xx
        
        return f_u
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self):
        
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        u_star = self.sess.run(self.u0_pred, tf_dict)  

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)  

        return u_star, f_u_star
    
if __name__ == "__main__": 
     
    #>>>>>>>設定欄-----------------------------------------------------------------
    N_0 = 50
    N_b = 50
    N_f = 10000
    t0 = 0.0
    t1 = 1
    x0 = -5
    x1 = 5
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    #>>>>>>>設定欄-----------------------------------------------------------------
            
    t = np.linspace(t0, t1, 100*(t1-t0))[:,None]
    x = np.linspace(x0, x1, 256*(x1 - x0))[:,None]

    #>>>>>>>初期条件-----------------------------------------------------------------
    c = 0.0
    c0 = 0.0
    ini_func = np.exp(-(x**2))
    #>>>>>>>初期条件-----------------------------------------------------------------

    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    lb = np.array([x0, t0])
    ub = np.array([x1, t1])
    zero_time = np.zeros(256*(x1 - x0))[:,None]

    #初期条件の座標-----------------------------------------------

    xx = np.hstack([x, zero_time])
    uu = ini_func
    idx = np.random.choice(xx.shape[0], N_0, replace=False)
    X_u_train = xx[idx, :]
    u_train = uu[idx,:]
    
    X_f = lb + (ub-lb)*lhs(2, N_f)

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
            
    model = PhysicsInformedNN(X_u_train, u_train, tb, X_f, layers, lb, ub)
             
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    u_pred, f_u_pred = model.predict(X_star)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')    

    X_u_train = xx

    #plotting style-------------------------------
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
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u$', fontsize = 10)
    

    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(top=1-0.5, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, :])

    ims = []

    for i in range(100*(t1-t0)):
        im = ax.plot(x, U_pred[i,:], 'b-', linewidth = 2, label = 'Prediction')
        ims.append(im)
        
    ani = animation.ArtistAnimation(fig, ims, interval=50)    
    ani.save("output.gif", writer="imagemagick")

    plt.show()
