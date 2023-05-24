import numpy as np
#
class neurnet():
    def __init__(self, num_ins:int, num_layers:list, num_outs:int, x:np.ndarray, afuns:list, y:np.ndarray, lfun:str) -> dict:
        self.input = x
        self.afuns = afuns
        self.output = y
        self.lfun = lfun
        self.num_layers = num_layers
        self.num_ins = num_ins
        self.num_outs = num_outs
        #
    def initnet(self):
        n = len(self.num_layers)+1
        w = n*[1]
        b = n*[1]
        if (len(self.num_layers)>=1):
            ni = self.num_ins
            nl = self.num_layers[0]
            low = -1/np.sqrt(ni)
            high = 1/np.sqrt(ni)
            #
            wl = np.random.uniform(low = low, high = high, size = (nl, self.num_ins))
            bl = np.random.uniform(low = low, high = high, size = (nl, 1))
            w[0]= wl
            b[0] = bl
            j = 0
            for i in range(1,len(self.num_layers)):
                ni = nl
                low = -1/np.sqrt(ni)
                high = 1/np.sqrt(ni)
                #
                wl = np.random.uniform(low = low, high = high, size = (self.num_layers[i], nl))
                bl = np.random.uniform(low = low, high = high, size = (self.num_layers[i], 1))
                w[i]= wl
                b[i] = bl
                nl = self.num_layers[i]
                j = i
            #
            ni = nl
            low = -1/np.sqrt(ni)
            high = 1/np.sqrt(ni)
            #
            wl = np.random.uniform(low = low, high = high, size = (self.num_outs, nl))
            bl = np.random.uniform(low = low, high = high, size = (self.num_outs, 1))
            
            w[j+1] = wl
            b[j+1] = bl
        else:
            ni = self.num_ins
            low = -1/np.sqrt(ni)
            high = 1/np.sqrt(ni)
            #
            wl = np.random.uniform(low = low, high = high, size = (self.num_outs, self.num_ins))
            bl = np.random.uniform(low = low, high = high, size = (self.num_outs, 1))
            w[0] = wl
            b[0] = bl
        self.weight = w
        self.bias = b
    #
    def act_fun(self, x:np.ndarray, name:str)->np.ndarray:
        if name =='sigmoid':
            y = 1.0/(np.exp(-1.0*x)+1)
        elif name=='relu':
            y = np.maximum(0, x)
        elif name =='tanh':
            y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return y
     #
    def dfun(self,x:np.ndarray, afun:str)->np.ndarray:
        a = self.act_fun(x, afun)
        if afun=='sigmoid':
            y = a*(1-a)
        elif afun == 'tanh':
            y = 1-a
        elif afun=='relu':
                y = np.where(x<=0, 0, 1)
        return y
    #
    def ffn(self)->tuple:
        nw = len(self.weight)
        w = self.weight
        b = self.bias
        z = nw*[1]
        a = nw*[1]
        al = self.input
        # a.append(al)
        for j in range(nw):
            afun = self.afuns[j]
            wl = w[j].astype('float64')
            bl = b[j].astype('float64')
            zl = np.dot(wl,al)+bl
            al = self.act_fun(zl, afun)
            z[j] = zl
            a[j] = al
        return (z, a)
    #
    def calc_loss(self)-> np.ndarray:
        afun = self.afuns[-1]
        y = self.output
        z, a = self.ffn()
        zL = z[-1]
        aL = a[-1]
        if self.lfun =='MSE':
            dL = (aL-y)*self.dfun(zL, afun)
        elif self.lfun == 'BCE':
            dL = ((1-y)/(1-aL)-y/aL)*self.dfun(zL, afun)
        return dL
    #
    def net_loss(self)->list:
        w = self.weight
        z = self.ffn()[0]
        afun = self.afuns[-1]
        dl = self.calc_loss()
        L = len(w)
        d = L*[1]
        j = 0
        d[j] = dl
        for i in range(L-2, -1, -1):
            afun = self.afuns[i]
            dl = (w[i+1].T@dl)*self.dfun(z[i], afun)
            j = j+1
            d[j] = dl
        d.reverse()
        return d
    #
    def update_network(self,lr:float):
        # lr: learning rate
        x = self.input
        m = x.shape[1]
        w0 = self.weight
        b0 = self.bias
        w = []
        b = []
        a = self.ffn()[1]
        d = self.net_loss()
        L = len(w0)
        for j in range(L-1, -1, -1):
            dl = d[j]
            ndl = dl.shape[0]
            #
            if j==0:
                al = x
            else:
                al = a[j-1]
            nal = al.shape[0]
            #
            dl1 = np.zeros((m, ndl, 1))
            al1 = np.zeros((m, 1, nal))
            #
            dl1[:,:,0] = dl.T
            al1[:,0,:] = al.T

            sw = np.sum(dl1@al1, 0)
            sb = np.sum(dl, 1).reshape(ndl,1)
            #
            wl = w0[j]-lr*sw/m
            bl = b0[j]-lr*sb/m
            #
            w.append(wl)
            b.append(bl)
        #
        w.reverse()
        b.reverse()
        self.weight = w
        self.bias = b