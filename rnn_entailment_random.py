import numpy as np
import collections
import rnn as rnn

# RNN for textual entailment task.
# Compute the word vectors for two sentences, and merge them at the comparison NN layer.

# Using randomly initialized word vectors. (before 11/05/15)
class RNNRTE:

    def __init__(self,wvecDim,outputDim,embeddingDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.embeddingDim = embeddingDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        # Init the composition layers
        self.rnn_left = rnn.RNN(wvecDim,outputDim,numWords,mbSize,rho)
        self.rnn_right = rnn.RNN(wvecDim,outputDim,numWords,mbSize,rho)



    # Init params for the comparison layer
    def initParams(self, W):
        #self.rnn_left.initParams(W)
        #self.rnn_right.initParams(W)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)# why not [numWords, wvecDim]?
        self.L = W


        # embedding transformation layer weights
        # embeddingDim = word2vec dim, wvecDim = composition layer's word vec dim.
        self.We = 0.01*np.random.rand(self.wvecDim, self.embeddingDim)
        self.be = np.zeros((self.embeddingDim))

        # Comparison layer weights
        self.Wc = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.bc = np.zeros((self.wvecDim))

        # Hidden activation weights
        self.W = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W, self.b, self.Ws, self.bs, self.Wc, self.bc, self.We, self.be]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))
        self.dWc = np.empty(self.W.shape)
        self.dbc = np.empty((self.wvecDim))
        self.dWe = np.empty(self.We.shape)
        self.dbe = np.empty((self.embeddingDim))



    def costAndGrad(self,mbdata,test=False):
        """
        mbdata => [TreePair]

        Each datum in the minibatch is a set of two trees (two sentences).
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.
        """
        cost = 0.0
        correct = 0.0
        total = 0.0

        self.L,self.W,self.b,self.Ws,self.bs,self.Wc,self.bc,self.We,self.be = self.stack
        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dWc[:] = 0
        self.dbc[:] = 0
        self.dWe[:] = 0
        self.dbe[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Initialize dL and other params of each rnn
        #self.rnn_left.init_param_grads()
        #self.rnn_right.init_param_grads()

        # Forward prop each tree in minibatch
        for tree_pair in mbdata:
            c,corr,tot = self.forward_prop_all(tree_pair, test)

            cost += c
            correct += corr
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,total

        # Back prop each tree in minibatch
        for tree_pair in mbdata:
            self.back_prop_all(tree_pair)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale

        # Add L2 Regularization
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
        cost += (self.rho/2)*np.sum(self.Wc**2)

        return scale*cost,[self.dL,scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs,scale*(self.dWc+self.rho*self.Wc),scale*self.dbc,scale*(self.dWe+self.rho*self.We),scale*self.dbe]


    def forward_prop_all(self, tree_pair, test=False):
        cost = correct =  total = 0.0

        # Get the feature of both sentences
        # TODO: do some additional initializations to each rnn's parameters?

        # sent_left/sent_right: Node class. we need to use sent_left.hActs for using representation!!!
        #sent_left = self.rnn_left.forwardProp(tree_pair.tree1.root)
        #sent_right = self.rnn_right.forwardProp(tree_pair.tree2.root)
        sent_left = self.forwardProp(tree_pair.tree1.root)
        sent_right = self.forwardProp(tree_pair.tree2.root)

        # Propagate to the comparison layer.
        # Use the representations!
        # Affine
        hActs = np.dot(self.Wc,
                np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.hActs])) + self.bc
                #np.hstack([sent_left.hActs, sent_right.hActs])) + self.bc
        # Relu
        hActs[hActs<0] = 0

        # Softmax layer
        probs = np.dot(self.Ws,hActs) + self.bs
        probs -= np.max(probs)
        probs = np.exp(probs)
        probs = probs/np.sum(probs)

        # TODO: Create a class for softmax layer
        # Save softmax layer values as member vars (temporary solution)
        #self.probs = probs
        tree_pair.probs = probs
        tree_pair.hActs = hActs
        if test: print probs

        label = tree_pair.label

        return cost - np.log(probs[label]), correct + (np.argmax(probs)==label), total+1


    # USED: RNN.forwardProp() will not be called from forward_prop_all()
    # Forward propagate each RNN. Returns the output vector of each sentence!
    def forwardProp(self,node):
        cost = correct =  total = 0.0

        if node.isLeaf:
            node.hActs = self.L[:,node.word]
            node.fprop = True

        else:
            if not node.left.fprop:
                c,corr,tot = self.forwardProp(node.left)
                cost += c
                correct += corr
                total += tot
            if not node.right.fprop:
                c,corr,tot = self.forwardProp(node.right)
                cost += c
                correct += corr
                total += tot
            # Affine
            node.hActs = np.dot(self.W,
                    np.hstack([node.left.hActs, node.right.hActs])) + self.b
            # Relu
            node.hActs[node.hActs<0] = 0

        """"""# Softmax
        node.probs = np.dot(self.Ws,node.hActs) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)
        node.fprop = True
        return cost - np.log(node.probs[node.label]), correct + (np.argmax(node.probs)==node.label),total + 1


    def back_prop_all(self, tree_pair):
        # Softmax layer grad
        # cross entropy error: dE/da = y-t = deltas
        # E.g. [0.11, 0.0008, 0.5] - [0, 1, 0]
        self.deltas = tree_pair.probs
        self.deltas[tree_pair.label] -= 1.0 # target value = 1
        # derivative of weights(softmax layer):
        # dE/dw = delta*z = deltas*hActs
        self.dWs += np.outer(self.deltas, tree_pair.hActs)
        self.dbs += self.deltas


        # Comparison layer grad
        # derivative of weights(comparison hidden layer)
        # delta_j = h'(a_j)*sigma_k(w_kj*delta_k)
        # delta^(l)=h'(a)*sum(w*delta^(l+1)
        # (for calc) => delta^(l) = W^(l).T * delta^(l+1) (element-wise *) f'(z^(l))
        self.deltas = np.dot(self.Ws.T,self.deltas)

        # TODO: Save splitted deltas into a custom comparison layer class
        #self.deltas_left = self.deltas[:self.wvecDim]
        #self.deltas_right = self.deltas[self.wvecDim:]
        #if self.rnn_left.deltas.size != 30 or self.rnn_right.deltas != 30:
        #    print 'debug'

        error=None
        if error is not None:
            self.deltas += error
        self.deltas *= (tree_pair.hActs != 0)
        # not a leaf node:
        self.dWc += np.outer(self.deltas,
                    np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.hActs]))
        self.dbc += self.deltas


        # added
        self.deltas = np.dot(self.Wc.T,self.deltas)
        # NOTE: make sure to create deep copy
        """self.rnn_left.deltas = np.empty_like(self.deltas[:self.wvecDim])
        self.rnn_left.deltas[:] = self.deltas[:self.wvecDim]
        self.rnn_right.deltas = np.empty_like(self.deltas[self.wvecDim:])
        self.rnn_right.deltas[:] = self.deltas[self.wvecDim:]

        # Composition layers grad. dL, dW, db will be calculated.
        self.rnn_left.backProp(tree_pair.tree1.root)
        self.rnn_right.backProp(tree_pair.tree2.root)"""
        self.deltas_left = np.empty_like(self.deltas[:self.wvecDim])
        self.deltas_left[:] = self.deltas[:self.wvecDim]
        self.deltas_right = np.empty_like(self.deltas[self.wvecDim:])
        self.deltas_right[:] = self.deltas[self.wvecDim:]

        # Composition layers grad. dL, dW, db will be calculated.
        self.backProp(tree_pair.tree1.root,self.deltas_left)
        self.backProp(tree_pair.tree2.root,self.deltas_right)


    # NOT USED
    def backProp(self,node,deltas,error=None):

        # Clear nodes
        node.fprop = False

        """# Softmax grad
        deltas = node.probs
        deltas[node.label] -= 1.0
        self.dWs += np.outer(deltas,node.hActs)
        self.dbs += deltas
        deltas = np.dot(self.Ws.T,deltas)"""

        # self.deltas is already calculated
        # [NOTE: DO NOT assign the instance itself. make sure to create a deep copied value!!!]
        deltas_local = np.empty_like(deltas)
        deltas_local[:] = deltas    # self.deltas has wvecDim

        if error is not None:
            deltas_local += error

        #if deltas.size != 30:#node.hActs.size != 30:
        #    print 'test'
        deltas_local *= (node.hActs != 0)


        # Leaf nodes update word vecs
        if node.isLeaf:
            self.dL[node.word] += deltas_local
            return

        # Hidden grad
        if not node.isLeaf:
            self.dW += np.outer(deltas_local,
                    np.hstack([node.left.hActs, node.right.hActs]))
            self.db += deltas_local
            # Error signal to children
            deltas = np.dot(self.W.T, deltas_local) # deltas.size = wvecDim*2 at this point
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])


    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        #print self.stack[1]
        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]
        #print self.stack[1]
        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid, last=False):
        import cPickle as pickle
        pickle.dump(self.stack,fid)
        if last:
            print self.stack[0][:,467]
            #print self.stack[0]
            #print self.stack[1]
            #print self.stack[4]

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)
        #print self.stack
        print self.stack[0]
        #print self.stack[1]
        #print self.stack[4]

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None]
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j],numGrad,err)

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)


if __name__ == '__main__':

    import tree_rte as treeM
    train,vocab = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5
    print numW

    rnn = RNNRTE(wvecDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]

    print "Numerical gradient check..."
    rnn.check_grad(mbData)






