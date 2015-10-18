import numpy as np
import collections
import rnn as rnn

# RNN for textual entailment task.
# Compute the word vectors for two sentences, and merge them at the comparison NN layer.
class RNNRTE:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        # Init the composition layers
        self.rnn_left = rnn.RNN(wvecDim,outputDim,numWords,mbSize,rho)
        self.rnn_right = rnn.RNN(wvecDim,outputDim,numWords,mbSize,rho)
        self.rnn_left.initParams()
        self.rnn_right.initParams()


    # Init params for the comparison layer
    def initParams(self):

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.W = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))


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

        self.L,self.W,self.b,self.Ws,self.bs = self.stack
        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Initialize dL and other params of each rnn
        self.rnn_left.init_param_grads()
        self.rnn_right.init_param_grads()

        # Forward prop each tree in minibatch
        for tree_pair in mbdata:
            c,corr,tot = self.forward_prop_all(tree_pair)

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

        return scale*cost,[self.dL,scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def forward_prop_all(self, tree_pair):
        cost = correct =  total = 0.0

        # Get the feature of both sentences
        # TODO: do some additional initializations to each rnn's parameters?
        sent_left = self.rnn_left.forwardProp(tree_pair.tree1.root)
        sent_right = self.rnn_right.forwardProp(tree_pair.tree2.root)

        # Propagate to the comparison layer
        # Affine
        hActs = np.dot(self.W,
                np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.right.hActs])) + self.b
        # Relu
        hActs[hActs<0] = 0

        # Softmax layer
        probs = np.dot(self.Ws,hActs) + self.bs
        probs -= np.max(probs)
        probs = np.exp(probs)
        probs = probs/np.sum(probs)

        # TODO: Create a class for softmax layer
        # Save softmax layer values as member vars (temporary solution)
        self.probs = probs
        self.hActs = hActs

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
        return node


    def back_prop_all(self, tree_pair):
        # Softmax layer grad
        self.deltas = self.probs
        self.deltas[tree_pair.label] -= 1.0
        self.dWs += np.outer(self.deltas, self.hActs)
        self.dbs += self.deltas
        self.deltas = np.dot(self.Ws.T,self.deltas)


        # TODO: Save splitted deltas into a custom comparison layer class
        #self.deltas_left = self.deltas[:self.wvecDim]
        #self.deltas_right = self.deltas[self.wvecDim:]

        #if self.rnn_left.deltas.size != 30 or self.rnn_right.deltas != 30:
        #    print 'bug'

        # Comparison layer grad
        self.dW += np.outer(self.deltas,
                    np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.hActs]))
        self.db += self.deltas

        # added
        self.deltas = np.dot(self.W.T,self.deltas)
        self.rnn_left.deltas = self.deltas[:self.wvecDim]
        self.rnn_right.deltas = self.deltas[self.wvecDim:]

        # Composition layers grad
        self.rnn_left.backProp(tree_pair.tree1.root)
        self.rnn_right.backProp(tree_pair.tree2.root)

    # NOT USED
    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        """# Softmax grad
        deltas = node.probs
        deltas[node.label] -= 1.0
        self.dWs += np.outer(deltas,node.hActs)
        self.dbs += deltas
        deltas = np.dot(self.Ws.T,deltas)"""

        if error is not None:
            self.deltas += error

        self.deltas *= (node.hActs != 0)

        # Leaf nodes update word vecs
        if node.isLeaf:
            self.dL[node.word] += self.deltas
            return

        # Hidden grad
        if not node.isLeaf:
            self.dW += np.outer(self.deltas,
                    np.hstack([node.left.hActs, node.right.hActs]))
            self.db += self.deltas
            # Error signal to children
            self.deltas = np.dot(self.W.T, self.deltas)
            self.backProp(node.left, self.deltas[:self.wvecDim])
            self.backProp(node.right, self.deltas[self.wvecDim:])


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

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

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
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5
    print numW

    rnn = RNNRTE(wvecDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]

    print "Numerical gradient check..."
    rnn.check_grad(mbData)






