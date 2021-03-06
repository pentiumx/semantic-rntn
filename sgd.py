import numpy as np
import random
import tree_rte as tr
import logging
import time

RUN_ON_SERVER = False

class SGD:

    def __init__(self, model,alpha=1e-2,minibatch=30,
                 optimizer='sgd', model_dg=None):
        self.model = model
        self.model_dg = model_dg

        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
        self.optimizer = optimizer
        if self.optimizer == 'sgd':
            print "Using sgd.."
        elif self.optimizer == 'adagrad':
            print "Using adagrad..."
            epsilon = 1e-8
            self.gradt = [epsilon + np.zeros(W.shape) for W in self.model.stack]
            self.gradt_dg = [epsilon + np.zeros(W.shape) for W in self.model_dg.stack]
        else:
            raise ValueError("Invalid optimizer")

        self.costt = []
        self.expcost = []
        logging.basicConfig(filename='train.log',level=logging.DEBUG)


    def run_using_denotation(self, trees, dg_lines):
        """
        Runs stochastic gradient descent with model as objective.
        """

        m = len(trees)

        # randomly shuffle data
        random.shuffle(trees)

        for i in xrange(0,m-self.minibatch+1,self.minibatch):
            self.it += 1

            # Update relevant params from the rnn_dg model
            self.model.transfer_params(self.model_dg.stack)

            # Get data as much as it's specified by minibatch num
            mb_data = trees[i:i+self.minibatch]

            # Get the gradient
            cost,grad = self.model.costAndGrad(mb_data)

            # compute exponentially weighted cost
            if np.isfinite(cost):
                if self.it > 1:
                    self.expcost.append(.01*cost + .99*self.expcost[-1])
                else:
                    self.expcost.append(cost)

            if self.optimizer == 'sgd':
                update = grad
                scale = -self.alpha

            elif self.optimizer == 'adagrad':
                # trace = trace+grad.^2
                self.gradt[1:] = [gt+g**2
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # update = grad.*trace.^(-1/2)
                update =  [g*(1./np.sqrt(gt))
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # handle dictionary separately
                dL = grad[0]
                dLt = self.gradt[0]
                for j in dL.iterkeys():
                    dLt[:,j] = dLt[:,j] + dL[j]**2
                    dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))
                update = [dL] + update
                scale = -self.alpha


            # update params
            self.model.updateParams(scale,update,log=False)

            self.costt.append(cost)
            if self.it%1 == 0:
                print "Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])

            # consider CPU pressure when running on server.
            if RUN_ON_SERVER:
                time.sleep(0.01)

            ###############################
            #  Now train from the DG source
            ###############################
            # Update relevant params from the rnn model
            self.model_dg.transfer_params(self.model.stack)
            self.train_denotation(dg_lines, i)

            if RUN_ON_SERVER:
                time.sleep(0.01)


    def run(self,trees):
        """
        Runs stochastic gradient descent with model as objective.
        """

        m = len(trees)

        # randomly shuffle data
        random.shuffle(trees)

        for i in xrange(0,m-self.minibatch+1,self.minibatch):
            self.it += 1

            # Get data as much as it's specified by minibatch num
            mb_data = trees[i:i+self.minibatch]

            # Get the gradient?
            cost,grad = self.model.costAndGrad(mb_data)

            # compute exponentially weighted cost
            if np.isfinite(cost):
                if self.it > 1:
                    self.expcost.append(.01*cost + .99*self.expcost[-1])
                else:
                    self.expcost.append(cost)

            if self.optimizer == 'sgd':
                update = grad
                scale = -self.alpha

            elif self.optimizer == 'adagrad':
                # trace = trace+grad.^2
                self.gradt[1:] = [gt+g**2
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # update = grad.*trace.^(-1/2)
                update =  [g*(1./np.sqrt(gt))
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # handle dictionary separately
                dL = grad[0]
                dLt = self.gradt[0]
                for j in dL.iterkeys():
                    dLt[:,j] = dLt[:,j] + dL[j]**2
                    dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))
                update = [dL] + update
                scale = -self.alpha


            # update params
            self.model.updateParams(scale,update,log=False)

            self.costt.append(cost)
            if self.it%1 == 0:
                print "Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])


    def train_denotation(self, lines, i, dataset='denotation'):
        #for i in xrange(0, m-self.minibatch+1, self.minibatch):
        self.it += 1

        # Get data as much as it's specified by minibatch num
        # Load&parse trees beforehand
        # Get data as much as it's specified by minibatch num
        # Parse&load tree data beforehand
        trees = list(tr.inputarray(lines[i:i+self.minibatch], dataset))

        # map word indices to loaded trees
        tr.map_words_to_trees(trees, dataset)
        mb_data = trees#mb_data = trees[i:i+self.minibatch]

        # Get the gradient?
        self.model.use_dg = True
        cost,grad = self.model_dg.costAndGrad(mb_data)

        # compute exponentially weighted cost
        if np.isfinite(cost):
            if self.it > 1:
                self.expcost.append(.01*cost + .99*self.expcost[-1])
            else:
                self.expcost.append(cost)

        if self.optimizer == 'sgd':
            update = grad
            scale = -self.alpha

        elif self.optimizer == 'adagrad':
            # trace = trace+grad.^2
            self.gradt_dg[1:] = [gt+g**2
                    for gt,g in zip(self.gradt_dg[1:],grad[1:])]
            # update = grad.*trace.^(-1/2)
            update =  [g*(1./np.sqrt(gt))
                    for gt,g in zip(self.gradt_dg[1:],grad[1:])]
            # handle dictionary separately
            dL = grad[0]
            dLt = self.gradt_dg[0]
            for j in dL.iterkeys():
                dLt[:,j] = dLt[:,j] + dL[j]**2
                dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))
            update = [dL] + update
            scale = -self.alpha


        # update params
        self.model_dg.updateParams(scale,update,log=False)

        self.costt.append(cost)
        if self.it % 1 == 0:
            print "Iter %d (DG): Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])

    def run_denotation(self, lines, dataset='denotation'):
        """
        Runs stochastic gradient descent with model as objective.
        This method is about training with a very large dataset.
        """
        m = len(lines)

        # randomly shuffle data
        #random.shuffle(trees)

        for i in xrange(0, m-self.minibatch+1, self.minibatch):
            self.it += 1

            # Get data as much as it's specified by minibatch num
            # Load&parse trees beforehand
            # Get data as much as it's specified by minibatch num
            # Parse&load tree data beforehand
            trees = list(tr.inputarray(lines[i:i+self.minibatch], dataset))

            # map word indices to loaded trees
            tr.map_words_to_trees(trees, dataset)
            mb_data = trees#mb_data = trees[i:i+self.minibatch]

            # Get the gradient?
            cost,grad = self.model.costAndGrad(mb_data)

            # compute exponentially weighted cost
            if np.isfinite(cost):
                if self.it > 1:
                    self.expcost.append(.01*cost + .99*self.expcost[-1])
                else:
                    self.expcost.append(cost)

            if self.optimizer == 'sgd':
                update = grad
                scale = -self.alpha

            elif self.optimizer == 'adagrad':
                # trace = trace+grad.^2
                self.gradt[1:] = [gt+g**2
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # update = grad.*trace.^(-1/2)
                update =  [g*(1./np.sqrt(gt))
                        for gt,g in zip(self.gradt[1:],grad[1:])]
                # handle dictionary separately
                dL = grad[0]
                dLt = self.gradt[0]
                for j in dL.iterkeys():
                    dLt[:,j] = dLt[:,j] + dL[j]**2
                    dL[j] = dL[j] * (1./np.sqrt(dLt[:,j]))
                update = [dL] + update
                scale = -self.alpha


            # update params
            self.model.updateParams(scale,update,log=False)

            self.costt.append(cost)
            if self.it % 1 == 0:
                print "Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])

