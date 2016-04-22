import numpy as np
import pickle, random, pdb
acc=0
def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))
def onehot(z):
    out=np.zeros(10)
    out[z]=1
    print " index ",z," ",out
    return out
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    print "SUM :",np.sum(dist)
    return dist
class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.01, reg_lambda=0.01):
        """
        Initialize the network with input, output sizes, weights, biases, learning_rate and regularization parameters
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :param reg_lambda: regularization rate lambda
        :return: None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01 # Weight matrix for input to hidden
        self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01 # Weight matrix for hidden to output
        self.bh = np.zeros(self.hidden_size) # hidden bias
	self.bh = np.array([self.bh])
        self.by = np.zeros(self.output_dim) # output bias
	self.by=np.array([self.by])
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    
    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # Add code to calculate h_a and probs
	#print self.bh
	print " X "
	print np.shape(X)
	print " Wxh "
	print np.shape(self.Wxh)
	h_a=sigmoid(np.dot(X,self.Wxh.transpose()+self.bh))
	#print (np.dot(self.Wxh,X)+self.bh)
	print " h_a "
	print np.shape(h_a)
	print " Why "
	print np.shape(self.Why)
	#pa = np.dot(h_a,self.Why.transpose())+self.by
	#print np.shape(pa)
		
	pa= sigmoid(np.dot(h_a,self.Why.transpose()+self.by))
	#print h_a[0] 
	#print probs 
	print np.shape(h_a)
	print self.Why.transpose()
	print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
	probs=softmax(np.dot(h_a,self.Why.transpose()))
	#probs=softmax(probs)
	probs=np.array(probs[0])
	print probs
	print np.shape(probs)
        return h_a, probs ,pa
    

    def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
        """
        Add regularization terms to the weights
        :param dWhy: weight derivative from hidden to output
        :param dWxh: weight derivative from input to hidden
        :param Why: weights from hidden to output
        :param Wxh: weights from input to hidden
        :return: dWhy, dWxh
        """
        # Add code to calculate the regularized weight derivatives
	
        return dWhy, dWxh

    def _update_parameter(self, dWxh, dWhy,dbh, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        # Add code to update all the weights and biases here
	print "update"
	self.bh=np.array([self.bh[0]])
	self.by=np.array([self.by[0]])
	print np.shape(self.bh)
	print np.shape(dbh)
	self.Wxh+=self.learning_rate*(dWxh.transpose())
	self.Why+=self.learning_rate*(dWhy.transpose())
	self.bh+=self.learning_rate*(dbh)
	self.by+=self.learning_rate*(dby)

    def _back_propagation(self, X, t, h_a, probs,pa):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
	#h_a=np.array([h_a])
	probs=np.array([probs])
	print np.shape(h_a)
	#pa=np.array([pa])
	X=np.array([X])
	print "XX",np.shape(X)
	print np.shape(probs)
	print "************************************"
	print np.shape(sigmoid_prime(pa))
	print "********************************"
	delta =np.multiply((probs - t) , sigmoid_prime(pa))
	print np.shape(delta)	
	print np.shape(h_a)
	print np.shape(h_a.transpose())
	dby=delta
	dWhy = np.dot(h_a.transpose(),delta)
		
	print "**********************************"
	print np.shape(dWhy),"DWHY"
        # Add code to compute the derivatives and return
	delta =np.multiply(np.dot(delta,self.Why),sigmoid_prime(h_a))
	print np.shape(delta)	
	print (delta.transpose())
	print np.shape(X)
	dbh=delta
	dWxh = np.dot(X.transpose(),delta)
	print dWxh
	print "BIAS ",np.shape(dbh),np.shape(dby),np.shape(self.bh),np.shape(self.by)	
        return dWxh, dWhy, dbh, dby

    def _calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
        """
        Calculate the smoothened loss over the set [of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: smooth loss
        """
        if regularizer_type == 'L2':
            # Add regulatization term to loss

            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    
    def predict(self, X,Y):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
	global acc
        # Implement the forward pass and return the output class (argmax of the softmax outputs)
	a,b,c=self._feed_forward(X)
	print " PREDICT ",(b.argmax()+1)
	if(b.argmax()==Y.argmax()):
		acc+=1
	print b
	print Y

    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

if __name__ == "__main__":
    """
    Toy problem where input = target
    """
    nn = NeuralNetwork(384,8,10)
    #data=pickle.load(open("/home/venkat/AML_LAB/data1","r"))
    '''inputs=data
    print inputs
    inputs = []
    targets = []
    for i in range(1000):
        num = random.randint(0,3)
        inp = np.zeros((4,))
        inp[num] = 1
        inputs.append(inp)	
        targets.append(num)
    '''
    #inputs =data
    target = []
    #print inputs
    targets = pickle.load(open("/home/hubatrix/AML_LAB/Group2_t2/this.p","r"))

    #print targets
    count=0
    for i in targets:
	target.append(onehot(i))
    #print target
    for i in range(1,2000):
	data=pickle.load(open("/home/hubatrix/AML_LAB/Group2_t2/data"+str(i)))
    	
    #nn.train(inputs[:800], targets[:800], (inputs[800:], targets[800:]), 10, regularizer_type='L2')
    #print nn.predict([0,1,0,0])
        ha,probs,pa = nn._feed_forward(np.array([data]))
	#print "Saaaa :",np.array(data)
        dWxh,dWhy,dbh,dby=nn._back_propagation(np.array(data),target[i],ha,probs,pa)
        nn._update_parameter(dWxh,dWhy,dbh,dby)
    for j in range(2001,2200):
       
	data=pickle.load(open("/home/hubatrix/AML_LAB/Group2_t2/data"+str(j)))
	nn.predict(np.array([data]),target[j])
	print targets[j]
	count+=1
    #data=pickle.load(open("/home/venkat/AML_LAB/data"+str(3)))
    #nn.predict(np.array([data]),target[3])
    #print targets[3]
    print "ACC",acc
    print " % ",acc/float(count)
