import numpy
import random
import pickle
'''
This version allows us to experiment with other activation functions: Leaky ReLU and sigmoid
'''
weights = []
biases = []
# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


# RELU activation function
def relu(x):
    return max(0,x)


# this function adds layers by adding a matrix of random weights to the
# list of weight matrices and by adding a vector of zeros for biases.
def add_layer(n_in,n_out):
    global weights
    global biases
    matrix= []
    list = []
    for r in range(0,n_out):
        #list.append(random.random())
        list.append(0)
        row = []
        for c in range(0,n_in):
            row.append(random.random()*0.001)
        matrix.append(row)

    weights.append(matrix)
    biases.append(list)


input_size = 18
output_size = 2
add_layer(18,50)
add_layer(50,50)
add_layer(50,50)
add_layer(50,25)
add_layer(25,2)


# forward propagation to evaluate the neural network
def forward_propagation(x):
    activations = [x]
    inp = x.copy()
    for l in range(0,len(weights)):
        outp = []
        for r in range(0,len(weights[l])):
            temp = biases[l][r]
            for c in range(0,len(weights[l][r])):
                temp += (weights[l][r][c] * inp[c])
            outp.append(relu(temp))
        activations.append(outp.copy())
        inp = outp.copy()
    return activations

# The output is the last column of the matrix returned by forward_propagation()
def eval(x):
    activations = forward_propagation(x)
    return activations[len(activations)-1]


# Back_propagation for training using gradient decent
# w is weights, b is biases, y is output and a is the list of activations.
def back_propagation(w, b, y, a):
    b_grad = [[0 for r in range(0,len(b[i]))] for i in range(0,len(b))]
    w_grad = [[[0 for c in range(0,len(w[l][r]))] for r in range(0,len(w[l]))] for l in range(0,len(w))]
    a_grad = [[0 for j in range(0,len(a[i]))] for i in range(0,len(a))]
    for i in range(0,len(y)):
        a_grad[-1][i] = 2*(a[-1][i] - y[i])
    for l in reversed(range(0,len(w))):
        for r in range(0,len(w[l])):
            for c in range(0,len(w[l][r])):
                if a[l+1][r] > 0:
                    w_grad[l][r][c] = a[l][c] * a_grad[l+1][r]
                    b_grad[l][r] = a_grad[l+1][r]
                    a_grad[l][c] += w[l][r][c] * a_grad[l+1][r]
                # This is if we want to use leaky ReLU
                '''else:
                    w_grad[l][r][c] = a[l][c] * a_grad[l + 1][r]*0.001
                    b_grad[l][r] = a_grad[l + 1][r]*0.001
                    a_grad[l][c] += w[l][r][c] * a_grad[l + 1][r]*0.001'''
                # This is if we want to use sigmoid
                '''w_grad[l][r][c] = a[l][c] * a_grad[l+1][r] * (sigmoid(a[l+1][r])*(1-sigmoid(a[l+1][r])))
                b_grad[l][r] = a_grad[l+1][r] * (sigmoid(a[l+1][r])*(1-sigmoid(a[l+1][r])))
                a_grad[l][c] += w[l][r][c] * a_grad[l+1][r] * (1-sigmoid(a[l+1][r]))'''
    return (w_grad, b_grad)


# another approach to back propagation
# w is weights, b is biases, y is output and a is the list of activations.
def back_propagation2(w, b, y, a):
    # indexing is as follows: l for layer, r for row and c for column
    b_grad = [[0 for r in range(0,len(b[i]))] for i in range(0,len(b))]
    w_grad = [[[0 for c in range(0,len(w[l][r]))] for r in range(0,len(w[l]))] for l in range(0,len(w))]
    # last layer: (a - y)^2, so gradient is 2(a-y)
    grad_withRespectTo_a = []
    for i in range(0,len(y)):
        grad_withRespectTo_a.append(2*(a[len(a)-1][i] - y[i]))

    for l in reversed(range(0,len(w))):
        if l != (len(w)-1):
            temp_g = grad_withRespectTo_a.copy()
            grad_withRespectTo_a = []
            for r1 in range(0,len(w[l])):
                temp = 0
                for r2 in range(0,len(w[l+1])):
                    if a[l + 2][r2] > 0:
                        grad_a_withRespectTo_z = 1
                    else:
                        grad_a_withRespectTo_z = 0
                    temp += (w[l+1][r2][r1] * grad_a_withRespectTo_z * temp_g[r2])
                grad_withRespectTo_a.append(temp)
        for r in range(0,len(w[l])):
            if a[l+1][r]>0:
                grad_a_withRespectTo_z = 1
            else:
                grad_a_withRespectTo_z = 0
            for c in range(0,len(w[l][r])):
                    w_grad[l][r][c] = (a[l][c] * grad_a_withRespectTo_z * grad_withRespectTo_a[r])
            b_grad[l][r] = (grad_a_withRespectTo_z * grad_withRespectTo_a[r]) #* grad_withRespectTo_a[r]
    return (w_grad,b_grad)


# this function uses the forward and backward propagations to train the neural net
# for a specific batch and learning rate. Note that the learning rate remains constant
# for all batches.
def train(batch,learning_rate):
    global weights
    global biases
    b_grad = [[0 for r in range(0,len(biases[l]))] for l in range(0,len(biases))]
    w_grad = [[[0 for c in range(0,len(weights[l][r]))] for r in range(0,len(weights[l]))] for l in range(0,len(weights))]
    for d in batch:
        activations = forward_propagation(d[:input_size])
        (w_grad_next,b_grad_next)= back_propagation(weights,biases,d[input_size:],activations)
        b_grad_t = [[b_grad[l][r]+b_grad_next[l][r] for r in range(0, len(biases[l]))] for l in range(0, len(biases))]
        w_grad_t = [[[w_grad[l][r][c]+w_grad_next[l][r][c] for c in range(0, len(weights[l][r]))] for r in range(0, len(weights[l]))] for l in
                  range(0, len(weights))]
        b_grad = b_grad_t
        w_grad = w_grad_t
    test = eval(batch[0][:input_size])
    biases_t = [[biases[l][r]-((b_grad[l][r]/len(batch))*learning_rate) for r in range(0, len(biases[l]))] for l in range(0, len(biases))]
    weights_t = [[[weights[l][r][c]-((w_grad[l][r][c]/len(batch))*learning_rate) for c in range(0, len(weights[l][r]))] for r in range(0, len(weights[l]))] for l in
              range(0, len(weights))]
    biases = biases_t
    weights = weights_t


# This will test the neural net to find current accuracy.
# Note that the data passed to test should not be used in training.
def test(datas):
    correct = 0
    cost = 0
    for d in datas:
        outp = eval(d[0:input_size])
        if outp == d[input_size:]:
            correct += 1
        for i,y in enumerate(d[input_size:]):
            cost += (outp[i] - y)**2
    print("test results:","cost =",cost/len(datas))


data = []
# these nested for loops create the training set
for i in range(0,64):
    for j in range(0,64):
        for k in range(0,64):
            inp = list("{0:06b}".format(i) + "{0:06b}".format(j) + "{0:06b}".format(k))
            for r in range(0,len(inp)):
                inp[r] = float(inp[r])
            if j^k < i:
                outp = [0,j^k]

                data.append(inp+outp)
            elif i^k < j:
                outp = [1,i^k]

                data.append(inp + outp)
            elif i^j < k:
                outp = [2,i^j]
                data.append(inp + outp)

data = data[:len(data)//20]

random.shuffle(data)

training_data = data[:len(data)//2]
test_data = data[len(data)//2:]

training_chunks = [training_data]
test(test_data)
for i in range(1,21):
    #print(weights[0])
    print("************ epoc"+str(i)+" ***************")
    num = 0
    first = True
    for batch in training_chunks:
        if first:
            print("training...")
            first = False
        num += 1
        train(batch,0.1)
    print("epoc",i,"is done :)")
    test(test_data)

    with open("../data/weights2.txt", "wb") as fp:   #Pickling
        pickle.dump(weights, fp)
    with open("../data/biases2.txt", "wb") as fp:   #Pickling
        pickle.dump(biases, fp)



