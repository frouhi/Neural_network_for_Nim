import random
import pickle

costs = []
weights = []
biases = []

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
            row.append(random.random()*0.01)
        matrix.append(row)

    weights.append(matrix)
    biases.append(list)
input_size = 18
output_size = 9
add_layer(18,100)
#add_layer(100,100)
add_layer(100,50)
add_layer(50,50)
add_layer(50,25)
add_layer(25,9)
def forward_propagation(x):
    #global weights
    #global biases
    activations = [x]
    inp = x.copy()
    for l in range(0,len(weights)):
        outp = []
        for r in range(0,len(weights[l])):
            z = biases[l][r]
            for c in range(0,len(weights[l][r])):
                z += (weights[l][r][c] * inp[c])
            outp.append(max(0,z))
        activations.append(outp.copy())
        inp = outp.copy()
    #print(activations[len(activations)-1])
    return activations

def eval(x):
    activations = forward_propagation(x)
    return activations[len(activations)-1]


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

    return (w_grad,b_grad)
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
                    #############################
                    if a[l + 2][r2] > 0:
                        grad_a_withRespectTo_z = 1
                    else:
                        grad_a_withRespectTo_z = 0
                    #############################
                    temp += (w[l+1][r2][r1] * grad_a_withRespectTo_z * temp_g[r2])
                grad_withRespectTo_a.append(temp)


        for r in range(0,len(w[l])):
            if a[l+1][r]>0:
                grad_a_withRespectTo_z = 1
                #print("here")
            else:
                grad_a_withRespectTo_z = 0
            for c in range(0,len(w[l][r])):
                    w_grad[l][r][c] = (a[l][c] * grad_a_withRespectTo_z * grad_withRespectTo_a[r])
            b_grad[l][r] = (grad_a_withRespectTo_z * grad_withRespectTo_a[r]) #* grad_withRespectTo_a[r]

    return (w_grad,b_grad)



def test(datas):
    correct = 0
    cost = 0
    for d in datas:
        outp = eval(d[0:input_size])
        if outp == d[input_size:]:
            correct += 1
        for i,y in enumerate(d[input_size:]):
            cost += (outp[i] - y)**2
    costs.append(cost/len(datas))
    print(len(datas))
    print("test results: accuracy:",correct/len(datas),"cost:",cost/len(datas))

def train(batch,learning_rate):
    global weights
    global biases
    #print(sum(weights[1][0]))

    #global data
    b_grad = [[0 for r in range(0,len(biases[l]))] for l in range(0,len(biases))]
    w_grad = [[[0 for c in range(0,len(weights[l][r]))] for r in range(0,len(weights[l]))] for l in range(0,len(weights))]
    for d in batch:
        activations = forward_propagation(d[:input_size])
        #print(eval(d[:input_size]))
        (w_grad_next,b_grad_next)= back_propagation(weights,biases,d[input_size:],activations)
        #print(sum(weights[0][0]),sum(biases[0]),sum(d[input_size:]),sum(activations[0]))
        b_grad = [[b_grad[l][r]+b_grad_next[l][r] for r in range(0, len(biases[l]))] for l in range(0, len(biases))]
        w_grad = [[[w_grad[l][r][c]+w_grad_next[l][r][c] for c in range(0, len(weights[l][r]))] for r in range(0, len(weights[l]))] for l in range(0, len(weights))]
    #print(w_grad)
    #print(weights)
    biases_t = [[biases[l][r]-((b_grad[l][r]/len(batch))*learning_rate) for r in range(0, len(biases[l]))] for l in range(0, len(biases))]
    weights_t = [[[weights[l][r][c]-((w_grad[l][r][c]/len(batch))*learning_rate) for c in range(0, len(weights[l][r]))] for r in range(0, len(weights[l]))] for l in
              range(0, len(weights))]
    #test(batch)
    biases = biases_t
    weights = weights_t






#data = {}
data = []

'''for txt in ["000101000110001001011011011","010010100101110010110110101","111011001010110010001011010"]:
    for i in range(0,1000):
        inp = list(txt)
        for r in range(0, len(inp)):
            inp[r] = float(inp[r])
        data.append(inp)
random.shuffle(data)'''
for i in range(0,64):
    for j in range(0,64):
        for k in range(0,64):
            inp = list("{0:06b}".format(i) + "{0:06b}".format(j) + "{0:06b}".format(k))
            for i in range(0,len(inp)):
                inp[i] = float(inp[i])
            if j^k < i:
                outp = list("100"+"{0:06b}".format(j ^ k))
                for i in range(0, len(outp)):
                    outp[i] = float(outp[i])
                data.append(inp+outp)
            elif i^k < j:
                outp = list("010" + "{0:06b}".format(i ^ k))
                for i in range(0, len(outp)):
                    outp[i] = float(outp[i])
                data.append(inp + outp)
            elif i^j < k:
                outp = list("001" + "{0:06b}".format(i ^ j))
                for i in range(0, len(outp)):
                    outp[i] = float(outp[i])
                data.append(inp + outp)
random.shuffle(data)
data = data[:len(data)//20]



training_data = data[:len(data)//2]
test_data = data[len(data)//2:]

test(test_data)
for i in range(1,21):
    random.shuffle(training_data)
    training_chunks = [training_data[x:x + 2000] for x in range(0, len(training_data), 2000)]
    print("************ epoc"+str(i)+" ***************")
    num = 0
    print("progress: ")
    for batch in training_chunks:
        if ((num/len(training_chunks))) % 5 == 0:
            print("#", end="", flush=True)
        num += 1
        train(batch,0.1)
    print("epoc",i,"is done :)")
    test(test_data)

    with open("costs5.txt","wb") as fp:
        pickle.dump(costs, fp)
    with open("weights2.txt", "wb") as fp:   #Pickling
        pickle.dump(weights, fp)
    with open("biases2.txt", "wb") as fp:   #Pickling
        pickle.dump(biases, fp)
#with open("test.txt", "rb") as fp:   # Unpickling
#    b = pickle.load(fp)

