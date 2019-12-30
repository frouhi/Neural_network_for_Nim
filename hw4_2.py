import pickle

with open("biases.txt", "rb") as fp:   # Unpickling
    biases = pickle.load(fp)

with open("weights.txt", "rb") as fp:   # Unpickling
    weights = pickle.load(fp)

def forward_propagation(x):
    #global weights
    #global biases
    activations = [x]
    inp = x.copy()
    for l in range(0,len(weights)):
        outp = []
        for r in range(0,len(weights[l])):
            temp = biases[l][r]
            for c in range(0,len(weights[l][r])):
                temp += (weights[l][r][c] * inp[c])
            #if temp<0:
            #    temp = 0

            outp.append(max(0,temp))
        activations.append(outp)
        inp = outp.copy()
    #print(activations[len(activations)-1])
    return activations

def eval(x):
    activations = forward_propagation(x)
    return activations[len(activations)-1]

def calc(a,b,c):
    inp = list("{0:06b}".format(a) + "{0:06b}".format(b) + "{0:06b}".format(c))
    for i in range(0, len(inp)):
        inp[i] = float(inp[i])
    print("NN output",eval(inp))
    if b ^ c < a:
        outp = list("100" + "{0:06b}".format(b^c))
        for i in range(0, len(outp)):
            outp[i] = float(outp[i])
        # data[inp] = outp
        # keys.append(inp)
        print("expected output",outp)
    elif a ^ c < b:
        outp = list("010" + "{0:06b}".format(a^c))
        for i in range(0, len(outp)):
            outp[i] = float(outp[i])
        # data[inp] = outp
        # keys.append(inp)
        print("expected output",outp)
    elif a ^ b < c:
        outp = list("001" + "{0:06b}".format(a ^ b))
        for i in range(0, len(outp)):
            outp[i] = float(outp[i])
        # data[inp] = outp
        # keys.append(inp)
        print("expected output",outp)