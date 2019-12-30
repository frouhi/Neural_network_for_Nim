import pickle
'''
This code uses the saved neural net to solve the nim game.
Original output has 9 bits. Each bit of the output of NN is rounded to closest integer.
The three most significant bits tell us which pile should change.
The pile that has a value of 1 must change (the order is from left to right).
The 6 least significant bits tell us what the selected pile should become (in binary).
The program will interpret the output and print the result in a readable format.
'''
with open("biases.txt", "rb") as fp:   # Unpickling
    biases = pickle.load(fp)

with open("weights.txt", "rb") as fp:   # Unpickling
    weights = pickle.load(fp)


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
            outp.append(max(0,temp))
        activations.append(outp)
        inp = outp.copy()
    return activations


# This function evaluates the neural net iusing the forward propagation.
def eval(x):
    activations = forward_propagation(x)
    return activations[len(activations)-1]


# This function gets the three piles and calculates the output in binary.
def calc(a,b,c):
    inp = list("{0:06b}".format(a) + "{0:06b}".format(b) + "{0:06b}".format(c))
    for i in range(0, len(inp)):
        inp[i] = float(inp[i])
    print("NN output",eval(inp))
    outp = []
    if b ^ c < a:
        outp = list("100" + "{0:06b}".format(b^c))
        for i in range(0, len(outp)):
            outp[i] = float(outp[i])
        # data[inp] = outp
        # keys.append(inp)
        print("expected output (in binary)",outp)
    elif a ^ c < b:
        outp = list("010" + "{0:06b}".format(a^c))
        for i in range(0, len(outp)):
            outp[i] = float(outp[i])
        # data[inp] = outp
        # keys.append(inp)
        print("expected output (in binary)",outp)
    elif a ^ b < c:
        outp = list("001" + "{0:06b}".format(a ^ b))
        for i in range(0, len(outp)):
            outp[i] = float(outp[i])
        # data[inp] = outp
        # keys.append(inp)
        print("expected output (in binary)",outp)
    else:
        print("No solutions found")
    return outp

# converts the 9 bit binary result to a more readable format
def print_result(outp):
    if outp is not []:
        if outp[0] == 1:
            print("pile #1 should become: ", end="")
        elif outp[1] == 1:
            print("pile #2 should become: ", end="")
        elif outp[2] == 1:
            print("pile #3 should become: ", end="")
        res = int("".join(str(int(x)) for x in outp[3:]), 2)
        print(res)


outp = calc(1,2,6)
print("Example: pile #1 = 1, pile #2 = 2, pile #3 = 6")
print_result(outp)
a = int(input("pile #1 as integer (0 to 64)"))
b = int(input("pile #2 as integer (0 to 64)"))
c = int(input("pile #3 as integer (0 to 64)"))
outp = calc(a,b,c)
print_result(outp)
