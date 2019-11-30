import math

#Math
def sigmoid(value):
    return 1/(1+(math.e**(-value)))
def costOnNode(nodeValue, shouldBe):
    return (nodeValue - shouldBe)**2

def cost(outputNodes, shouldBe):
    totalCost = 0
    onNode = 0
    for currentNode in outputNodes:
        totalCost += costOnNode(currentNode.value, shouldBe[onNode])
        onNode += 1
    return totalCost

#Classes
class node:
    value = 0
    weights = []
    bias = 0
    def compute(self, lastNodes):
        additionOfNodes = 0
        onNode = 0
        for currentNode in lastNodes:
            additionOfNodes += currentNode.value*self.weights[onNode]
            onNode += 1
        return sigmoid(additionOfNodes+self.bias)

class layer:
    nodes = []
    def compute(self, lastLayer):
        for currentNode in self.nodes:
            currentNode.value = currentNode.compute(lastLayer.nodes)

class network:
    layers = []
    def compute(self):
        onLayer = 0
        for currentLayer in self.layers:
            if onLayer != 0:
                currentLayer.compute(self.layers[onLayer-1])
            onLayer += 1

#Backpropogation
def backProp(network, correctOutput, learningRateW, learningRateB):
    print("#Instance of Backprop#")
    network.compute()
    finalLayerNo = len(network.layers)-1
    print("#Total Cost [" +str(cost(network.layers[finalLayerNo].nodes, correctOutput))+ "]#")
    allNodesInNetwork = []
    for currentLayer in network.layers:
        for currentNode in currentLayer.nodes:
            allNodesInNetwork.append(currentNode)
    for currentNode in allNodesInNetwork:
        counter = 0
        #weights
        for weight in currentNode.weights:
            costBefore = cost(network.layers[finalLayerNo].nodes, correctOutput)
            currentNode.weights[counter] += learningRateW
            network.compute()
            costAfter = cost(network.layers[finalLayerNo].nodes, correctOutput)
            currentNode.weights[counter] -= learningRateW
            if costAfter<costBefore:
                currentNode.weights[counter] += learningRateW#*((costAfter-costBefore)**2)
            if costAfter>costBefore:
                currentNode.weights[counter] -= learningRateW#*((costAfter-costBefore)**2)
            counter += 1
        #bias
        costBefore = cost(network.layers[finalLayerNo].nodes, correctOutput)
        currentNode.bias += learningRateB
        network.compute()
        costAfter = cost(network.layers[finalLayerNo].nodes, correctOutput)
        currentNode.bias -= learningRateB
        if costAfter<costBefore:
            currentNode.bias += learningRateB#*((costAfter-costBefore)**2)
        if costAfter>costBefore:
            currentNode.bias -= learningRateB#*((costAfter-costBefore)**2)
        
#Tests
def test2():
    myNetwork = network()
    myNetwork.layers = [layer(),layer()]
    myNetwork.layers[0].nodes = [node(),node(),node()]
    myNetwork.layers[0].nodes[0].value = 1
    myNetwork.layers[0].nodes[1].value = 1
    myNetwork.layers[0].nodes[2].value = 0

    myNetwork.layers[1].nodes = [node(),node(),node()]
    myNetwork.layers[1].nodes[0].weights = [1,1,1]
    myNetwork.layers[1].nodes[1].weights = [0,0,1]
    myNetwork.layers[1].nodes[2].weights = [0,0,1]
    myNetwork.layers[1].nodes[1].bias = 100

    myNetwork.compute()

    for i in myNetwork.layers[1].nodes:
        print(str(i.value) + " - " + str(costOnNode(i.value, 1)))
    print()
    print(cost(myNetwork.layers[1].nodes,[1,1,1]))
    while(cost(myNetwork.layers[1].nodes,[1,1,1])>0.00001):
        backProp(myNetwork, [1,1,1] ,0.1 , 0.01)
        for i in myNetwork.layers[1].nodes:
            print(str(i.value) + " - " + str(costOnNode(i.value, 1)))
        print()
        print(cost(myNetwork.layers[1].nodes,[1,1,1]))
        
def test():
    myNetwork = network()
    myNetwork.layers = [layer(),layer(),layer(),layer()]
    myNetwork.layers[0].nodes = [node(),node(),node()]
    myNetwork.layers[0].nodes[0].value = 1
    myNetwork.layers[0].nodes[1].value = 1
    myNetwork.layers[0].nodes[2].value = 0

    myNetwork.layers[1].nodes = [node(),node(),node()]
    myNetwork.layers[1].nodes[0].weights = [1,1,1]
    myNetwork.layers[1].nodes[1].weights = [0,0,1]
    myNetwork.layers[1].nodes[2].weights = [0,0,1]

    myNetwork.layers[2].nodes = [node(),node(),node()]
    myNetwork.layers[2].nodes[0].weights = [1,0,1]
    myNetwork.layers[2].nodes[1].weights = [1,0,1]
    myNetwork.layers[2].nodes[2].weights = [0,2,0]
    
    myNetwork.layers[3].nodes = [node(),node(),node(),node(),node()]
    myNetwork.layers[3].nodes[0].weights = [1,5,6]
    myNetwork.layers[3].nodes[1].weights = [2,5,3]
    myNetwork.layers[3].nodes[2].weights = [0,2,1]
    myNetwork.layers[3].nodes[3].weights = [1,2,1]
    myNetwork.layers[3].nodes[4].weights = [5,2,1]
    
    myNetwork.compute()

    meantToBe = [0,0.2, 0.4, 0.6, 0.8]
    while(cost(myNetwork.layers[3].nodes, meantToBe)>0.0001):
        backProp(myNetwork, meantToBe ,0.1 , 0.01)
        a = 0
        for i in myNetwork.layers[3].nodes:
            print(str(i.value) + " - " + str(costOnNode(i.value, meantToBe[a])))
            a+=1
        print()
        print(cost(myNetwork.layers[1].nodes,[1,1,1]))
    
