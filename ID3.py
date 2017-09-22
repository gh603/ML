import math

################### Method to read dataset ##########################
# Method to read dataset
def readData(path):         
    with open(path) as file: lines = file.readlines()
    data = []
    for line in lines: data.append(line.strip().split(','))
    # The first line of data is header, and the following lines are data.
    header = data[0]
    data = data[1:]
    return header, data

# Method to get attributes from the input dataset, including # of attributes, # of instances, 
def getAttributes(data): 
	return len(data[0]), len(data)

################ Methods to select the attribute with most Information Gain ###################################
# Method to get counts of each case: 
# Let x denotes explanatory variable, and y denotes resposne variable.
# We need to get the counts of (X, Y): (True, True), (True, False), (False, True), (False, False); 
def getCounts(data): 
    if(data == None or len(data) == 0): 
        return [[0,0,0,0]]
    nrow = len(data)
    ncol = len(data[0])
    counts = [[0 for i in range(4)] for i in range(ncol)]
    
    for row in range(nrow): 
        for col in range(ncol): 
            num = data[row][col]
            if(num == '1'): 
                if(data[row][ncol - 1] == '0'): counts[col][1] += 1
                else: counts[col][0] += 1
            else: 
                if(data[row][ncol - 1] == '1'): counts[col][2] += 1
                else: counts[col][3] += 1
    return counts 

# Given data, calculate the entropy of the response variable.
def calEntropy(numT, numF): 
    total = numT + numF
    if(total == 0): return 0
    pT = percentage(numT, total)
    pF = percentage(numF, total)
    return -1 * pT * log2(pT) - pF * log2(pF)

# Calculate the log2 value of a given number num
def log2(num): 
	if(num == 0): return 0
	return math.log(num, 2)

# Calculate the percentage of True given the number of true and the number of total
def percentage(numT, total): 
    return float(numT) / total

# Get the index of atttribute that maximizes the Information Gain from a counts matrix
def maximizedIG(counts):
    numAttr = len(counts)
    maxId = 0
    minH = 1
    for i in range(numAttr - 1):
    	temp = calH(counts, i)
    	if(temp < minH): 
    		maxId, minH = i, temp
    return maxId

# Caldulate the H(Y|X) of of the i-th variable
def calH(counts, i): 
    count = counts[i]
    numT = count[0] + count[1]
    numF = count[2] + count[3]
    if(numT == 0 or numF == 0): return 0
    return calEntropy(count[0], count[1]) * percentage(numT, numT + numF) + calEntropy(count[2], count[3]) * percentage(numF, numT + numF)

# Divide the original dataset into two subset according to the value of an attribute i
def divMatrix(data, maxId): 
    dataOne = []
    dataTwo = []
    for i in range(len(data)):
        row = data[i]
        val = row[maxId]
        del row[maxId]
        if(val =='0'): dataOne.append(row) 
        else: dataTwo.append(row)
    return dataOne, dataTwo


################ Methods to build up trees ###################################
# To implement a tree, we need a class of TreeNode
class TreeNode(object): 
    def __init__(self, attrName, attrVal, nodeData, leftChild, rightChild, level, outputClass):
        self.attrName = attrName
        self.attrVal = attrVal
        self.nodeData = nodeData
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.level = level
        self.outputClass = outputClass

    def toString(self): 
        if(self.level == 0): return ""
        s = ''
        if(self.level > 0): 
            for i in range(self.level - 1): 
                s = s + '|'
        s = s + self.attrName + " = " + self.attrVal + ":"
        if(self.leftChild == None and self.rightChild == None): 
            s = s + self.outputClass
        return s


# Method to build up a Decision tree 
def buildTree(data, header, level, attrName, attrVal): 	
    print 'calling buildeTree(data, header, ', level, ',', attrVal,')'
    #Find the attribute with most information gain
    counts = getCounts(data)
    #Stop building tree if reaching the leaf: only single class included or no more attributes to divide
    yDist = counts[len(counts) - 1]
    dataEntropy = calEntropy(yDist[0] + yDist[1], yDist[2] + yDist[3])

    if(dataEntropy == 0 or dataEntropy == -0 or len(counts) == 1): 
        outputClass = "0"
        if(yDist[2] + yDist[3] == 0): 
            outputClass = "1"
        return TreeNode(attrName, attrVal, data, None, None, level, outputClass); 

    maxId = maximizedIG(counts)
    #Divide the data into two sub data according to value of the attribute with most information gain
    dataOne, dataTwo = divMatrix(data, maxId)

    #Two sub data are used as the left child and right child of node
    left = buildTree(dataOne, header, level + 1, header[maxId], "0")
    right = buildTree(dataTwo, header, level + 1, header[maxId], "1")

    #Return the node
    node = TreeNode(attrName, attrVal, data, left, right, level, None)
    return node

# Method to print a Decision Tree
def printTree(node): 
    if(node == None): return 
    print node.toString()
    printTree(node.leftChild)
    printTree(node.rightChild)



##################################### Main ###################################

#Read the Training set
trainSet_Path = '/Users/gh603/career/UTD/Course/Fall_2017/ML/HW/HW2/data_sets1/training_set.csv'
header, data = readData(trainSet_Path)
ncol, nrow = getAttributes(data)
print "nrow = ", nrow
print "ncol = ", ncol

# Test methods for selecting the attribute with most information gain
counts = getCounts(data)
print("counts = ", counts)

#Test for finding the attribute with most information gain
maxId = maximizedIG(counts)
print("maxId = ", maxId)

#Test for split dataset
dataOne, dataTwo = divMatrix(data, maxId)
print(len(dataOne))
print(len(dataTwo))

node = buildTree(data, header, 0, None, None)
#print(node.nodeData)
printTree(node)
