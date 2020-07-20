# MNIST HANDWRITTEN DIGITS RECOGNITION

## Data fetch

Data is being fetched from the official website of the MNIST dataset. The 
downloader checks whether the download worked, if not, it just passes the 
function - the actuall NN building will probably just fail then.

Data itself is downloaded, normalized and vectorized into the shape that NN
can work with.

The details why it is done so etc. will be discussed in the report.

## Building NN

The actual NN stuff. It is not optimal in any shape or form but that was not 
the point of this project. The point was to understand underlying concepts of 
NN without usage of third party libs such as Tensorflow, Lasagne, Scilearn etc.

The actual network accepts data in form of (x, y) where x is the actual input 
image, reshaped into (784, 1) a y is the label. which is column matrix with
ten digits composed of zeroes and one at the correct position.

The activation function that has been used is CCE (Categorical cross entropy) 
and it's derivative with respect to the activation function, the Sigmoid function.
It has been learned with SGD (Stochastic gradient descent). Underlaying maths
will be explained in later submitted report.

Logging is done in percents for each epoch, with the format being :
    LRxxxNExxxBSxxx where:
        LR - Learning rate
        NE - Number of epochs that the NN was learning for
        BS - Size of minibatches
        
The NN is in the end pickled so we don't have to re-learn the NN everytime we 
want to use it.

The network averages accuracy of 89% with LR05NE9BS50.
        
