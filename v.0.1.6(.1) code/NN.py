from numba import jit, njit
import numpy as np
import random
import PIL.Image as Image
import timeit

# The following is based on Tom M. Mitchells Machine Learning chapter about neural networks particularly p. 98, although
# the matrix version is something I've tried to derive so it may be erronous, look carefully through before use.
# 0.1.3: Added very basic adaptive adjustment of learning rate, only down
# 0.1.4: Images are now not thumbnails but just scaled.
# 0.1.4: Added "proper" gradient descent mode
# 0.1.4: Now remembers momentum across epochs.
# 0.1.4: Increased network size - hidden layer 1 doubled, hidden layer 2 octoupled. - reverted
# 0.1.4: Hidden layer 1 and 2 node count halved
# 0.1.4: Also had a look at scipy for using sparse matricies. Seemed to have rather poor support of elementwise operations...
# 0.1.5: Turns out Pillows resize function returns a resized image instead of resizing in place. This means the training data only
#        saved the upper left corner of each image instead of the entire image, explaining why the network was unable to learn... as there was little to learn.
# 0.1.5.1: Reverted hidden network size to ImageSize.
# 0.1.6: Added individual learning rates to weights, removed gradient descent - REVERTED
# 0.1.6.1: Re-implemented global adaptive learning rate


Version = "0.1.6.2"
LearningRate = 0.1
ImageSize = 110
TargetEpochs = 1000000
output_path = "D:\\jonod\\Pictures\\BackpropExperiment\\Outputs" # path to output folder, also folder with training data in it.
mode = "Stochastic_Gradient_Descent" # only stochastic gradient descent is currently implemented

# Consider making a sparse matrix at some point to improve efficiency?
# Maybe a "reverse" convolutional neural network?

# Slightly misleading names, are actually weight matrices, consider refactoring
# Init layers
nodes_input = np.zeros((ImageSize, 8), np.float64)  # from 8 input nodes to 8 nodes
nodes_hidden1 = np.zeros((ImageSize, ImageSize), np.float64)  # 6 nodes to 4 nodes
nodes_hidden2 = np.zeros((ImageSize*ImageSize, ImageSize), np.float64) # 6 nodes to ImageSize*ImageSize*4 output nodes (ImageSize x ImageSize Black and White)

# This one does not work with @njit, the issue seems to be related to the dot product always returning the same value. The same happens when all values are not 0
def calculate_output(in1, in2, in3, in4, in5, in6, in7, in8):
    #print(np.asarray([[in1], [in2], [in3], [in4], [in5], [in6], [in7], [in8]], np.float64))
    #print(sigmoid(np.asarray([[in1], [in2], [in3], [in4], [in5], [in6], [in7], [in8]], np.float64)))
    #print(np.dot(nodes_input, sigmoid(np.asarray([[in1], [in2], [in3], [in4], [in5], [in6], [in7], [in8]], np.float64)))) # Error seems to be here, this always is 0? - Issue is with njit
    #print(sigmoid(np.dot(nodes_input, sigmoid(np.asarray([[in1], [in2], [in3], [in4], [in5], [in6], [in7], [in8]], np.float64)))))
    hidden1_output = sigmoid(np.dot(nodes_input, sigmoid(np.asarray([[in1], [in2], [in3], [in4], [in5], [in6], [in7], [in8]], np.float64))))  # 4 x 3 * 3 x 1 => 4 x 1
    hidden2_output = sigmoid(np.dot(nodes_hidden1, hidden1_output))  # 3 x 4 * 4 x 1 => 3 x 1
    output = sigmoid(np.dot(nodes_hidden2, hidden2_output))  # ImageSize*ImageSize*4 x 3 * 3 x 1 = ImageSize*ImageSize*4 x 1
    return hidden1_output, hidden2_output, output

@njit
def calc_output_layer_error(network_output, training_value):
    return np.multiply(np.multiply(network_output, (1-network_output)), np.subtract(training_value, network_output))

@njit
def calc_hidden_layer_error(hiddenOutput, hiddenWeights, outputError):
    # Seems to return only the one same value
    return np.multiply(np.multiply(hiddenOutput, np.subtract(np.ones(hiddenOutput.shape, np.float64), hiddenOutput)), np.dot(np.transpose(hiddenWeights), outputError))


@njit
def sigmoid(a):
    return 1.0/(1.0 + np.power(np.e, -a)) # 0.5 is bias so it is not centered around 0

@njit
def update(error, the_input, learning_rate, last_update):
    # W/momentum, calculated as in Tom Mitchell book
    return learning_rate * (error * np.transpose(the_input)) + learning_rate * last_update  # I think the_input is just the output of the previous layer in terms of what matrix to input, but I haven't checked so may be error source



def save_image(outputValues, filepath):
    img = Image.new('RGB', (ImageSize, ImageSize))
    for x in range(0, ImageSize):
        for y in range(0, ImageSize):
            img.putpixel((x, y), (int(outputValues[(y + x * ImageSize)][0]*255.0),
                                  int(float(outputValues[(y + x * ImageSize)][0]) * 255.0),
                                  int(float(outputValues[(y + x * ImageSize)][0]) * 255.0)))
    img.save(filepath)

# Used to test that input images are output correctly
def save_image_input_test(outputValues, filepath):
    img = Image.new('RGB', (ImageSize, ImageSize))
    for x in range(0, ImageSize):
        for y in range(0, ImageSize):
            img.putpixel((x, y), (int(float(outputValues[(y + x * ImageSize)]) * 255.0),
                                  int(float(outputValues[(y + x * ImageSize)]) * 255.0),
                                  int(float(outputValues[(y + x * ImageSize)]) * 255.0)))
    img.save(filepath)

# Start benchmark timer
start_time = timeit.default_timer()

# Load in training examples:
# Correct output format r1 g1 b1 a1 r2 g2 b2 a2 ...
# training file will be: seed1|seed2|seed3|r1|g1|b1|a1|r2|...!
training_file = open(output_path + "\\" + Version + "_training.txt", 'r')
text = training_file.read()
training_file.close()
process_step_zero = text.split("!")
training_data = []
for y in range(len(process_step_zero)): # Potential optimization
    training_data.append([x for x in process_step_zero[y].split("|")])

# Test that input is stored and shown properly
for y in training_data:
    save_image_input_test(y[4:], output_path + "\\" + Version + y[0] + "_test.png",)
epoch = 0
endNext = False
#last_average_error = float("inf")
average_error = 0
# Do training:
if mode == "Stochastic_Gradient_Descent": # This may seem like unneseccary code duplication but putting the if statement outside of the loop is marginally more efficient computationally
    last_nodes_input = np.zeros((ImageSize, 8), np.float64)
    last_nodes_hidden1 = np.zeros((ImageSize, ImageSize), np.float64)
    last_nodes_hidden2 = np.zeros((ImageSize * ImageSize, ImageSize), np.float64)
    while True:
        average_error = 0
        random.shuffle(training_data)  # Randomize order of training data
        for num in range(0, len(training_data)):
            imageName = training_data[num][0]  # Used only for filenames
            in1, in2, in3, in4, in5, in6, in7, in8 = float(training_data[num][1]), float(training_data[num][2]), \
                                                     float(training_data[num][3]), float(training_data[num][4]), \
                                                     float(training_data[num][5]), float(training_data[num][6]), \
                                                     float(training_data[num][7]), float(training_data[num][8]),
            target_output = np.asarray(training_data[num][9:],
                                       np.float64)  # Potential errors: Wrong indexing, wrong matrix format, strings not floats
            target_output.shape = (ImageSize * ImageSize, 1)
            hidden1_output, hidden2_output, output = calculate_output(in1, in2, in3, in4, in5, in6, in7, in8)
            output_error = calc_output_layer_error(output, target_output)
            hidden2_error = calc_hidden_layer_error(hidden2_output, nodes_hidden2, output_error)
            hidden1_error = calc_hidden_layer_error(hidden1_output, nodes_hidden1, hidden2_error)
            average_error += sum(np.power(np.subtract(output, target_output), 2)) * 1 / 2
            update_nodes_input = update(hidden1_error, np.asarray([in1, in2, in3, in4, in5, in6, in7, in8], np.float64),
                                        LearningRate, last_nodes_input)
            update_nodes_hidden1 = update(hidden2_error, hidden1_output, LearningRate, last_nodes_hidden1)
            update_nodes_hidden2 = update(output_error, hidden2_output, LearningRate, last_nodes_hidden2)
            nodes_input = nodes_input + update_nodes_input
            nodes_hidden1 = nodes_hidden1 + update_nodes_hidden1
            nodes_hidden2 = nodes_hidden2 + update_nodes_hidden2
            # Momentum
            last_nodes_input = update_nodes_input
            last_nodes_hidden1 = update_nodes_hidden1
            last_nodes_hidden2 = update_nodes_hidden2
            if endNext:
                save_image(output,
                           output_path + "\\" + Version + "_Filename_" + imageName + "_Epoch_" + str(epoch) + ".png")
        if epoch % 100 == 0:
            #if average_error > last_average_error:
            #    LearningRate *= 0.5
            #last_average_error = average_error
            print(epoch, average_error / len(training_data))
        epoch += 1

        if endNext:
            for num in range(0,
                             2):  # This feels super dumb but it is the first and best solution I thought of so it'll do for prototyping
                for num2 in range(0, 2):
                    for num3 in range(0, 2):
                        for num4 in range(0, 2):
                            for num5 in range(0, 2):
                                for num6 in range(0, 2):
                                    for num7 in range(0, 2):
                                        for num8 in range(0,
                                                          2):  # Generate every possible permutation of the eight inputs
                                            hidden1_output, hidden2_output, output = calculate_output(num, num2, num3,
                                                                                                      num4, num5, num6,
                                                                                                      num7, num8)
                                            save_image(output, output_path + "\\" + Version + "_Output_" + str(num)
                                                       + str(num2) + str(num3) + str(num4) +
                                                       str(num5) + str(num6) + str(num7) + str(num8) + ".png")
            break
        else:
            endNext = average_error / len(training_data) < 1 or epoch > TargetEpochs
