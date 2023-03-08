import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
	return 1 / (1 + math.e ** (-1 * x))


def sigmoid_derivative(x):
	a = sigmoid(x)
	a = np.reshape(a, (-1,1))
	b = 1 - sigmoid(x)
	b = np.reshape(b, (-1,1))
	b = np.transpose(b)
	product = np.matmul(a, b)
	return np.matmul(a,b)

split_ratio = 0.7
eta = 0.3
epochs = 50

data = pd.read_excel('data.xlsx', header = None)
data = np.array(data)
min = np.min(data)
max = np.max(data)

#Normalize 
for i in range(np.shape(data)[0]):
	for j in range(np.shape(data)[1]):
		data[i, j] = (data[i, j] - min) / (max - min)

split_line_number = int(np.shape(data)[0] * split_ratio)
x_train = data[:split_line_number, :4]
x_test = data[split_line_number:, :4]
y_train = data[:split_line_number, 4]
y_test = data[split_line_number:, 4]

input_dimention = np.shape(x_train)[1]
l1_neurons = 6
l2_neurons = 5
l3_neurons = 1

w1 = np.random.uniform(low = -1, high = 1, size = (input_dimention, l1_neurons))
w2 = np.random.uniform(low = -1, high = 1, size = (l1_neurons, l2_neurons))
w3 = np.random.uniform(low = -1, high = 1, size = (l2_neurons, l3_neurons))

# x = input("first Press any key + Enter:")
MSE_train = []
MSE_test = []

for i in range(epochs):
	sqr_err_epoch_train = []
	sqr_err_epoch_test = []

	output_train = []
	output_test = []
	# Here Batch size is equal to 1
	for j in range(np.shape(x_train)[0]):
		#Feedforward
		#Layer1
		#From previous lines we have 
		#(1 * input_dimention)(input_dimention * l1_neurons)
		net1 = np.matmul(x_train[j], w1) 
		o1 = sigmoid(net1)
		o1 = np.reshape(o1, (-1,1))
		#Layer2
		net2 = np.matmul(np.transpose(o1), w2)
		o2 = sigmoid(net2)
		o2 = np.reshape(o2, (-1,1))
		############################################
		#A FUCKIN' HINT:
		# if you got confused in the process just look at w1 w2 w3's dimentions
		#FUCKIN' A
		#############################################
		#Layer3
		net3 = np.matmul(np.transpose(o2), w3)
		o3 = net3

		output_train.append(o3[0])
		# x = input("sec Press any key + Enter:")
		#Error
		err = y_train[j] - o3[0]
		sqr_err_epoch_train.append(err ** 2)

		#Back Propagation
		f1_derivative = sigmoid_derivative(net1)
		f2_derivative = sigmoid_derivative(net2)
		w3_f2_derivative = np.matmul(f2_derivative, w3)

		
		# Update w1
		w2_f1_derivative = np.matmul(f1_derivative, w2)
		w2_f1_derivative_w3_f2_derivative = np.matmul(w2_f1_derivative, w3_f2_derivative)
		w2_f1_derivative_w3_f2_derivative_x = np.matmul(
			 w2_f1_derivative_w3_f2_derivative, np.transpose(np.reshape(x_train[j], (-1,1))))
		w1 = np.subtract(w1, (eta * err * (-1)* (1) * np.transpose(w2_f1_derivative_w3_f2_derivative_x)))
		
		# For Debugging purposes I got screwed
		# print(f"w2_f1_derivative: {np.shape(w2_f1_derivative)}")
		# print(np.shape(np.reshape(x_train[j], (-1, 1))))
		# print(f"w3_f2_derivative: {np.shape(w3_f2_derivative)}")
		# print(f"x_train[j]: {np.shape(x_train[j])}")
		# print(f"np.reshape(x_train[j], (-1,1)): {np.shape(np.reshape(x_train[j], (-1,1)))}")
		# print(f"w2_f1_derivative_w3_f2_derivative: {np.shape(w2_f1_derivative_w3_f2_derivative)}")
		# print(np.shape(np.transpose(np.reshape(x_train[j], (-1,1)))))
		# print(np.shape(w2_f1_derivative_w3_f2_derivative_x))
		
		# update w2
		# o1: (6, 1)
		# w3_f2_derivative: (5, 1)
		w3_f2_derivative_o1 = np.matmul(o1, np.transpose(w3_f2_derivative))
		# w3_f2_derivative_o1:(6, 5)
		w2 = np.subtract(w2, (eta * err * (-1) * (1) * w3_f2_derivative_o1))

		# update w3
		w3 = np.subtract(w3, (eta * err * (-1) * o2))

	mse_epoch_train = 0.5 * (sum(sqr_err_epoch_train)) / np.shape(x_train)[0]
	MSE_train.append(mse_epoch_train)


	for j in range(np.shape(x_test)[0]):
		# Feedforward
		# Layer 1
		##### What is np.reshape(,(-1,1)) good for????? 
		net1 = np.matmul(x_test[j], w1)
		o1 = sigmoid(net1)
		o1 = np.reshape(o1, (-1,1))
		# o1: (6, 1)


		# Layer 2
		net2 = np.matmul(np.transpose(o1), w2)
		o2 = sigmoid(net2)
		o2 = np.reshape(o2, (-1, 1))

		# Layer 3
		net3 = np.matmul(np.transpose(o2), w3)
		o3 = net3

		output_test.append(o3[0])

		#Error
		err = y_test[j] - o3[0]
		sqr_err_epoch_test.append(err ** 2)

	mse_epoch_test = 0.5 * (sum(sqr_err_epoch_test)) / np.shape(x_test)[0]
	MSE_test.append(mse_epoch_test)

	#Plot fits

	#Train
	m_train, b_train = np.polyfit(y_train, output_train, 1)
	# Test
	m_test, b_test = np.polyfit(y_test, output_test, 1)


	# plot
	fig, axs = plt.subplots(3, 2)
	axs[0, 0].plot(MSE_train, 'b')
	axs[0, 0].set_title("MSE Train")
	axs[0, 1].plot(MSE_test, 'r')
	axs[0, 1].set_title('MSE test')

	axs[1, 0].plot(y_train, 'b')
	axs[1, 0].plot(output_train, 'r')
	axs[1, 0].set_title("Output Train")
	axs[1, 1].plot(y_test, 'b')
	axs[1, 1].plot(output_test, 'r')
	axs[1, 1].set_title('Output Test')

	axs[2, 0].plot(y_train, output_train, 'b*')
	axs[2, 0].plot(y_train, m_train * y_train + b_train, 'r')
	axs[2, 0].set_title("Regression Train")
	axs[2, 1].plot(y_test, output_test, 'b*')
	axs[2, 1].plot(y_test, m_test * y_test + b_test, 'r')
	axs[2, 1].set_title("Resgression Test")
	if i == (epochs - 1):
		plt.savefig("Results.jpg")
	##### plt.show vs plt.plot?
	plt.plot()
	plt.pause(0.1)
	plt.close(fig)
