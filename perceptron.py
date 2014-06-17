import numpy as np

# calculate a random number where:  a <= rand < b
def rand(a, b):	return (b-a)*random.random() + a

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):	return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):	return 1.0 - y**2

class perceptron():
	def __init__(self):
		self.train_folder = ""
		self.input_file = ""
		self.nb_hidden = 1
		self.input_w = []
		self.input_o = []
		self.output_w = []
		self.output_o = []
		self.hidden_w = []
		self.hidden_o = []
		self.hidden_size = 0
		self.h = 0
		self.w = 0

	def set_input_size(self, h, w):
		"""Set the network size as the size of the input picture, 1 neuron per pixel"""
		self.h = h
		self.w = w

	def reset(self):
		self.input_w = np.zeros((h,w))
		self.hidden_w = np.zeros(h*w)
		self.output_w = [0] #one boolean output

	def train(self, nb_steps, error_treshold):
		for i in xrange(nb_steps):
			for neuron in self.input_row:


	nbS set_input(self, filename = ""):
		self.input_file = filename

	def run(self):
		pass
