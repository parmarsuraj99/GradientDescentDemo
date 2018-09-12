import numpy as np

def calculate_error(x, y, m, b):
  error=0
  error = np.sum((m*x+b - y)**2)
  error/=len(x)
  return error

def update_weights(m_current, b_current, x, y, learning_rate):
  m_gradient=0
  b_gradient=0
  n=len(x)
  b_gradient=np.sum(-(1/n)*(y-(m_current*x+b_current)))
  m_gradient=np.sum(-(1/n)*(y-(m_current*x+b_current))*x)
  b_new=b_current-(learning_rate)*b_gradient
  m_new=m_current-(learning_rate)*m_gradient
  return [b_new, m_new]

def run_(x, y, m_current, b_current, learning_rate,steps):
	for i in range(steps):
#	  if(i%5000==0):
#	   print("error: {0} ".format(calculate_error(x, y, m_current, b_current)))
	  b_current, m_current = update_weights(m_current, b_current, x, y, learning_rate)
	return [b_current, m_current]

if __name__=='__main__':
	#Equatin of a line is Y = m*X + b; 
	#m = slope of the line respect to +ve X-axis and
	#b = Y-intercept of the line

	x = np.array([2, 4, 5, 7, 8, 3]) 
	y = 2*x+5 
	#We'have created a y having m=2 and b=5 we'll try to fit a line to x and check if we get these values of m and b in the end

	m_current=0 #initial slope (guess)
	b_current=0 #initial y-intercept (guess)

	#learning rate is the rate(magnitude) at which our weights(parameters) will be updated
	#steps is how many times we want to update our parameters
	learning_rate = 0.01
	steps =10000
	b_current, m_current = run_(x, y, m_current, b_current,learning_rate, steps)
	print("running...")
	print("After {0} iterations: m is {1} and b is {2} with error {3}. ".format(steps, m_current, b_current, calculate_error(x, y, m_current, b_current)))