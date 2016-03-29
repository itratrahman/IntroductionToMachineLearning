##import statements for numpy, matplotlib, and scipy from stats
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

##data set x to store the sizes of the houses
x = np.array([112,345,198,305,372,550,302,420,578])

##data set y to store the prices of the houses
y = np.array([1120,1523,2102,2230,2600,3200,3409,3689,4460])

##Training the data using linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

##Ploting the data
plt.plot(x,y,'ro',color = 'black')

##Setting the y label
plt.ylabel('Price')

##Setting the x label
plt.xlabel('Size of house')

##Setting the axis
plt.axis([0,600, 0,6000])

##Plotting the linear regression line
plt.plot(x, x*slope+intercept,'b')

##Plot command
plt.plot()

##Plot show command
plt.show()

####Making a prediction with a data
##data
newX = 150
##Calculating the result
newY = newX*slope+intercept
##Printing the result i.e. the price of the given house
print("The newY is: ",newY)