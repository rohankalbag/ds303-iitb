# %% [markdown]
# # DS 303 Assignment 1
# ## Rohan Rajesh Kalbag
# ### 20D170033
# 
# #### Honour Code
# 
# 
# I hereby declare that I have not indulged in any academic malpractices such as plagiarism and would like to assure you that, any code written in this is written by me alone. I would like to add that any resources/threads found on online sites will be mentioned in the document that was used for reference.
# 

# %%
#Importing all libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
#Reading from CSV file
l = pd.read_csv("restaurent.csv").sample(50,random_state=0)
#Sampling 50 entries randomly from the csv file
l.reset_index(drop=True, inplace=True)
print(len(l))
l.head()

# %%
#Plotting scatter plot between food and price
plt.scatter(l['food'],l['price'])
plt.title("Scatter Plot")
plt.xlabel("Food")
plt.ylabel("Price")
plt.show()

# %%
#Plotting scatter plot between service and price
plt.scatter(l['service'],l['price'])
plt.title("Scatter Plot")
plt.xlabel("Service")
plt.ylabel("Price")
plt.show()

# %%
#Plotting scatter plot between decor and price
plt.scatter(l['decor'],l['price'])
plt.title("Scatter Plot")
plt.xlabel("Decor")
plt.ylabel("Price")
plt.show()

# %%
#Training dataset
X_train = l[['food','decor','service']]
y_train = l['price']

# %%
#Using the sklearn Linear Regression Model for Multivariate Linear Regression
model = LinearRegression()
#Fitting the model with the train data
model.fit(X_train,y_train)

# %%
# Prediction using the model for the input of (125, 148, 265) 
prediction = model.predict(pd.DataFrame({'food':[125],'decor':[148],'service':[265]}))

# %%
#The value of the coefficients obtained after Linear Regression
print(model.coef_)
print(model.intercept_)

# %%
#Standard deviation with degrees of freedom n-2 to obtain 95% confidence interval
stdev = np.sqrt(sum((model.predict(X_train) - y_train)**2) / (len(y_train) - 2))

# %%
print(stdev)

# %%
#Evaluating the confidence interval
confidence_interval = (prediction[0]-stdev*1.96, prediction[0]+stdev*1.96)
print(confidence_interval)

# %%
#Single variable linear regression for Food and Price
model2 = LinearRegression()
X_train, y_train = np.array(l['food']).reshape(-1,1), np.array(l['price']).reshape(-1,1)
model2.fit(X_train,y_train)
#Printing the coefficient and the y intercept for the regression model
print(model2.coef_)
print(model2.intercept_)

# %%
x = np.linspace(10,25,100)
y = model2.coef_[0]*x + model2.intercept_
plt.plot(x,y)
# Comparision of the scatter plot and single variable linear regression
plt.scatter(l['food'],l['price'])
plt.title("Scatter Plot Compared To Fitted Line")
plt.xlabel("Food")
plt.ylabel("Price")
plt.plot()
plt.show()

# %%
#Plotting scatter plot between food and service
plt.scatter(l['food'],l['service'])
plt.title("Scatter Plot")
plt.xlabel("Food")
plt.ylabel("Service")
plt.show()

# %%
model.predict(pd.DataFrame({'food':[20],'decor':[3],'service':[17]}))


