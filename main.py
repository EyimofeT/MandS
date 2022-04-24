import visualizeData as vd
import linearRegression as lr
import classification as cl

print("Cutaneous Melonoma Predictive Analysis")

#training and testing using Linear Regression Model
print("Linear Regression Model")
lr.train()

#training and testing using Classifier Model 
#Decision Tree classifier
#Random Forest classifier
#Gradient Boosting Classifier
print("Classification Models")
cl.train()

# print("Data Visuals")
# #visualize
# vd.visualizeData()
