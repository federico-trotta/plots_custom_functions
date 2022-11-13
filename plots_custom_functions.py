#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# # LABELING PLOTS

# In[2]:


def get_label(TITLE, X, Y):
    ''' 
    The get_label() function appends the labels to a seaborn plot. Arguments in the following order:
    Title, x label, y label
    '''
    plt.title(TITLE)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show


# In[ ]:





# # PLOTTING KDE

# In[4]:


def plot_kde(y_test, y_test_pred):
    ''' 
    The plot(kde) function plots the KDE. Inputs are just real and predicted y values, in this order:
    y_test, y_test_pred
    '''
    #figsize
    plt.figure(figsize=(10, 7))

    #Kernel Density Estimation plot
    ax = sns.kdeplot(y_test, color='r', label='Actual Values') #actual values
    sns.kdeplot(y_test_pred, color='b', label='Predicted Values', ax=ax) #predicted values

    #showing title
    plt.title('Actual vs Predicted values')
    #showing legend
    plt.legend()
    #showing plot
    plt.show()


# In[ ]:





# # PLOTTING ACTUAL VS PREDICTED SCATTERPOT WITH A LINE

# In[13]:


def scatter_with_regr(y_test, y_test_pred):
    ''' 
    The scatter_with_regr() function plots actual vs predicted values with a regression line.
    Arguments in the following order:
    y_test, y_test_pred
    '''
    #figure size
    plt.figure(figsize=(10, 7))

    #scatterplot of y_test and y_test_pred
    plt.scatter(y_test, y_test_pred)
    plt.plot(y_test, y_test, color='r')

    #labeling
    plt.title('ACTUAL VS PREDICTED VALUES')
    plt.xlabel('ACTUAL VALUES')
    plt.ylabel('PREDICTED VALUES')

    #showig plot
    plt.show()


# In[ ]:





# # RESIDUALS PLOT

# In[15]:


def plot_residuals(y_test, y_test_pred, title, x_label, y_label):
    ''' 
    The plot_residuals() function plots the residuals. Arguments have to be passed in thid order:
    y_test, y_test_pred, title, x_label and y_label.
    REMEMBER: when invoking the function, y_test e y_test_pred have to be passed
    with no quotation marks if are not from
    the same dataframe (as usually it is).
    Then, pass: title, x_label and y_label (usying quotation marks)
    '''
    #figure size
    plt.figure(figsize=(10, 7))

    #residual plot
    sns.residplot(x=y_test, y=y_test_pred)

    #labeling
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


# In[ ]:


## LEARNING CURVES

#coding the learning curves
def plot_learning_curves(model, X, y):
    '''
    before invoking it, fix a seed. The typical value is seed=42.
    Aftter that, invoke the function like so:
    
    #plotting
    plot_learning_curves(your model, X, y) #"your model" is the model you selected
    
    then, invoke the "get_label" function
    '''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')


# # PRINT DOCUMENTATION

# In[16]:


print(get_label.__doc__)
print(plot_kde.__doc__)
print(scatter_with_regr.__doc__)
print(plot_residuals.__doc__)
print(plot_learning_curves.__doc__)


# In[ ]:




