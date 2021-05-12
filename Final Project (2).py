#!/usr/bin/env python
# coding: utf-8

# PCA for Removing Risk From a Portfolio
# By Nikhil Shah, Parsh Jain, and Bhaardwaj Vemapulli

# In this research project, we are going to use PCA to remove some of the variation in a portfolio over time, creating a more "market-neutral" portfolio. It has been determined that by using the eigenvectors of PCA, you can find different weightings for assets in a portfolio. By incorporating more and more of these eigenvectors, we account for more variation in the overall portfolio.
# 
# In modern portfolio theory, using 1 principal component is often called the "market portfolio" in a market of just the assets involved.
# 
# It is often necessary to remove the correlated market portfolio from your portfolio, so that you are not prone to what are called "systemic" risk factors.

# In[239]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# We will be working with the following assets.

# In[240]:


data = pd.read_csv("data.csv")
data.head()


# We convert to log-returns. These are often used in finance instead of normal returns, as they are much easier to compare across assets in different time periods.

# In[241]:


S = np.array(data)
R = np.diff(np.log(S), axis = 0)


# In[242]:


equal_weight = np.array([1/len(S[0])] * len(S[0]))
X = np.dot(equal_weight,R.T)/ np.std(R.T, axis = 0)
plt.plot(X)
plt.xlabel("Dates")
plt.ylabel("Returns")
plt.show()
print("Volatility of Equal Weighted Portfolio:")
print(np.std(X))


# First we create a correlation matrix (We will be doing correlation-based PCA).

# In[243]:


Y = (R - np.mean(R, axis = 0)) / np.std(R, axis = 0)
cor_matrix = np.corrcoef(Y.T)


# We fit the PCA and show our explained variance ratios.

# In[244]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 8)
pca.fit_transform(cor_matrix)

pca.explained_variance_ratio_


# This demonstrates how much the PCAs explain, as well as their cumulative explained variance.

# In[245]:


import matplotlib.pyplot as plt

plt.bar(range(1,len(pca.explained_variance_ratio_ )+1),pca.explained_variance_ratio_ )
plt.ylabel('Explained variance')
plt.xlabel('No. of PCs')
plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),
         np.cumsum(pca.explained_variance_ratio_),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='upper left')
plt.show()


# We get our eigenvectors as well. We show that using the eigenvectors as weights, we are able to successfully decrease the amount of volatility that would be in the portfolio.

# In[246]:


S = np.dot(Y.T, Y) / (len(Y) - 1)
w, v = np.linalg.eig(S)
w = np.diag(w)
sd = []
for i in range(1,9):
    Q = v[:i] / np.std(R, axis = 0)
    F = np.sum(np.dot(Q, R.T),axis = 0)
    plt.plot(F)
    plt.xlabel("Dates")
    plt.ylabel("Returns")
    plt.show()
    s = np.std(F)
    sd.append(s)
    print("Volatility with " + str(i) + " Principal Components: ")
    print(s)
    


# We have successfully obtained a much more risk-free portfolio through using PCA. By using less principal components, we get less volatility in our overall portfolio. 

# In[247]:


plt.plot(range(1,9),sd)
plt.xlabel("# of Principal Components")
plt.ylabel


# In[ ]:





# In[ ]:




