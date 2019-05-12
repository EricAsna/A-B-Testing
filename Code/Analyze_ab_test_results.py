#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# This project aims at understanding the results of an A/B test run by an e-commerce website.  The goal is to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# <a id='probability'></a>
# #### Part I - Probability
# 

# In[42]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# a. Read in the dataset and take a look at the top few rows here:

# In[43]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[44]:


rows = df.shape[0]
rows


# c. The number of unique users in the dataset.

# In[45]:


df.nunique()


# d. The proportion of users converted.

# In[46]:


df.query('converted == 1').user_id.nunique()/df.user_id.nunique()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[47]:


df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]


# f. Do any of the rows have missing values?

# In[48]:


df.isnull().sum()


# `2.` Those rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, is removed from the dateset.

# In[49]:


ind1 = df.query('group == "treatment" and landing_page != "new_page"').index.tolist()
ind2 = df.query('group != "treatment" and landing_page == "new_page"').index.tolist()
ind = ind1 + ind2
ind = sorted(ind)
df2 = df.drop(ind , axis = 0)
df2 = df2.reset_index(drop = True)
df2.head()


# In[50]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# a. How many unique **user_id**s are in **df2**?

# In[55]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[56]:


df2[df2.user_id.duplicated() == True]['user_id']


# c. What is the row information for the repeat **user_id**? 

# In[57]:


df2[df2.user_id.duplicated() == True]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[58]:


df2.drop_duplicates(subset = 'user_id', keep = 'first', inplace = True)
len(df2)


# a. What is the probability of an individual converting regardless of the page they receive?

# In[59]:


len(df2.query('converted == 1')) / len(df2)


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[60]:


conv_ctr = len(df2.query('group == "control" and converted == 1')) / len(df2.query('group == "control"'))
conv_ctr


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[61]:


conv_treat = len(df2.query('group == "treatment" and converted == 1')) / len(df2.query('group == "treatment"'))
conv_treat


# d. What is the probability that an individual received the new page?

# In[62]:


len(df2.query('landing_page == "new_page"')) / len(df2)


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# **The probablity of conversion for the treatment and control groups is very close to one another with the control group having slightly higher value. Nonetheless, performing A/B test through bootstrapping will enable us to understand whether the difference is statistically significant.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# `1.` If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# $$H_0 : P_{old} - P_{new} = 0$$
# $$H_1 : P_{new} - P_{old} \neq 0$$

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[63]:


p_new = len(df.query('converted == 1')) / len(df)
p_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[64]:


p_old = len(df.query('converted == 1')) / len(df)
p_old


# c. What is $n_{new}$?

# In[65]:


n_new = len(df2.query('group == "treatment"'))
n_new


# d. What is $n_{old}$?

# In[66]:


n_old = len(df2.query('group == "control"'))
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[67]:


new_page_converted = np.random.binomial(1, p_new, n_new)


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[68]:


old_page_converted = np.random.binomial(1, p_old, n_old)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[69]:


p_new2 = new_page_converted.mean()
p_old2 = old_page_converted.mean()
samp_diff = p_new2 - p_old2
samp_diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[70]:


p_diffs = []
p_diffs = np.random.binomial(n_new, p_new, 10000)/n_new - np.random.binomial(n_old, p_old, 10000)/n_old                                                                


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[71]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[72]:


obs_diff = conv_treat - conv_ctr
pval = (p_diffs > obs_diff).mean()
pval


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **The value calculated in part j is called p-value. P-value shows the probability of our statistic in favour of the alternative if the null hypothesis is true. Since the p-value of 0.9 is larger than our specified type 1 error, we fail to reject the null. In other words, the difference between the converted success rates for the control and treatment groups is not statistically significant.**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[73]:


import statsmodels.api as sm

convert_old = len(df2.query('converted == 1 and landing_page == "old_page"'))
convert_new = len(df2.query('converted == 1 and landing_page == "new_page"'))
n_old = len(df2.query('landing_page == "old_page"'))
n_new = len(df2.query('landing_page == "new_page"'))
convert_old, convert_new, n_old, n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value. 

# In[75]:


z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative = 'larger')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Z-score is a measure of standard deviation and p-value is associated with the probability. Since the confidence level is 95%, the critical z-score is between -1.96 and 1.96 standard deviations. The p-value obtained here to be 0.905 is similar to what calculated in the prevous section. Both the z-score and p-value suggest that we cannot reject the null hypothesis.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Multiple linear regression because a categorical x-variable is going to be used to predict a quantitative response.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[76]:


df2['intercept'] = 1
dummy = pd.get_dummies(df['group'])
df2['ab_page'] = dummy['treatment']
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[77]:


import statsmodels.api as sm
lm = sm.OLS(df2['converted'], df2[['intercept', 'ab_page']])
results = lm.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[78]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?

# **The p-value associated with ab_page is 0.94. It is different from part 2 because here the null hypothersis is that the coefficient is equal to zero and to understand whether there is a statistically linear relationship between converted and the receiving page.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **From the calculated p-value in the regression model, there is no statistical linear relationship between conversion and receiving page. Therefore, it is a  good idea to explore for other factors which may help us in predicting conversion.
# Multicollinearity and linearity are some disadvantages we may face by adding extra terms into our regression model.**
# 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  
# 
# Does it appear that country had an impact on conversion?  

# In[80]:


country = pd.read_csv('countries.csv')
country['user_id'].duplicated().sum()
df_new = df2.merge(country, on = 'user_id', how = 'right')
df_new.head(5)


# In[81]:


df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head(5)


# In[82]:


lm = sm.OLS(df_new['converted'], df_new[['intercept', 'ab_page', 'CA', 'UK']])
results = lm.fit()
results.summary()


# **From the calculated coefficients we can say that country does not have a tangible impact on the conversion rate. The p-values are both greater than 5% meaning that null cannot be rejected. No statistical linear relationship exists for conversion rate vs country.**

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[83]:


df_new['UK_ab_page'] = df_new['UK'] * df_new['ab_page']
df_new['CA_ab_page'] = df_new['CA'] * df_new['ab_page']

lm2 = sm.OLS(df_new['converted'], df_new[['intercept', 'ab_page', 'CA', 'UK', 'UK_ab_page', 'CA_ab_page']])
results2 = lm2.fit()
results2.summary()


# **The above results show that even the interaction of countries and receiving page does not have a significant effect on conversion.**

# In[84]:


# Understanding the duration of time used to collect the data.
df2['timestamp'].min(), df2['timestamp'].max()


# ## Summary & Conclusions
# 
# An A/B test and regressions were conducted for an e-commerce webpage to understand if the new page design attracts more users to buy a product compared to the old one. 
# For the A/B test the null and alternative hypotheses were defined and the sampling dsitribution for the difference in conversion rate was simulated. The computed p-value suggested that the null hypothesis cannot be rejected and therefore the difference between the conversion rates of the two pages is not statistically significant.
# Multiple linear regression model was also used to find out whether there's a statistical linear relationship between explanatory variables and conversion. No statistical relaionship was obtained between neither of conversion-receiving webpage, conversion-country and even between the conversion and the interaction of country-receiving page. R-squared values for the all models fitted were almost 0. It means that our models did not fit at all to the data. Other models might be used to achieve a better fit.
# 
# Note: From the timestamp column, the total duration of time used to collect the data is nearly 22 days. This duration is not enough to make our A/B testing results reliable. In my opinion, more data need to be collected for a duration of at least 3 months in order to take into account the novelty effect and change aversion. Moreover, I believe smarter metrcis should be considered for the prediction of conversion rate.

# In[85]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

