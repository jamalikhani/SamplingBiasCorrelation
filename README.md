
# Sampling bias correction by applying Bayes Rule 

#### Author: James Alikhani    
<alikhani.james@gmail.com>   
November 2018

### Problem Definition

Having a general information of universal (national) demographic distribution, weight the sampling from a biased population for bootstrapping technique. 

### Summary
In this practice, I have applied the Bayesian Inference concept to a bias selected population to correct the biases by assigning new weights to each combination of categories; that can be later used in a bootstrapping (random selection with replacement) techniques to realize a newly unbiased (or less biased) population from the existing biased population. 


```python
import pandas as pd
import numpy as np
import itertools
```

### *P(person|categories)*: the observed likelihood (probability)

####  Loading biased population


```python
# sampled dataset: known to be biased
df = pd.read_csv("demographic_attributes.csv")
```


```python
df.shape
```




    (400000, 4)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person id</th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>75_84</td>
      <td>Some College</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85_120</td>
      <td>HS Diploma</td>
      <td>white</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>25_34</td>
      <td>Some College</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>55_64</td>
      <td>HS Diploma</td>
      <td>black</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>45_54</td>
      <td>Bachelor Degree</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>




```python
# missing data analysis
print('% of missing data in each column:')
miss_lst = df.isnull().sum(axis=0)/len(df)*100
print(miss_lst)
```

    % of missing data in each column:
    person id    0.0000
    age          0.0000
    education    5.5475
    ethnicity    0.0000
    dtype: float64
    

Only "education" field has some missing values. We replace the Nan vlaues in this category with "Missing" that makes it easier for the further analysis. 


```python
df.loc[df.education.isnull(), 'education'] = 'Missing'
```

#### counting the number of individuals in each category

categories are: age, education, and ethnicity. 


```python
df_selected_categories = df.groupby(['age','education','ethnicity']).count().reset_index()
df_selected_categories.columns = ['age','education','ethnicity', 'count']
df_selected_categories.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>asian</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>black</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>hispanic</td>
      <td>89</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>white</td>
      <td>155</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18_24</td>
      <td>Bachelor Degree</td>
      <td>asian</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_selected_categories['count'].sum()
```




    400000




```python
df_selected_categories['p_likelihood'] = df_selected_categories['count']/\
df_selected_categories['count'].sum()
```


```python
df_selected_categories.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
      <th>count</th>
      <th>p_likelihood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>asian</td>
      <td>6</td>
      <td>0.000015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>black</td>
      <td>25</td>
      <td>0.000063</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>hispanic</td>
      <td>89</td>
      <td>0.000222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>white</td>
      <td>155</td>
      <td>0.000387</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18_24</td>
      <td>Bachelor Degree</td>
      <td>asian</td>
      <td>90</td>
      <td>0.000225</td>
    </tr>
  </tbody>
</table>
</div>



### *P(categories)*: prior probabality from national distribution

#### Loading referenced demographical categories


```python
df_ref = pd.read_csv("national_demographic_census-derived.csv")
```


```python
df_ref.shape
```




    (22, 2)




```python
df_ref.loc[0:12,'category'] = 'age'
df_ref.loc[12:17,'category'] = 'education'
df_ref.loc[17:,'category'] = 'ethnicity'
```


```python
df_ref_sum = df_ref.groupby(['category'])['number of individuals'].sum().reset_index()
df_ref_sum.columns = ['category','sum_per_category']
df_ref_sum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>sum_per_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>120482688</td>
    </tr>
    <tr>
      <th>1</th>
      <td>education</td>
      <td>92659102</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ethnicity</td>
      <td>116753640</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ref = df_ref.merge(df_ref_sum, on='category', how='left')
```


```python
df_ref['prob'] = df_ref['number of individuals']/df_ref['sum_per_category']
```


```python
df_ref.drop(['sum_per_category'], axis=1, inplace=True)
df_ref.columns = ['demographic category','count_national','category','prob_national']
df_ref
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>demographic category</th>
      <th>count_national</th>
      <th>category</th>
      <th>prob_national</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>11839159</td>
      <td>age</td>
      <td>0.098264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25_34</td>
      <td>16399632</td>
      <td>age</td>
      <td>0.136116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35_44</td>
      <td>15335704</td>
      <td>age</td>
      <td>0.127286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45_54</td>
      <td>16430762</td>
      <td>age</td>
      <td>0.136374</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55_64</td>
      <td>15148777</td>
      <td>age</td>
      <td>0.125734</td>
    </tr>
    <tr>
      <th>5</th>
      <td>65_74</td>
      <td>9990412</td>
      <td>age</td>
      <td>0.082920</td>
    </tr>
    <tr>
      <th>6</th>
      <td>75_84</td>
      <td>5221430</td>
      <td>age</td>
      <td>0.043338</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0_4</td>
      <td>7500407</td>
      <td>age</td>
      <td>0.062253</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5_9</td>
      <td>7748669</td>
      <td>age</td>
      <td>0.064314</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10_14</td>
      <td>7815759</td>
      <td>age</td>
      <td>0.064870</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15_17</td>
      <td>4758751</td>
      <td>age</td>
      <td>0.039497</td>
    </tr>
    <tr>
      <th>11</th>
      <td>85_120</td>
      <td>2293226</td>
      <td>age</td>
      <td>0.019034</td>
    </tr>
    <tr>
      <th>12</th>
      <td>&lt; Than HS Diploma</td>
      <td>12274025</td>
      <td>education</td>
      <td>0.132464</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bachelor Degree</td>
      <td>16305721</td>
      <td>education</td>
      <td>0.175975</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Graduate Degree</td>
      <td>9343192</td>
      <td>education</td>
      <td>0.100834</td>
    </tr>
    <tr>
      <th>15</th>
      <td>HS Diploma</td>
      <td>25799018</td>
      <td>education</td>
      <td>0.278429</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Some College</td>
      <td>28937146</td>
      <td>education</td>
      <td>0.312297</td>
    </tr>
    <tr>
      <th>17</th>
      <td>asian</td>
      <td>6145151</td>
      <td>ethnicity</td>
      <td>0.052633</td>
    </tr>
    <tr>
      <th>18</th>
      <td>black</td>
      <td>14626476</td>
      <td>ethnicity</td>
      <td>0.125276</td>
    </tr>
    <tr>
      <th>19</th>
      <td>hispanic</td>
      <td>21953456</td>
      <td>ethnicity</td>
      <td>0.188032</td>
    </tr>
    <tr>
      <th>20</th>
      <td>islander</td>
      <td>190389</td>
      <td>ethnicity</td>
      <td>0.001631</td>
    </tr>
    <tr>
      <th>21</th>
      <td>white</td>
      <td>73838168</td>
      <td>ethnicity</td>
      <td>0.632427</td>
    </tr>
  </tbody>
</table>
</div>



By assuming that the age, ethnicity, and education are independent, we can say that the probability of the combination (intersection) is a product of probabilities. NOTE: The education and age dependency cannot be driven from the current information. 


```python
age = list(df_ref[df_ref.category=='age']['demographic category'])
edu = list(df_ref[df_ref.category=='education']['demographic category'])
edu.append('Missing')
eth = list(df_ref[df_ref.category=='ethnicity']['demographic category'])
categories = [age, edu, eth]
```


```python
categories_combination = list(itertools.product(*categories))
```


```python
df_p_prior = pd.DataFrame(categories_combination, columns=['age','education','ethnicity'])
```


```python
df_p_prior.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>asian</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>hispanic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>islander</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in ['age','education','ethnicity']:
    df_p_prior = df_p_prior.merge(df_ref[['demographic category','prob_national']],
                                  left_on=col,
                                  right_on='demographic category', how='left')
    df_p_prior['p_'+col]=df_p_prior['prob_national']
    df_p_prior.drop(['demographic category','prob_national'],axis=1, inplace=True)
```

For missing values in education, we just use the probabability of the age and ethnicity. 


```python
df_p_prior.loc[df_p_prior.education=='Missing', 'p_education'] = 1
```


```python
df_p_prior['p_prior'] = df_p_prior['p_age']*df_p_prior['p_education']*df_p_prior['p_ethnicity']
```


```python
df_p_prior.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
      <th>p_age</th>
      <th>p_education</th>
      <th>p_ethnicity</th>
      <th>p_prior</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>asian</td>
      <td>0.098264</td>
      <td>0.132464</td>
      <td>0.052633</td>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>black</td>
      <td>0.098264</td>
      <td>0.132464</td>
      <td>0.125276</td>
      <td>0.001631</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>hispanic</td>
      <td>0.098264</td>
      <td>0.132464</td>
      <td>0.188032</td>
      <td>0.002448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>islander</td>
      <td>0.098264</td>
      <td>0.132464</td>
      <td>0.001631</td>
      <td>0.000021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>white</td>
      <td>0.098264</td>
      <td>0.132464</td>
      <td>0.632427</td>
      <td>0.008232</td>
    </tr>
  </tbody>
</table>
</div>



### Bayesian Inference:
#### P(categories|person)=P(person|categories)*P(categories)/p(E)_

After calculating the prior and likelihood probabilities, we can now obtain the posterior probabilities by using Bayesian rule. We later use these posterior probabilities to construct the weights requested in this problem.


```python
df_selected_categories = df_selected_categories.merge(
    df_p_prior[['age','education','ethnicity','p_prior']],
    on=['age','education','ethnicity'], how='left')
```


```python
# Bayesian inference numerator: P_likelihood*P_prior
df_selected_categories['p_posterior'] = df_selected_categories['p_likelihood']*\
df_selected_categories['p_prior']
```


```python
# normalizing the p_posteriosrs (a numerical approach to the denominator of the Bayesian inference)
df_selected_categories['p_posterior'] = df_selected_categories['p_posterior']/\
df_selected_categories['p_posterior'].sum()
```


```python
# check for normalized p_posterior
df_selected_categories['p_posterior'].sum()
```




    1.0



### Resampling weights to correct bias selection


```python
# This is the expected value of each categories based on posterior probabilities
df_selected_categories['count_posterior'] = df_selected_categories['p_posterior']*400000
```


```python
# comparing values with expected values 
df_selected_categories['sampling_weight'] = df_selected_categories['count_posterior']/\
df_selected_categories['count']
```


```python
df_selected_categories.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
      <th>count</th>
      <th>p_likelihood</th>
      <th>p_prior</th>
      <th>p_posterior</th>
      <th>count_posterior</th>
      <th>sampling_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>asian</td>
      <td>6</td>
      <td>0.000015</td>
      <td>0.000685</td>
      <td>6.552209e-07</td>
      <td>0.262088</td>
      <td>0.043681</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>black</td>
      <td>25</td>
      <td>0.000063</td>
      <td>0.001631</td>
      <td>6.498059e-06</td>
      <td>2.599223</td>
      <td>0.103969</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>hispanic</td>
      <td>89</td>
      <td>0.000222</td>
      <td>0.002448</td>
      <td>3.472137e-05</td>
      <td>13.888547</td>
      <td>0.156051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>white</td>
      <td>155</td>
      <td>0.000387</td>
      <td>0.008232</td>
      <td>2.033839e-04</td>
      <td>81.353552</td>
      <td>0.524862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18_24</td>
      <td>Bachelor Degree</td>
      <td>asian</td>
      <td>90</td>
      <td>0.000225</td>
      <td>0.000910</td>
      <td>1.305666e-05</td>
      <td>5.222663</td>
      <td>0.058030</td>
    </tr>
  </tbody>
</table>
</div>



#### Weights:


```python
df_weights = df_selected_categories[['age','education','ethnicity','sampling_weight']]
```


```python
df_weights.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>ethnicity</th>
      <th>sampling_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>asian</td>
      <td>0.043681</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>black</td>
      <td>0.103969</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>hispanic</td>
      <td>0.156051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18_24</td>
      <td>&lt; Than HS Diploma</td>
      <td>white</td>
      <td>0.524862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18_24</td>
      <td>Bachelor Degree</td>
      <td>asian</td>
      <td>0.058030</td>
    </tr>
  </tbody>
</table>
</div>




```python
print_weights = True
if print_weights:
    df_weights.to_csv("weights.csv")
```

### Modifications
we know that there are some correlations between age and education. For example, a 10 years old person cannot have an advanced degree. We may need some other information to modify these prior probabilities for some combination of categories. Please note that research and data are needed for accurate modifications. Since we didn't have age below 18 in the selected population, these modifications are not crucial here.

### Conclusion
Each person belongs to a combination of three categories including *age*, *education*, and *ethnicity*.  We call this combination as *categories*. Therefore, according to the Bayesian Inference, we expect to find the following relationship between the observed probability (likelihood) of belonging each person to each category to the expected probability calculated from the nationally distributed data:

P(categories|person) = P(person|categories) * P(categories) / (normalizing factor)

In this practice, the probabilities of the right-hand side of the equation are calculated from the given data to calculate the posterior distribution (histogram). The posterior probabilities then applied to calculate the expected value of each group in the selected population of 400,000 individuals. These expected values are then compared with the existing values to calculate the weights. Please see the weights.csv!


```python
# END
```


```python

```
