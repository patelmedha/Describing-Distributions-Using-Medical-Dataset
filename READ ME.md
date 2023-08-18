# Describing Distributions
I will be analyzing & visualizing several features in the [Medical Dataset](https://docs.google.com/spreadsheets/d/1APV3pXiAszS_0mSgkiEt9IUNH-QmyX7KwxSAwuADl6Y/gviz/tq?tqx=out:csv&sheet=medical_data).

- The features to analyze: 
    - VitD_levels
    - Doc_visits
    - TotalCharge

- For each feature listed:
1) Plot a histogram with a kde (kernel density estimate)
    - Add a line for the mean (red)
    - Add a line for the median (green)
    - Add a line for for +1 std from the mean (black)
    - Add a line for the - 1 std from the mean (black)
    - Highlight the range between +1 and =1 std (yellow)

2) Answer the following questions:
    -Is it Discrete or Continuous?
    - Does it have a skew? If so, which direction (+/-)
    - What type of kurtosis does it display?
        - Using the Pearson calculation: 
            - Mesokurtic (Kurtosis ~3)
            - Leptokurtic (Kurtosis >3)     
            - Platykurtic (Kurtosis < 3) 


```python
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```


```python
#Loading Dataset
df = pd.read_csv('Data/data.csv')
df.head(2)
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
      <th>State</th>
      <th>Lat</th>
      <th>Lng</th>
      <th>Area</th>
      <th>Children</th>
      <th>Age</th>
      <th>Income</th>
      <th>Marital</th>
      <th>Gender</th>
      <th>ReAdmis</th>
      <th>...</th>
      <th>Hyperlipidemia</th>
      <th>BackPain</th>
      <th>Anxiety</th>
      <th>Allergic_rhinitis</th>
      <th>Reflux_esophagitis</th>
      <th>Asthma</th>
      <th>Services</th>
      <th>Initial_days</th>
      <th>TotalCharge</th>
      <th>Additional_charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>34.34960</td>
      <td>-86.72508</td>
      <td>Suburban</td>
      <td>1.0</td>
      <td>53</td>
      <td>86575.93</td>
      <td>Divorced</td>
      <td>Male</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>Blood Work</td>
      <td>10.585770</td>
      <td>3726.702860</td>
      <td>17939.40342</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FL</td>
      <td>30.84513</td>
      <td>-85.22907</td>
      <td>Urban</td>
      <td>3.0</td>
      <td>51</td>
      <td>46805.99</td>
      <td>Married</td>
      <td>Female</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>Intravenous</td>
      <td>15.129562</td>
      <td>4193.190458</td>
      <td>17612.99812</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 32 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 32 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   State               995 non-null    object 
     1   Lat                 1000 non-null   float64
     2   Lng                 1000 non-null   float64
     3   Area                995 non-null    object 
     4   Children            993 non-null    float64
     5   Age                 1000 non-null   int64  
     6   Income              1000 non-null   float64
     7   Marital             995 non-null    object 
     8   Gender              995 non-null    object 
     9   ReAdmis             1000 non-null   int64  
     10  VitD_levels         1000 non-null   float64
     11  Doc_visits          1000 non-null   int64  
     12  Full_meals_eaten    1000 non-null   int64  
     13  vitD_supp           1000 non-null   int64  
     14  Soft_drink          1000 non-null   int64  
     15  Initial_admin       995 non-null    object 
     16  HighBlood           1000 non-null   int64  
     17  Stroke              1000 non-null   int64  
     18  Complication_risk   995 non-null    object 
     19  Overweight          1000 non-null   int64  
     20  Arthritis           994 non-null    float64
     21  Diabetes            994 non-null    float64
     22  Hyperlipidemia      998 non-null    float64
     23  BackPain            992 non-null    float64
     24  Anxiety             998 non-null    float64
     25  Allergic_rhinitis   994 non-null    float64
     26  Reflux_esophagitis  1000 non-null   int64  
     27  Asthma              1000 non-null   int64  
     28  Services            995 non-null    object 
     29  Initial_days        1000 non-null   float64
     30  TotalCharge         1000 non-null   float64
     31  Additional_charges  1000 non-null   float64
    dtypes: float64(14), int64(11), object(7)
    memory usage: 250.1+ KB



```python
# Filtering required features
df =df[['VitD_levels', 'Doc_visits', 'TotalCharge']]
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
      <th>VitD_levels</th>
      <th>Doc_visits</th>
      <th>TotalCharge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.141466</td>
      <td>6</td>
      <td>3726.702860</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.940352</td>
      <td>4</td>
      <td>4193.190458</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.057507</td>
      <td>4</td>
      <td>2434.234222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.576858</td>
      <td>4</td>
      <td>2127.830423</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.439069</td>
      <td>5</td>
      <td>2113.073274</td>
    </tr>
  </tbody>
</table>
</div>



# VitD_levels



```python
#Determing Kurtosis type 
kurt = stats.kurtosis(df['VitD_levels'], fisher = False)
kurt

```




    3.013147515833447




```python
mean = df["VitD_levels"].mean()
median = df["VitD_levels"].median()
std = df["VitD_levels"].std()
plus_std = mean + std
minus_std = mean - std
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(df["VitD_levels"], stat="probability", kde=True)
ax.axvline(mean, color="red", label=f"Mean - {mean:,.2f}")
ax.axvline(median, color="green", label = f"Median - {median:,.2f}")
ax.axvline(plus_std, color="black", label=f"+1 STD - {plus_std:,.2f}")
ax.axvline(minus_std, color="black", label=f"-1 STD - {minus_std:,.2f}")
ax.axvspan(plus_std, minus_std, color="yellow", zorder=0)
ax.set_title("VitD Levels")
ax.legend();
```


    
![png](output_7_0.png)
    


1) Is it Discrete or Continuous?
    - Continuous
2) Does it have a skew? If so, which direction (+/-)
    - Doesn't have a skew
3) What type of kurtosis does VitD_Levels display?
    - Mesokurtic (Kurtosis ~3)

# Doc_visits


```python
#Determing Kurtosis type 
kurt = stats.kurtosis(df['Doc_visits'], fisher = False)
kurt
```




    2.9919958083381206




```python
mean = df["Doc_visits"].mean()
median = df["Doc_visits"].median()
std = df["Doc_visits"].std()
plus_std = mean + std
minus_std = mean - std
fig, ax = plt.subplots(figsize = (10,6))
sns.histplot(df["Doc_visits"], stat="probability", kde=True)
ax.axvline(mean, color="red", label=f"Mean - {mean:,.2f}")
ax.axvline(median, color="green", label = f"Median - {median:,.2f}")
ax.axvline(plus_std, color="black", label=f"+1 STD - {plus_std:,.2f}")
ax.axvline(minus_std, color="black", label=f"-1 STD - {minus_std:,.2f}")
ax.axvspan(plus_std, minus_std, color="yellow", zorder=0)
ax.set_title("Doc Visits")
ax.legend();
```


    
![png](output_11_0.png)
    


1) Is it Discrete or Continuous?
    - Discrete
2) Does it have a skew? If so, which direction (+/-)
    - Has a slight positive skew
3) What type of kurtosis does Doc_visits display?
    - Mesokurtic (Kurtosis ~3)

# TotalCharge


```python
#Determining kurtosis type
kurt = stats.kurtosis(df['TotalCharge'], fisher = False)
kurt
```




    3.2650077463439384




```python
mean = df["TotalCharge"].mean()
median = df["TotalCharge"].median()
std = df["TotalCharge"].std()
plus_std = mean + std
minus_std = mean - std
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(df["TotalCharge"], stat="probability", kde=True)
ax.axvline(mean, color="red", label=f"Mean - {mean:,.2f}")
ax.axvline(median, color="green", label=f"Median - {median:,.2f}")
ax.axvline(plus_std, color="black", label=f"+1 STD - {plus_std:,.2f}")
ax.axvline(minus_std, color="black", label=f"-1 STD - {minus_std:,.2f}")
ax.axvspan(plus_std, minus_std, color="yellow", zorder=0)
ax.set_title("Total Charge")
ax.legend();
```


    
![png](output_15_0.png)
    


1) Is it Discrete or Continuous?
    - Continuous
2) Does it have a skew? If so, which direction (+/-)
    - Has a positive skew
3) What type of kurtosis does TotalCharge display?
    - Leptokurtic (Kurtosis >3)


```python

```
