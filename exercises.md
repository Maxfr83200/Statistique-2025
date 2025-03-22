# Exercise 1. - Getting and Knowing your Data

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out [Occupation Exercises Video Tutorial](https://www.youtube.com/watch?v=W8AB5s-L3Rw&list=PLgJhDSE2ZLxaY_DigHeiIDC1cD09rXgJv&index=4) to watch a data scientist go through the exercises

### Step 1. Import the necessary libraries

```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 

### Step 3. Assign it to a variable called users and use the 'user_id' as index

```python
users = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user", sep='|', index_col='user_id')

```



### Step 4. See the first 25 entries

```python
users.head(25)
```

### Step 5. See the last 10 entries

```python
users.tail(10)
```

### Step 6. What is the number of observations in the dataset?

```python
users.shape[0]
```

### Step 7. What is the number of columns in the dataset?

```python
users.shape[1]
```

### Step 8. Print the name of all the columns.

```python
for column in users.columns:
    print(column)
```

### Step 9. How is the dataset indexed?

```python
list(users.index)
```

### Step 10. What is the data type of each column?

```python
users = users.astype({'age': 'int'})
    
users.dtypes
```

### Step 11. Print only the occupation column

```python
users[['occupation']]
```

### Step 12. How many different occupations are in this dataset?

```python
users['occupation'].nunique()
```

### Step 13. What is the most frequent occupation?

```python
users['occupation'].value_counts().head(1)
```

### Step 14. Summarize the DataFrame.

```python
users.info()
```

### Step 15. Summarize all the columns

```python
users.describe(include='all')
```

### Step 16. Summarize only the occupation column

```python
users.occupation.describe()
```

### Step 17. What is the mean age of users?

```python
users.age.mean()
```

### Step 18. What is the age with least occurrence?

```python
users.age.value_counts().tail(1)
```



# Exercise 2. - Filtering and Sorting Data

Check out [Euro 12 Exercises Video Tutorial](https://youtu.be/iqk5d48Qisg) to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet.

### Step 1. Import the necessary libraries

```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv). 

### Step 3. Assign it to a variable called euro12.

```python
euro12 = pd.read_csv("https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv", sep=',')
euro12
```

### Step 4. Select only the Goal column.

```python
euro12.Goals
```

### Step 5. How many team participated in the Euro2012?

```python
euro12.shape[0]
```

### Step 6. What is the number of columns in the dataset?

```python
euro12.shape[1]
```

### Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline

```python
euro12[['Team', 'Yellow Cards', 'Red Cards']]
```

### Step 8. Sort the teams by Red Cards, then to Yellow Cards

```python
euro12.sort_values(by=['Red Cards', 'Yellow Cards'], ascending=[False,False])
```

### Step 9. Calculate the mean Yellow Cards given per Team

```python
euro12['Yellow Cards'].mean()
```

### Step 10. Filter teams that scored more than 6 goals

```python
euro12[euro12.Goals > 6]
```

### Step 11. Select the teams that start with G

```python
euro12[euro12.Team.str.startswith('G')]
```

### Step 12. Select the first 7 columns

```python
euro12.iloc[:, 0:7]
```

### Step 13. Select all columns except the last 3.

```python
euro12.iloc[:, :-3]
```

### Step 14. Present only the Shooting Accuracy from England, Italy and Russia

```python
euro12[euro12['Team'].isin(['England', 'Italy', 'Russia'])][['Team', 'Shooting Accuracy']]
```






# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries

```python
import pandas as pd

```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 

### Step 3. Assign it to a variable called drinks.

```python
drinks = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv", sep=',')
drinks
```

### Step 4. Which continent drinks more beer on average?

```python
drinks.sort_values(by='beer_servings', ascending=False).head(1)
```

### Step 5. For each continent print the statistics for wine consumption.

```python
drinks.groupby('continent')['wine_servings'].describe()
```

### Step 6. Print the mean alcohol consumption per continent for every column

```python
drinks.groupby('continent').mean(numeric_only=True)
```

### Step 7. Print the median alcohol consumption per continent for every column

```python
drinks.groupby('continent').median(numeric_only=True)
```

### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame

```python
pd.DataFrame({
    'Mean': [drinks['spirit_servings'].mean()],
    'Min': [drinks['spirit_servings'].min()],
    'Max': [drinks['spirit_servings'].max()]
})
```

