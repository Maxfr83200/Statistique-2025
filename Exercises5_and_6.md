# Exercise 5

# Data visualization in Python (`pyplot`)

## Looking ahead: April, Weeks 1-2

- In April, weeks 1-2, we'll dive deep into **data visualization**.  
  - How do we make visualizations in Python?
  - What principles should we keep in mind?

## Goals of this exercise

- What *is* data visualization and why is it important?
- Introducing `matplotlib`.
- Univariate plot types:
  - **Histograms** (univariate).
  - **Scatterplots** (bivariate).
  - **Bar plots** (bivariate).

## Introduction: data visualization

### What is data visualization?

[Data visualization](https://en.wikipedia.org/wiki/Data_visualization) refers to the process (and result) of representing data graphically.

For our purposes today, we'll be talking mostly about common methods of **plotting** data, including:

- Histograms  
- Scatterplots  
- Line plots
- Bar plots

### Why is data visualization important?

- Exploratory data analysis
- Communicating insights
- Impacting the world

### Exploratory Data Analysis: Checking your assumptions 

[Anscombe's Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)

![title](img/anscombe.png)

### Communicating Insights

[Reference: Full Stack Economics](https://fullstackeconomics.com/18-charts-that-explain-the-american-economy/)

![title](img/work.png)

### Impacting the world

[Florence Nightingale](https://en.wikipedia.org/wiki/Florence_Nightingale) (1820-1910) was a social reformer, statistician, and founder of modern nursing.

![title](img/polar.jpeg)

### Impacting the world (pt. 2)

[John Snow](https://en.wikipedia.org/wiki/John_Snow) (1813-1858) was a physician whose visualization of cholera outbreaks helped identify the source and spreading mechanism (water supply). 

![title](img/cholera.jpeg)

## Introducing `matplotlib`

### Loading packages

Here, we load the core packages we'll be using. 

We also add some lines of code that make sure our visualizations will plot "inline" with our code, and that they'll have nice, crisp quality.

```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
```

```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```

### What is `matplotlib`?

> [`matplotlib`](https://matplotlib.org/) is a **plotting library** for Python.

- Many [tutorials](https://matplotlib.org/stable/tutorials/index.html) available online.  
- Also many [examples](https://matplotlib.org/stable/gallery/index) of `matplotlib` in use.

Note that [`seaborn`](https://seaborn.pydata.org/) (which we'll cover soon) uses `matplotlib` "under the hood".

### What is `pyplot`?

> [`pyplot`](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) is a collection of functions *within* `matplotlib` that make it really easy to plot data.

With `pyplot`, we can easily plot things like:

- Histograms (`plt.hist`)
- Scatterplots (`plt.scatter`)
- Line plots (`plt.plot`) 
- Bar plots (`plt.bar`)

### Example dataset

Let's load our familiar Pokemon dataset, which can be found in `data/pokemon.csv`.

```python
df_pokemon = pd.read_csv("pokemon.csv")
df_pokemon.head(3)
```

## Histograms

### What are histograms?

> A **histogram** is a visualization of a single continuous, quantitative variable (e.g., income or temperature). 

- Histograms are useful for looking at how a variable **distributes**.  
- Can be used to determine whether a distribution is **normal**, **skewed**, or **bimodal**.

A histogram is a **univariate** plot, i.e., it displays only a single variable.

### Histograms in `matplotlib`

To create a histogram, call `plt.hist` with a **single column** of a `DataFrame` (or a `numpy.ndarray`).

**Check-in**: What is this graph telling us?

```python
p = plt.hist(df_pokemon['Attack'])
```

#### Changing the number of bins

A histogram puts your continuous data into **bins** (e.g., 1-10, 11-20, etc.).

- The height of each bin reflects the number of observations within that interval.  
- Increasing or decreasing the number of bins gives you more or less granularity in your distribution.

```python
### This has lots of bins
p = plt.hist(df_pokemon['Attack'], bins = 30)
```

```python
### This has fewer bins
p = plt.hist(df_pokemon['Attack'], bins = 5)
```

#### Changing the `alpha` level

The `alpha` level changes the **transparency** of your figure.

```python
### This has fewer bins
p = plt.hist(df_pokemon['Attack'], alpha = .6)
```

#### Check-in:

How would you make a histogram of the scores for `Defense`?

```python
plt.hist(df_pokemon['Defense'], alpha=0.6)
plt.xlabel("Defense")
plt.ylabel("Count")
plt.title("Distribution of Defense Scores")

```

#### Check-in:

Could you make a histogram of the scores for `Type 1`?

```python
df_pokemon['Type 1'].value_counts().plot(kind='bar', alpha=0.6)
plt.xlabel("Type 1")
plt.ylabel("Count")
plt.title("Number of Pokémon per Type 1")

```

### Learning from histograms

Histograms are incredibly useful for learning about the **shape** of our distribution. We can ask questions like:

- Is this distribution relatively [normal](https://en.wikipedia.org/wiki/Normal_distribution)?
- Is the distribution [skewed](https://en.wikipedia.org/wiki/Skewness)?
- Are there [outliers](https://en.wikipedia.org/wiki/Outlier)?

#### Normally distributed data

We can use the `numpy.random.normal` function to create a **normal distribution**, then plot it.

A normal distribution has the following characteristics:

- Classic "bell" shape (**symmetric**).  
- Mean, median, and mode are all identical.

```python
norm = np.random.normal(loc = 10, scale = 1, size = 1000)
p = plt.hist(norm, alpha = .6)
```

#### Skewed data

> **Skew** means there are values *elongating* one of the "tails" of a distribution.

- Positive/right skew: the tail is pointing to the right.  
- Negative/left skew: the tail is pointing to the left.

```python
rskew = ss.skewnorm.rvs(20, size = 1000) # make right-skewed data
lskew = ss.skewnorm.rvs(-20, size = 1000) # make left-skewed data
fig, axes = plt.subplots(1, 2)
axes[0].hist(rskew)
axes[0].set_title("Right-skewed")
axes[1].hist(lskew)
axes[1].set_title("Left-skewed")
```

#### Outliers

> **Outliers** are data points that differ significantly from other points in a distribution.

- Unlike skewed data, outliers are generally **discontinuous** with the rest of the distribution.
- Next week, we'll talk about more ways to **identify** outliers; for now, we can rely on histograms.

```python
norm = np.random.normal(loc = 10, scale = 1, size = 1000)
upper_outliers = np.array([21, 21, 21, 21]) ## some random outliers
data = np.concatenate((norm, upper_outliers))
p = plt.hist(data, alpha = .6)
plt.arrow(20, 100, dx = 0, dy = -50, width = .3, head_length = 10, facecolor = "red")
```

#### Check-in

How would you describe the following distribution?

- Normal vs. skewed?  
- With or without outliers?

- The distribution is right-skewed.
- It contains clear outliers on the right side.
- The mean is greater than the median, which confirms the skewness.


#### Check-in

In a somewhat **right-skewed distribution** (like below), what's larger––the `mean` or the `median`?

```python



plt.hist(data, alpha=0.6)
plt.title("Distribution with Outliers")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.xlim(5, 15) 

mean1 = np.mean(data)
median1 = np.median(data)


print("Mean:", mean1)
print("Median:", median1)

plt.axvline(mean1, color='blue',  label='Mean')
plt.axvline(median1, color='green',  label='Median')
plt.legend()
```

### Modifying our plot

- A good data visualization should also make it *clear* what's being plotted.
   - Clearly labeled `x` and `y` axes, title.
- Sometimes, we may also want to add **overlays**. 
   - E.g., a dashed vertical line representing the `mean`.

#### Adding axis labels

```python
p = plt.hist(df_pokemon['Attack'], alpha = .6)
plt.xlabel("Attack")
plt.ylabel("Count")
plt.title("Distribution of Attack Scores")
```

#### Adding a vertical line

The `plt.axvline` function allows us to draw a vertical line at a particular position, e.g., the `mean` of the `Attack` column.

```python
p = plt.hist(df_pokemon['Attack'], alpha = .6)
plt.xlabel("Attack")
plt.ylabel("Count")
plt.title("Distribution of Attack Scores")
plt.axvline(df_pokemon['Attack'].mean(), linestyle = "dotted")
```

## Scatterplots

### What are scatterplots?

> A **scatterplot** is a visualization of how two different continuous distributions relate to each other.

- Each individual point represents an observation.
- Very useful for **exploratory data analysis**.
   - Are these variables positively or negatively correlated?
   
A scatterplot is a **bivariate** plot, i.e., it displays at least two variables.

### Scatterplots with `matplotlib`

We can create a scatterplot using `plt.scatter(x, y)`, where `x` and `y` are the two variables we want to visualize.

```python
x = np.arange(1, 10)
y = np.arange(11, 20)
p = plt.scatter(x, y)
```

#### Check-in

Are these variables related? If so, how?

```python
x = np.random.normal(loc = 10, scale = 1, size = 100)
y = x * 2 + np.random.normal(loc = 0, scale = 2, size = 100)
plt.scatter(x, y, alpha = .6);

correlation = np.corrcoef(x, y)[0, 1]
print("Correlation coefficient:", correlation)
```

Yes, the variables are related.  
There is a **strong positive linear relationship** between `x` and `y`.  
The correlation coefficient is close to 1, which means as `x` increases, `y` also increases.


#### Check-in

Are these variables related? If so, how?

```python
x = np.random.normal(loc = 10, scale = 1, size = 100)
y = -x * 2 + np.random.normal(loc = 0, scale = 2, size = 100)
plt.scatter(x, y, alpha = .6);

correlation = np.corrcoef(x, y)[0, 1]
print("Correlation coefficient:", correlation)
```

Yes, the variables are related.

There is a **strong negative linear relationship** between `x` and `y`.  
The correlation coefficient is close to **-1**, which means that as `x` increases, `y` tends to decrease.


#### Scatterplots are useful for detecting non-linear relationships

```python
x = np.random.normal(loc = 10, scale = 1, size = 100)
y = np.sin(x)
plt.scatter(x, y, alpha = .6);



```

#### Check-in

How would we visualize the relationship between `Attack` and `Speed` in our Pokemon dataset?

```python
x = df_pokemon["Attack"]
y = df_pokemon["Speed"]

plt.scatter(x, y, alpha=0.6)
plt.title("Attack vs Speed")
plt.xlabel("Attack")
plt.ylabel("Speed")
```

We can visualize the relationship between `Attack` and `Speed` using a **scatter plot**.  
Each point represents one Pokémon. This helps us see if there's a pattern or correlation between the two stats.


## Barplots

### What is a barplot?

> A **barplot** visualizes the relationship between one *continuous* variable and a *categorical* variable.

- The *height* of each bar generally indicates the mean of the continuous variable.
- Each bar represents a different *level* of the categorical variable.

A barplot is a **bivariate** plot, i.e., it displays at least two variables.

### Barplots with `matplotlib`

`plt.bar` can be used to create a **barplot** of our data.

- E.g., average `Attack` by `Legendary` status.
- However, we first need to use `groupby` to calculate the mean `Attack` per level.

#### Step 1: Using `groupby`

```python
summary = df_pokemon[['Legendary', 'Attack']].groupby("Legendary").mean().reset_index()
summary
```

```python
### Turn Legendary into a str
summary['Legendary'] = summary['Legendary'].apply(lambda x: str(x))
summary
```

#### Step 2: Pass values into `plt.bar`

**Check-in**:

- What do we learn from this plot?  
- What is this plot missing?

```python
plt.bar(x = summary['Legendary'],height = summary['Attack'],alpha = .6);
plt.xlabel("Legendary status");
plt.ylabel("Attack");


plt.title("Average Attack by Legendary Status")

```

- We learn that **Legendary Pokémon** tend to have a **higher average Attack** than non-Legendary ones.
- However, this plot is **missing a title**, which would make it clearer what the bars represent.


## Conclusion

This concludes our first introduction to **data visualization**:

- Working with `matplotlib.pyplot`.  
- Creating basic plots: histograms, scatterplots, and barplots.

Next time, we'll move onto discussing `seaborn`, another very useful package for data visualization.


---

# Exercise 6

# Data visualization, pt. 2 (`seaborn`)

## Goals of this exercise

- Introducting `seaborn`. 
- Putting `seaborn` into practice:
  - **Univariate** plots (histograms).  
  - **Bivariate** continuous plots (scatterplots and line plots).
  - **Bivariate** categorical plots (bar plots, box plots, and strip plots).

## Introducing `seaborn`

### What is `seaborn`?

> [`seaborn`](https://seaborn.pydata.org/) is a data visualization library based on `matplotlib`.

- In general, it's easier to make nice-looking graphs with `seaborn`.
- The trade-off is that `matplotlib` offers more flexibility.

```python
import seaborn as sns ### importing seaborn
import pandas as pd
import matplotlib.pyplot as plt ## just in case we need it
import numpy as np
```

```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```

### The `seaborn` hierarchy of plot types

We'll learn more about exactly what this hierarchy means today (and in next lecture).

![title](img/seaborn_library.png)

### Example dataset

Today we'll work with a new dataset, from [Gapminder](https://www.gapminder.org/data/documentation/). 

- **Gapminder** is an independent Swedish foundation dedicated to publishing and analyzing data to correct misconceptions about the world.
- Between 1952-2007, has data about `life_exp`, `gdp_cap`, and `population`.

```python
df_gapminder = pd.read_csv("gapminder_full.csv")
```

```python
df_gapminder.head(2)
```

```python
df_gapminder.shape
```

## Univariate plots

> A **univariate plot** is a visualization of only a *single* variable, i.e., a **distribution**.

![title](img/displot.png)

### Histograms with `sns.histplot`

- We've produced histograms with `plt.hist`.  
- With `seaborn`, we can use `sns.histplot(...)`.

Rather than use `df['col_name']`, we can use the syntax:

```python
sns.histplot(data = df, x = col_name)
```

This will become even more useful when we start making **bivariate plots**.

```python
# Histogram of life expectancy
sns.histplot(df_gapminder['life_exp']);
```

#### Modifying the number of bins

As with `plt.hist`, we can modify the number of *bins*.

```python
# Fewer bins
sns.histplot(data = df_gapminder, x = 'life_exp', bins = 10, alpha = .6);
```

```python
# Many more bins!
sns.histplot(data = df_gapminder, x = 'life_exp', bins = 100, alpha = .6);
```

#### Modifying the y-axis with `stat`

By default, `sns.histplot` will plot the **count** in each bin. However, we can change this using the `stat` parameter:

- `probability`: normalize such that bar heights sum to `1`.
- `percent`: normalize such that bar heights sum to `100`.
- `density`: normalize such that total *area* sums to `1`.


```python
# Note the modified y-axis!
sns.histplot(data = df_gapminder, x = 'life_exp', stat = "percent", alpha = .6);
```

### Check-in

How would you make a histogram showing the distribution of `population` values in `2007` alone? 

- Bonus 1: Modify this graph to show `probability`, not `count`.
- Bonus 2: What do you notice about this graph, and how might you change it?

```python

df_2007 = df_gapminder[df_gapminder['year'] == 2007]

sns.histplot(data=df_2007, x='population', stat='probability')
plt.title('Distribution de la population en 2007 (probabilité)')
plt.show()


```

The histogram shows that the population distribution in 2007 is highly skewed.
Most countries have a small population, while only a few countries (such as China and India) have an extremely large population.

As a result:

Most of the data is compressed near zero on the x-axis,

The few very large countries dominate the upper end of the graph.

To improve the visualization, we can use a logarithmic scale on the x-axis.
This spreads out the data and makes the distribution easier to interpret.

```python
sns.histplot(data=df_2007, x='population', stat='probability')
plt.xscale('log')
plt.title('Distribution of Population in 2007 (log scale)')
plt.show()

```

## Bivariate continuous plots

> A **bivariate continuous plot** visualizes the relationship between *two continuous variables*.

![title](img/seaborn_relplot.png)

### Scatterplots with `sns.scatterplot`

> A **scatterplot** visualizes the relationship between two continuous variables.

- Each observation is plotted as a single dot/mark. 
- The position on the `(x, y)` axes reflects the value of those variables.

One way to make a scatterplot in `seaborn` is using `sns.scatterplot`.

#### Showing `gdp_cap` by `life_exp`

What do we notice about `gdp_cap`?

```python
sns.scatterplot(data = df_gapminder, x = 'gdp_cap',
               y = 'life_exp', alpha = .3);
```

#### Showing `gdp_cap_log` by `life_exp`

```python
## Log GDP
df_gapminder['gdp_cap_log'] = np.log10(df_gapminder['gdp_cap']) 
## Show log GDP by life exp
sns.scatterplot(data = df_gapminder, x = 'gdp_cap_log', y = 'life_exp', alpha = .3);
```

#### Adding a `hue`

- What if we want to add a *third* component that's categorical, like `continent`?
- `seaborn` allows us to do this with `hue`.

```python
## Log GDP
df_gapminder['gdp_cap_log'] = np.log10(df_gapminder['gdp_cap']) 
## Show log GDP by life exp
sns.scatterplot(data = df_gapminder[df_gapminder['year'] == 2007],
               x = 'gdp_cap_log', y = 'life_exp', hue = "continent", alpha = .7);
```

#### Adding a `size`

- What if we want to add a *fourth* component that's continuous, like `population`?
- `seaborn` allows us to do this with `size`.

```python
## Log GDP
df_gapminder['gdp_cap_log'] = np.log10(df_gapminder['gdp_cap']) 
## Show log GDP by life exp
sns.scatterplot(data = df_gapminder[df_gapminder['year'] == 2007],
               x = 'gdp_cap_log', y = 'life_exp',
                hue = "continent", size = 'population', alpha = .7);
```

#### Changing the position of the legend

```python
## Show log GDP by life exp
sns.scatterplot(data = df_gapminder[df_gapminder['year'] == 2007],
               x = 'gdp_cap_log', y = 'life_exp',
                hue = "continent", size = 'population', alpha = .7);

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0);
```

### Lineplots with `sns.lineplot`

> A **lineplot** also visualizes the relationship between two continuous variables.

- Typically, the position of the line on the `y` axis reflects the *mean* of the `y`-axis variable for that value of `x`.
- Often used for plotting **change over time**.

One way to make a lineplot in `seaborn` is using [`sns.lineplot`](https://seaborn.pydata.org/generated/seaborn.lineplot.html).

#### Showing `life_exp` by `year`

What general trend do we notice?

```python
sns.lineplot(data = df_gapminder,
             x = 'year',
             y = 'life_exp');
```

#### Modifying how error/uncertainty is displayed

- By default, `seaborn.lineplot` will draw **shading** around the line representing a confidence interval.
- We can change this with `errstyle`.

```python
sns.lineplot(data = df_gapminder,
             x = 'year',
             y = 'life_exp',
            err_style = "bars");
```

#### Adding a `hue`

- We could also show this by `continent`.  
- There's (fortunately) a positive trend line for each `continent`.

```python
sns.lineplot(data = df_gapminder,
             x = 'year',
             y = 'life_exp',
            hue = "continent")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0);
```

#### Check-in

How would you plot the relationship between `year` and `gdp_cap` for countries in the `Americas` only?

```python
sns.set_style('whitegrid')


df_americas = df_gapminder[df_gapminder['continent'] == 'Americas']


plt.figure(figsize=(10,6)) 
scatter = sns.scatterplot(data=df_americas, x='year', y='gdp_cap', label='Individual countries')

line = sns.lineplot(data=df_americas, x='year', y='gdp_cap', ci=None, estimator='mean', color='red', label='Average trend')

plt.title('GDP per Capita over Time (Americas)')
plt.xlabel('Year')
plt.ylabel('GDP per Capita')
plt.legend()
plt.show()


```

#### Heteroskedasticity in `gdp_cap` by `year`

- [**Heteroskedasticity**](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity) is when the *variance* in one variable (e.g., `gdp_cap`) changes as a function of another variable (e.g., `year`).
- In this case, why do you think that is?

#### Plotting by country

- There are too many countries to clearly display in the `legend`. 
- But the top two lines are the `United States` and `Canada`.
   - I.e., two countries have gotten much wealthier per capita, while the others have not seen the same economic growth.

```python
sns.lineplot(data = df_gapminder[df_gapminder['continent']=="Americas"],
             x = 'year', y = 'gdp_cap', hue = "country", legend = None);
```

### Using `relplot`

- `relplot` allows you to plot either line plots or scatter plots using `kind`.
- `relplot` also makes it easier to `facet` (which we'll discuss momentarily).

```python
sns.relplot(data = df_gapminder, x = "year", y = "life_exp", kind = "line");
```

#### Faceting into `rows` and `cols`

We can also plot the same relationship across multiple "windows" or **facets** by adding a `rows`/`cols` parameter.

```python
sns.relplot(data = df_gapminder, x = "year", y = "life_exp", kind = "line", 
            col = "continent");
```

## Bivariate categorical plots

> A **bivariate categorical plot** visualizes the relationship between one categorical variable and one continuous variable.

![title](img/seaborn_catplot.png)

### Example dataset

Here, we'll return to our Pokemon dataset, which has more examples of categorical variables.

```python
df_pokemon = pd.read_csv("pokemon.csv")
```

### Barplots with `sns.barplot`

> A **barplot** visualizes the relationship between one *continuous* variable and a *categorical* variable.

- The *height* of each bar generally indicates the mean of the continuous variable.
- Each bar represents a different *level* of the categorical variable.

With `seaborn`, we can use the function `sns.barplot`.

#### Average `Attack` by `Legendary` status

```python
sns.barplot(data = df_pokemon,
           x = "Legendary", y = "Attack");
```

#### Average `Attack` by `Type 1`

Here, notice that I make the figure *bigger*, to make sure the labels all fit.

```python
plt.figure(figsize=(15,4))
sns.barplot(data = df_pokemon,
           x = "Type 1", y = "Attack");
```

#### Check-in

How would you plot `HP` by `Type 1`?

```python

sns.set_style('whitegrid')


sns.barplot(data=df_pokemon, x='Type 1', y='HP')


plt.xticks(rotation=45, ha='right') 


plt.title('Average HP by Pokemon Type')
plt.xlabel('Pokemon Type')
plt.ylabel('Average HP')


plt.show()

```

#### Modifying `hue`

As with `scatterplot` and `lineplot`, we can change the `hue` to give further granularity.

- E.g., `HP` by `Type 1`, further divided by `Legendary` status.

```python
plt.figure(figsize=(15,4))
sns.barplot(data = df_pokemon,
           x = "Type 1", y = "HP", hue = "Legendary");
```

### Using `catplot`

> `seaborn.catplot` is a convenient function for plotting bivariate categorical data using a range of plot types (`bar`, `box`, `strip`).

```python
sns.catplot(data = df_pokemon, x = "Legendary", 
             y = "Attack", kind = "bar");
```

#### `strip` plots

> A `strip` plot shows each individual point (like a scatterplot), divided by a **category label**.

```python
sns.catplot(data = df_pokemon, x = "Legendary", 
             y = "Attack", kind = "strip", alpha = .5);
```

#### Adding a `mean` to our `strip` plot

We can plot *two graphs* at the same time, showing both the individual points and the means.

```python
sns.catplot(data = df_pokemon, x = "Legendary", 
             y = "Attack", kind = "strip", alpha = .1)
sns.pointplot(data = df_pokemon, x = "Legendary", 
             y = "Attack", hue = "Legendary");
```

#### `box` plots

> A `box` plot shows the interquartile range (the middle 50% of the data), along with the minimum and maximum.

A typical **boxplot** contains several components that are part of its anatomy:

- Median: This is the middle value of the data, represented by a line in the boxplot.
- Boxes: These represent the interquartile range (IQR) of the data, which represents the range between Q1 and Q3. The bottom and top edges represent Q1 and Q3, respectively.
- Whiskers: These are vertical lines that extend from both ends of the box to represent the minimum and maximum values, excluding outliers.
- Outliers: These are points outside the whiskers that are considered abnormal or extreme compared to the rest of the data.
- Limiters: These are the horizontal lines at the ends of the whiskers, representing minimum and maximum values, including any outliers.

**Do the whiskers show the minimum and maximum?**

From a statistical point of view - the ends of the whiskers are therefore not min and max - because they do not contain part of the outliers. 

*Standard rules:*

- **Outliers:** These are points outside the whiskers' range. These values are considered abnormal compared to the rest of the data.
- **Whiskers range:** Whiskers extend to values that are within limits:
  - Lower whisker: $Q1 - 1.5 * IQR$
  - Upper whisker: $Q3 + 1.5 * IQR$
- **Extreme outliers:** If the values are much further outside the whiskers (e.g., $Q1 - 3 * IQR$ or $Q3 + 3 * IQR$), they may be considered extreme outliers.

**Why are outliers important?**

- May indicate errors in data (e.g., typos, measurement errors).
- They may represent real but rare events that are worth investigating.
- Outliers can significantly affect descriptive statistics, such as the mean, so their identification is crucial in data analysis.

**Why 1.5 × IQR?**

$1.5$ × $IQR$ is the standard value used to identify moderate outliers.

The interquartile range (IQR) is the difference between the third quartile (Q3) and the first quartile (Q1), the range within which the middle 50% of the data is located.

Values that fall outside the range:

- Lower threshold: $Q1$ - $1.5$ × $IQR$
- Upper threshold: $Q3$ + $1.5$ × $IQR$ are considered outliers.

The $1.5$ value was empirically chosen as a reasonable compromise between detecting outliers and ignoring natural fluctuations in the data.

**Why 3 × IQR?**

$3$ × $IQR$ is used to identify extreme outliers that are much more distant from the rest of the data.

Outlier values:

- Lower threshold: $Q1$ - $3$ × $IQR$
- Upper threshold: $Q3$ + $3$ × $IQR$ are considered extreme outliers.

The $3$ × $IQR$ value is more stringent and identifies points that are highly unusual and may indicate data errors or rare events.

```python
sns.catplot(data = df_pokemon, x = "Legendary", 
             y = "Attack", kind = "box");
```

Try to consider converting the boxplots into violin plots.

## Conclusion

As with our lecture on `pyplot`, this just scratches the surface.

But now, you've had an introduction to:

- The `seaborn` package.
- Plotting both **univariate** and **bivariate** data.
- Creating plots with multiple layers.

