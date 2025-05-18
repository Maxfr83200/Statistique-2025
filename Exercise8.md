# Univariate Analysis

---

## Looking ahead: April Week 4, May Week 1

- In the end of April and early May, we'll dive deep into **statistics** finally.  
  - How do we calculate descriptive statistics in Python?
  - What principles should we keep in mind?

Univariate analysis is a type of statistical analysis that involves examining the distribution and characteristics of a single variable. The prefix “uni-” means “one,” so univariate analysis focuses on one variable at a time, without considering relationships between variables.

Univariate analysis is the foundation of data analysis and is essential for understanding the basic structure of your data before moving on to more complex techniques like bivariate or multivariate analysis.

---

# Measurement scales

Measurement scales determine what mathematical and statistical operations can be performed on data. There are four basic types of scales:

1. **Nominal** scale
- Data is used only for naming or categorizing.
- The order between values cannot be determined.
- Possible operations: count, mode, frequency analysis.

Examples:
- Pokémon type (type_1): “fire”, ‘water’, ‘grass’, etc.
- Species, gender, colors, brands etc.

---

2. **Ordinal** scale
- Data can be ordered, but the distances between them are not known.
- Possible operations: median, quantiles, rank tests (e.g. Spearman).

---

Examples:
- Strength level: "low", "medium", "high".
- Quality ratings: "weak", "good", "very good".

---

3. **Interval** scale
- The data is numerical, with equal intervals, but lacks an absolute zero.
- Differences, mean, and standard deviation can be calculated.
- Ratios (e.g., "twice as much") do not make sense.

Examples:
- Temperature in °C (but not in Kelvin!). Why? There is no absolute zero—zero does not mean the absence of the property; it is just a conventional reference point. 0°C does not mean no temperature; 20°C is not 2 × 10°C.
- Year in a calendar (e.g., 1990). Why? Year 0 does not mark the beginning of time; 2000 is not 2 × 1000.
- Time in the hourly system (e.g., 13:00). Why? 0:00 does not mean no time, but rather an established reference point.

4. **Ratio** scale
- Numerical data with an absolute zero.
- All mathematical operations, including division, can be performed.
  
> **Not all numerical data is on a ratio scale!** For example, temperature in degrees Celsius is not on a ratio scale because 0°C does not mean the absence of temperature. However, temperature in Kelvin (K) is, as 0 K represents the absolute absence of thermal energy.

Examples:
- Height, weight, number of Pokémon attack points (attack), HP, speed.

---

### Table: Measurement scales in statistics

| Scale          | Example                           | Is it possible to order? | Equal spacing? | Absolute zero? | Sample statistical calculations       |
|----------------|-------------------------------------|--------------------------|----------------|------------------|------------------------------------------|
| **Nominal**  | Pokémon type (`fire`, `water` etc.)| ❌                       | ❌             | ❌               | Mode, counts, frequency analysis      |
| **Ordinal** | Ticket class (`First`, `Second`, `Third`) | ✅                       | ❌             | ❌               | Median, quantiles         |
| **Interval** | Temperature in °C                  | ✅                       | ✅             | ❌               | Mean, standard deviation         |
| **Ratio**  | HP, attack, height                   | ✅                       | ✅             | ✅               | All mathematical operations/statistical |

---

**Conclusion**: The type of scale affects the choice of statistical methods - for example, the Pearson correlation test requires quotient or interval data, while the Chi² test requires nominal data.

---

![title](img/scales.jpg)

---

### Quiz: measurement scales in statistics.

Answer the following questions by choosing **one correct answer**. You will find the solutions at the end.

---

#### 1. Which scale **enables ordering of data**, but **does not have equal spacing**?
- A) Nominal  
- B) Ordinal <- this
- C) Interval  
- D) Ratio  

---

#### 2. An example of a variable on the **nominal scale** is:
- A) Temperature in °C  
- B) Height  
- C) Type of Pokémon (`fire`, `grass`, `water`)  <- this
- D) Satisfaction level (`low`, `medium`, `high`).  

---

#### 3. Which scale **does not have absolute zero**, but has **equal spacing**?
- A) Ratio  
- B) Ordinal  
- C) Interval  <- this
- D) Nominal  

---

#### 4. What operations are **allowed** on variables **on an ordinal scale**?
- A) Mean and standard deviation  
- B) Mode and Pearson correlation  
- C) Median and rank tests  <- this
- D) Quotients and logarithms  

---

#### 5. The variable `“class”` in the Titanic set (`First`, `Second`, `Third`) is an example:
- A) Nominal scale  
- B) Ratio scale  
- C) Interval scale  
- D) Ordinal scale  <- this

---

---

# Descriptive statistics

**Descriptive statistics** deals with the description of the distribution of data in a sample. Descriptive statistics give us basic summary measures about a set of data. Summary measures include measures of central tendency (mean, median and mode) and measures of variability (variance, standard deviation, minimum/maximum values, IQR (interquartile range), skewness and kurtosis).

---

## This week

Now we're going to look at **describing** our data - as well as the **basics of statistics**.

There are many ways to *describe* a distribution. 

Here we will discuss:
- Measures of **central tendency**: what is the typical value in this distribution?
- Measures of **variability**: how much do the values differ from each other?  
- Measures of **skewness**: how strong is the asymmetry of the distribution?
- Measures of **curvature**: what is the intensity of extreme values?

---

## Central tendency

The **central tendency** refers to the “typical value” in a distribution.

The **central tendency** refers to the central value that describes the distribution of a variable. It can also be referred to as the center or location of the distribution. The most common measures of central tendency are **average**, **median** and **mode**. The most common measure of central tendency is the **mean**. In the case of skewed distributions or when there is concern about outliers, the **median** may be preferred. The median is thus a more reliable measure than the mean.

There are many ways to *measure* what is “typical” - average:

- Arithmetic mean
- Median (middle value)
- Mode (dominant)

---

### Why is this useful?

- A dataset may contain *many* observations.  
   - For example, $N$ = $5000$ of survey responses regarding `height'.  
- One way to “describe” this distribution is to **visualize** it.  
- But it is also helpful to reduce this distribution to a *single number*.

This is necessarily a **simplification** of our dataset!

---

### *Arithmetic average*

> **Arithmetic average** is defined as the `sum` of all values in a distribution, divided by the number of observations in that distribution.

---

- The most common measure of central tendency is the average.
- The mean is also known as the simple average.
- It is denoted by the Greek letter $µ$ for a population and $\bar{x}$ for a sample.
- We can find the average of the number of elements by adding all the elements in the data set and then dividing by the number of elements in the data set.
- This is the most popular measure of central tendency, but it has a drawback.
- The average is affected by the presence of outliers.
- Thus, the average alone is not sufficient for making business decisions.

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$



---

#### `numpy.mean`

The `numpy` package has a function that calculates an `average` on a `list` or `numpy.ndarray`.

---

#### `scipy.stats.tmean`

The [scipy.stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html) library has a variety of statistical functions.

---

#### Calculating the `average` of a `pandas` column.

If we work with `DataFrame`, we can calculate the `average` of specific columns.

---

#### Your turn

How to calculate the mean life expectancy for EUROPEan countries (2007).

---

#### *Average* and skewness

> **Skewness** means that there are values *extending* one of the “tails” of the distribution.

Of the measures of **central tendency**, “average” is the most dependent on the direction of skewness.

- How would you describe the following **skewness**?  
- Do you think the “mean” would be higher or lower than the “median”?

---

#### Your turn

Is it possible to calculate the average of the column “continent”? Why or why not?

---

No, it is not possible to calculate the average of the column continent.

 The variable continent is qualitative nominal, which means it represents categories like Europe, Asia, etc., without any numerical meaning or ordering.

Mathematical operations like mean or standard deviation cannot be applied to nominal data. The only valid statistics for such data are counts, mode, or frequency distributions.


---

#### Your turn

- Subtract each observation in `numbers` from the `average` of this `list`.  
- Then calculate the **sum** of these deviations from the `average`.

What is their sum?

---

#### Summary of the first part

- The mean is one of the most common measures of central tendency.  
- It can only be used for **continuous** interval/ratio data.  
- The **sum of deviations** from the mean is equal to `0`. 
- The “mean” is most affected by **skewness** and **outliers**.

---

### *Median*

> *Median* is calculated by sorting all values from smallest to largest and then finding the value in the middle.

- The median is the number that divides a data set into two equal halves.
- To calculate the median, we need to sort our data set of n numbers in ascending order.
- The median of this data set is the number in the position $(n+1)/2$ if $n$ is odd.
- If n is even, the median is the average of the $(n/2)$ third number and the $(n+2)/2$ third number.
- The median is robust to outliers.
- Thus, in the case of skewed distributions or when there is concern about outliers, the median may be preferred.

---

#### Comparison of `median` and `average`.

The direction of inclination has less effect on the `median`.

---

#### Your turn

Is it possible to calculate the median of the column “continent”? Why or why not?

---

No, it is not possible to calculate the median of the column continent.

The variable continent is a nominal categorical variable it represents names of categories (like "Asia", "Europe", etc.) with no inherent order.

The median requires a meaningful order to determine a middle value, which is only possible with ordinal, interval, or ratio data. Since continents are unordered categories, computing a median is meaningless.


---

### *Mode*

> **Mode** is the most common value in a data set. 

Unlike `median` or `average`, `mode` can be used with **categorical** data.

---

#### `mode()` returns multiple values?

- If multiple values *bind* for the most frequent one, `mode()` will return them all.
- This is because technically, a distribution can have multiple values for the most frequent - modal!

---

### Measures of central tendency - summary

|Measure|Can be used for:|Limitations|
|-------|----------------|-----------|
|Mean|Continuous data|Influence on skewness and outliers|
|Median|Continuous data|Does not include the *value* of all data points in the calculation (ranks only)|
|Mode|Continuous and categorical data|Considers only *frequent*; ignores other values|

---

## Quantiles

**Quantiles** are descriptive - positional statistics that divide an ordered data set into equal parts. The most common quantiles are:

- **Median** (quantile of order 0.5),
- **Quartiles** (divide the data into 4 parts),
- **Deciles** (into 10 parts),
- **Percentiles** (into 100 parts).

### Definition

A quantile of order $q \in (0,1)$ is a value of $x_q$ such that:

$$
P(X \leq x_q) = q
$$

In other words: $q \cdot 100\%$ of the values in the data set are less than or equal to $x_q$.

### Formula (for an ordered data set)

For a data sample $x_1, x_2, \ldots, x_n$ ordered in ascending order, the quantile of order $q$ is determined as:

1. Calculate the positional index:

$$
i = q \cdot (n + 1)
$$

2. If $i$ is an integer, then the quantile is $x_i$.

3. If $i$ is not integer, we interpolate linearly between adjacent values:

$$
x_q = x_{\lfloor i \rfloor} + (i - \lfloor i \rfloor) \cdot (x_{\lceil i \rceil} - x_{\lfloor i \rfloor})
$$

**Note:** In practice, different methods are used to determine quantiles - libraries such as NumPy or Pandas have different modes (e.g. `method='linear'`, `method='midpoint'`).

### Example - we calculate step by step:

For data:
$
[3, 7, 8, 5, 12, 14, 21, 13, 18]
$

1. We arrange the data in ascending order:

$
[3, 5, 7, 8, 12, 13, 14, 18, 21]
$

2. Median (quantile of order 0.5):

The number of elements $n = 9$, the middle element is the 5th value:

$
\text{Median} = x_5 = 12
$

3. First quartile (Q1, quantile of order 0.25):

$
i = 0.25 \cdot (9 + 1) = 2.5
$

Interpolation between $x_2 = 5$ and $x_3 = 7$:

$
Q_1 = 5 + 0.5 \cdot (7 - 5) = 6
$

4. Third quartile (Q3, quantile of 0.75):

$
i = 0.75 \cdot 10 = 7.5
$

Interpolation between $x_7 = 14$ and $x_8 = 18$:

$
Q_3 = 14 + 0.5 \cdot (18 - 14) = 16
$

### Deciles

**Deciles** divide data into 10 equal parts. For example:

- **D1** is the 10th percentile (quantile of 0.1),
- **D5** is the median (0.5),
- **D9** is the 90th percentile (0.9).

The formula is the same as for overall quantiles, just use the corresponding $q$. E.g. for D3:

$
q = \frac{3}{10} = 0.3
$

### Percentiles

**Percentiles** divide data into 100 equal parts. E.g.:

- **P25** = Q1,
- **P50** = median,
- **P75** = Q3,
- **P90** is the value below which 90% of the data is.

With percentiles, we can better understand the distribution of data - for example, in standardized tests, a score is often given as a percentile (e.g., “85th percentile” means that someone scored better than 85% of the population).

---

### Quantiles - summary

| Name     | Symbol | Quantile \( q \) | Meaning                          |
|-----------|--------|------------------|-------------------------------------|
| Q1        | Q1     | 0.25             | 25% of data ≤ Q1                     |
| Median   | Q2     | 0.5              | 50% of data ≤ Median                |
| Q3        | Q3     | 0.75             | 75% of data ≤ Q3                     |
| Decile 1   | D1     | 0.1              | 10% of data ≤ D1                     |
| Decile 9   | D9     | 0.9              | 90% of data ≤ D9                     |
| Percentile 95 | P95 | 0.95             | 95% of data ≤ P95                    |

---

---

### Example - calculations of quantiles

---

### Your turn!

Try to change the boxplot into the violin plot (or add it). 

Looking at the aforementioned quantile results and the box plot, try to interpret these measures. 

---

The violin plot shows the distribution of life expectancy for each continent, including the density of the values (unlike the boxplot which only shows quartiles and outliers).

We can observe:
- The spread (variability) is different across continents.
- For example, Africa has a lower median and wider spread, while Europe has a higher median and more concentrated values.
- This gives more insight than a boxplot alone, especially when distributions are skewed or multimodal.


---

## Variability

> **Variability** (or **dispersion**) refers to the degree to which values in a distribution are *dispersed*, i.e., differ from each other.

The **dispersion** is an indicator of how far from the center we can find data values. The most common measures of dispersion are **variance**, **standard deviation** and **interquartile range (IQR)**. The **variance** is a standard measure of dispersion. The **standard deviation** is the square root of the variance. The **variance** and **standard deviation** are two useful measures of scatter.

---

### The `mean` hides the variance!

Both distributions have *the same* mean, but *different* **standard deviations**.

---

### Volatility detection

There are at least *three* main approaches to quantifying variability:

- **Range**: the difference between the “maximum” and “minimum” value. 
- **Interquartile range (IQR)**: The range of the middle 50% of the data.  
- **Variance** and **Standard Deviation**: the typical value by which results deviate from the mean.

---

### Range

> **Range** Is the difference between the `maximum` and `minimum` values.

Intuitive, but only considers two values in the entire distribution.

---

### IQR

> The **interquartile range (IQR)** is the difference between a value in the 75% percentile and a value in the 25% percentile.

It focuses on the **center 50%**, but still only considers two values.

- IQR is calculated using the limits of the data between the 1st and 3rd quartiles. 
- The interquartile range (IQR) can be calculated as follows: $IQR = Q3 - Q1$
- In the same way that the median is more robust than the mean, the IQR is a more robust measure of scatter than the variance and standard deviation and should therefore be preferred for small or asymmetric distributions. 
- It is a robust measure of scatter.

---

### Variance and standard deviation.

The **Variance** measures the dispersion of a set of data points around their mean value. It is the average of the squares of the individual deviations. The variance gives the results in original units squared.

$$
s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

**Standard deviation (SD)** measures the *typical value* by which the results in the distribution deviate from the mean.

$$
s = \sqrt{s^2} = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

where:
	- $n$ - the number of elements in the sample
	- $\bar{x}$ - the arithmetic mean of the sample

What to keep in mind:

- SD is the *square root* of [variance](https://en.wikipedia.org/wiki/Variance).  
- There are actually *two* measures of SD:
 - SD of a population: when you measure the entire population of interest (very rare).  
   - SD of a sample: when you measure a *sample* (typical case); we'll focus on that.

---

#### SD, explained

- First, calculate the total *square deviation*.
   - What is the total square deviation from the “mean”? 
- Then divide by `n - 1`: normalize to the number of observations.
   - What is the *average* squared deviation from the `average'?
- Finally, take the *square root*:
   - What is the *average* deviation from the “mean”?

The **standard deviation** represents the *typical* or “average” deviation from the “mean”.

---

#### SD calculation in `pandas`

---

#### Note on `numpy.std`!!!

- By default, `numpy.std` calculates the **population standard deviation**!  
- You need to modify the `ddof` parameter to calculate the **sample standard deviation**.

This is a very common error.

---

### Coefficient of variation (CV).

- The coefficient of variation (CV) is equal to the standard deviation divided by the mean.
- It is also known as “relative standard deviation.”

$$
CV = \frac{s}{\bar{x}} \cdot 100%
$$

---

## Interquartile deviation

Interquartile deviation (sometimes called the semi-interquartile range) is defined as half of the interquartile range:

$$ \text{IQR deviation} = \frac{Q3 - Q1}{2} $$

This value shows the average distance from the median to the quartiles and is a robust measure of variability.

- A small interquartile deviation means the middle 50% of the data are close to the median.
- A large interquartile deviation means the middle 50% are more spread out.

It is less sensitive to outliers than the standard deviation or range!

---

# Your turn!

Calculate STD and CV for the SPEED of LEGENDARY and NOT LEGENDARY pokemons. What is the IQR deviation? 

---

The analysis of Pokémon speed by group shows clear differences:

Legendary Pokémon
- Average Speed: 100.18 — much higher on average.
- Standard Deviation (STD): 22.95 — some variability, but not extreme.
- Coefficient of Variation (CV): 0.23 — low relative dispersion, meaning Legendary Pokémon are not only faster but also more consistent.
- Interquartile Range (IQR): 20 — the middle 50% are quite concentrated.

Non-Legendary Pokémon
- Average Speed: 65.46 — much lower on average.
- STD: 27.84 — more spread out values.
- CV: 0.43 — higher relative variability, meaning more differences among regular Pokémon.
- IQR: 40 — more variation in typical speeds.

Conclusion: Legendary Pokémon are generally faster and have more consistent speed stats, while Non-Legendary ones show greater diversity.


---

## Measures of the shape of the distribution

Now we will look at measures of the shape of the distribution. There are two statistical measures that can tell us about the shape of a distribution. These are **skewness** and **curvature**. These measures can be used to tell us about the shape of the distribution of a data set.

---

## Skewness
- **Skewness** is a measure of the symmetry of a distribution, or more precisely, the lack of symmetry. 
- It is used to determine the lack of symmetry with respect to the mean of a data set. 
- It is a characteristic of deviation from the mean. 
- It is used to indicate the shape of a data distribution.

---

Skewness is a measure of the asymmetry of the distribution of data relative to the mean. It tells us whether the data are more ‘stretched’ to one side.

Interpretation:

- Skewness > 0 - right-tailed (positive): long tail on the right (larger values are more dispersed)
- Skewness < 0 - left (negative): long tail on the left (smaller values are more dispersed)
- Skewness ≈ 0 - symmetric distribution (e.g. normal distribution)

Formula (for the sample):

$$
A = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3
$$

where:
- $n$ - number of observations
- $\bar{x}$ - sample mean
- $s$ - standard deviation of the sample

---

![title](img/skew.png)

---


#### Negative skewness

- In this case, the data are skewed or shifted to the left. 
- By skewed to the left, we mean that the left tail is long relative to the right tail. 
- The data values may extend further to the left, but are concentrated on the right. 
- So we are dealing with a long tail, and the distortion is caused by very small values that pull the mean down and it is smaller than the median. 
- In this case we have **Mean < Median < Mode**.
      

#### Zero skewness

- This means that the dataset is symmetric. 
- A dataset is symmetric if it looks the same to the left and right of the midpoint. 
- A dataset is bell-shaped or symmetric. 
- A perfectly symmetrical dataset will have a skewness of zero. 
- So a normal distribution that is perfectly symmetric has a skewness of 0. 
- In this case we have **Mean = Median = Mode**.
      

#### Positive skewness

- The dataset is skewed or shifted to the right. 
- By skewed to the right we mean that the right tail is long relative to the left tail. 
- The data values are concentrated on the right side. 
- There is a long tail on the right side, which is caused by very large values that pull the mean upwards and it is larger than the median. 
- So we have **Mean > Median > Mode**.

---

### Your turn

Try to interpret the above-mentioned result and calculate example slant ratios for several groups of Pokémon.

---

Legendary Pokémon have a higher average speed and more concentrated values, so they are fast and consistent. Non-legendary Pokémon are slower on average and more spread out, with a wide range of speeds.

For the slant ratio, we can take groups like "Fire", "Water", etc., and use the formula:
(Q3 + Q1 - 2 × median) / IQR

This helps to see if the distribution is balanced or skewed to the left or right. A ratio close to zero means the distribution is symmetric.

---

### Interquartile Skewness

**IQR skewness** is a robust, non-parametric measure of skewness that uses the positions of the quartiles rather than the mean and standard deviation. It is particularly useful for detecting asymmetry in data distributions, especially when outliers are present.

The formula for IQR Skewness is:

$$
IQR\ Skewness = \frac{(Q3 - Median) - (Median - Q1)}{Q3 - Q1}
$$
This method is **less sensitive to outliers** and more **robust** than moment-based skewness, making it ideal for exploratory data analysis.

---

### Your turn

Try to calculate the IQR Skewness coefficient for the sample data:

---

## Kurtosis

Contrary to what some textbooks claim, kurtosis does not measure the ‘flattening’, the ‘peaking’ of a distribution.

> **Kurtosis** depends on the intensity of the extremes, so it measures what happens in the ‘tails’ of the distribution, the shape of the ‘top’ is irrelevant!

**Excess kurtosis** is just kurtosis minus 3. It’s used to compare a distribution to the normal distribution (which has kurtosis = 3).


Sample kurtosis:

$$
\text{Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4
$$

$$
\text{Normalized kurtosis} = \text{Kurtosis} - 3
$$

#### Reference range for kurtosis
- The reference standard is the normal distribution, which has a kurtosis of 3. 
- Often **Excess** is presented instead of kurtosis, where **excess** is simply **Kurtosis - 3**. 

#### Mesocurve
- A normal distribution has a kurtosis of exactly 3 (**Excess** exactly 0). 
- Any distribution with kurtosis $≈3$ (exces ≈ 0) is called **mezocurtic**.

#### Platykurtic curve
- A distribution with kurtosis $<3$ (**Excess** < 0) is called **platykurtic**. 
- Compared to a normal distribution, its central peak is lower and wider and its tails are shorter and thinner.

#### Leptokurtic curve

- A distribution with kurtosis $>3$ (**Excess** > 0) is called **leptocurtic**. 
- Compared to a normal distribution, its central peak is higher and sharper and its tails are longer and thicker.

---

![title](img/ku.png)

---

So:
- Excess Kurtosis ≈ 0 → Normal distribution
- Excess Kurtosis > 0 → Leptokurtic (heavy tails)
- Excess Kurtosis < 0 → Platykurtic (light tails)

---

### Interquartile Kurtosis

**IQR Kurtosis** is a robust, non-parametric measure of kurtosis that focuses on the tails of the distribution using interquartile ranges. It is particularly useful for detecting the intensity of extreme values in data distributions, especially when outliers are present.

The formula for IQR Kurtosis is:

$$
IQR\ Kurtosis = \frac{Q3 - Q1}{2*(C90 - C10)}
$$

Where:
- $Q1$ is the first quartile (25th percentile),
- $Q3$ is the third quartile (75th percentile),
- $C90$ is the 90th percentile,
- $C10$ is the 10th percentile.

**Interpretation**:

IQR Kurtosis differs from traditional kurtosis in its interpretation. While traditional kurtosis focuses on the intensity of the tails of a distribution (e.g., heavy or light tails), IQR Kurtosis is a robust measure that emphasizes the relative spread of the interquartile range (IQR) and the symmetry of the distribution around the median.

---

### Your turn

Try to calculate the IQR Kurtosis coefficient for the sample data:

---

## Summary statistics

A great tool for creating elegant summaries of descriptive statistics in Markdown format (ideal for Jupyter Notebooks) is pandas, especially in combination with the .describe() function and tabulate.

Example with pandas + tabulate (a nice table in Markdown):

---

To make a summary table cross-sectionally (i.e. **by group**), you need to use the groupby() method on the DataFrame and then, for example, describe() or your own aggregate function. 

Let's say you want to group the data by the ‘Type 1’ column (i.e. e.g. Pokémon type: Fire, Water, etc.) and then summarise the quantitative variables (mean, variance, min, max, etc.).

---

## Cross-sectional analysis

Let's try to calculate all those statistics by group i.e. perform descriptive analysis for Attack points by Legendary (for legendary and not legendary pokemons.)

---

### Your turn!

Add some cross-sectional plots and try to interpret the results.

---

Cross-sectional plots allow us to compare distributions between groups.

In the Pokémon plot, we see that Flying-type Pokémon tend to have higher and more consistent speed values, while types like Rock and Steel have lower speeds with wider variation.

In the Gapminder plot, Europe and Oceania have the highest life expectancy in 2007 with low variation, while Africa has the lowest and most spread out values, indicating inequality in health outcomes across the continent.


---

### Quiz answers on measurement scales:
1. B  
2. C  
3. C  
4. C  
5. D