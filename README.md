# Feature_Engineering_Python
Comprehensive study of various feature engineering methods to analyze, interpret and transform custom datasets.

## What is feature engineering?
All machine learning algorithms use some input data to generate outputs. Input data contains many features which may not be in proper form to be given to the model directly. It needs some kind of processing and here feature engineering helps. Feature engineering fulfils mainly two goals:

1. It prepares the input dataset in the form which is required for a specific model or machine learning algorithm.
2. Feature engineering helps in improving the performance of machine learning models magically.

## Prerequisites:
1. Install Python and get its basic hands-on knowledge.

2. Pandas library in python. Command to install: pip install pandas

3. Numpy library in python. Command to install: pip install numpy

## The main feature engineering techniques that will be explored are:

1. Missing data imputation

2. Categorical encoding

3. Variable transformation

4. Outlier engineering

5.  Date and time engineering

## Missing Data Imputation for Feature Engineering
In our input data, there may be some features or columns which will have missing data, missing values. It occurs if there is no data stored for a certain observation in a variable. Missing data is very common and it is an unavoidable problem especially in real-world data sets. If this data containing a missing value is used then we can see the significance in the results. So, imputation is the act of replacing missing data with statistical estimates of the missing values. It helps us to complete our training data which can then be provided to any model or an algorithm for prediction.

There are multiple techniques for missing data imputation. These are as follows:-

1. Complete case analysis
2. Mean / Median / Mode imputation
3. Missing Value Indicator

### Complete Case Analysis for Missing Data Imputation

Complete case analysis is basically analyzing those observations in the dataset that contains values in all the variables. Or we can say, remove all the observations that contain missing values. But this method can only be used when there are only a few observations which has a missing dataset otherwise it will reduce the dataset size and then it will be of not much use.

So, it can be used when missing data is small but in real-life datasets, the amount of missing data is always big. So, practically, complete case analysis is never an option to use, although we can use it if the missing data size is small.

### Mean/ Median/ Mode for Missing Data Imputation

Missing values can also be replaced with the mean, median, or mode of the variable(feature). It is widely used in data competitions and in almost every situation. It is suitable to use this technique where data is missing at random places and in small proportions.

### Missing Value Indicator For Missing Value Indication

This technique involves adding a binary variable to indicate whether the value is missing for a certain observation. This variable takes the value 1 if the observation is missing, or 0 otherwise. But we still need to replace the missing values in the original variable, which we tend to do with mean or median imputation. By using these 2 techniques together, if the missing value has predictive power, it will be captured by the missing indicator, and if it doesn’t it will be masked by the mean / median imputation.

## Categorical encoding in Feature Engineering

Categorical data is defined as that data that takes only a number of values. Let’s understand this with an example. Parameter Gender in a dataset will have categorical values like Male, Female. If a survey is done to know which car people own then the result will be categorical (because the answers would be in categories like Honda, Toyota, Hyundai, Maruti, None, etc.). So, the point to notice here is that data falls in a fixed set of categories.

If we directly give this dataset with categorical variables to a model, we will get an error. Hence, they are required to be encoded. There are multiple techniques to do so:

1. One-Hot encoding (OHE)
2. Ordinal encoding
3. Count and Frequency encoding
4. Target encoding / Mean encoding

### One-Hot Encoding

It is a commonly used technique for encoding categorical variables. It basically creates binary variables for each category present in the categorical variable. These binary variables will have 0 if it is absent in the category or 1 if it is present. Each new variable is called a dummy variable or binary variable.

Example: If the categorical variable is Gender with labels female and male, two boolean variables can be generated called male and female. Male will take 1 if the person is male or 0 otherwise. Similarly for a female variable.

### Ordinal Encoding

What does ordinal mean? It simply means a categorical variable whose categories can be ordered and that too meaningfully.

For example, Student’s grades in an exam are ordinal. (A,B,C,D, Fail). In this case, a simple way to encode is to replace the labels with some ordinal number.

### Count and Frequency Encoding

In this encoding technique, categories are replaced by the count of the observations that show that category in the dataset. Replacement can also be done with the frequency of the percentage of observations in the dataset. Suppose, if 30 of 100 genders are male we can replace male with 30 or by 0.3.

This approach is popularly used in data science competitions, so basically it represents how many times each label appears in the dataset.

### Target / Mean Encoding

In target encoding, also called mean encoding, we replace each category of a variable with the mean value of the target for the observations that show a certain category. For example, there is a categorical variable “city”, and we want to predict if the customer will buy a TV provided we send a letter. If 30 percent of the people in the city “London” buy the TV, we would replace London with 0.3. So it helps in capturing some information regarding the target at the time of encoding the category and it also does not expands the feature space. Hence, it also can be considered as an option for encoding. But it may cause over-fitting to the model, so we must be careful.

## Variable Transformation

Machine learning algorithms like linear and logistic regression assume that the variables are normally distributed. If a variable is not normally distributed, sometimes it is possible to find a mathematical transformation so that the transformed variable is Gaussian. Gaussian distributed variables many times boost the machine learning algorithm performance.

Commonly used mathematical transformations are:

1. Logarithm transformation – log(x)
2. Square root transformation – sqrt(x)
3. Reciprocal transformation – 1 / x
4. Exponential transformation – exp(x)

## Outlier engineering

Outliers are defined as those values that are unusually high or low with respect to the rest of the observations of the variable. Some of the techniques to handle outliers are:

1. Outlier removal

2. Treating outliers as missing values

3. Outlier capping

### How to identify outliers?

For that, the basic form of detection is an extreme value analysis of data. If the distribution of the variable is Gaussian then outliers will lie outside the mean plus or minus three times the standard deviation of the variable. But if the variable is not normally distributed, then quantiles can be used. Calculate the quantiles and then inter quartile range:

Inter quantile is 75th quantile-25quantile.

upper boundary: 75th quantile + (IQR * 1.5)

lower boundary: 25th quantile – (IQR * 1.5)

So, the outlier will sit outside these boundaries.

### Outlier removal
In this technique, simply remove outlier observations from the dataset. In datasets if outliers are not abundant, then dropping the outliers will not affect the data much. But if multiple variables have outliers then we may end up removing a big chunk of data from our dataset. So, this point has to be kept in mind whenever dropping the outliers.

### Treating outliers as missing values
You can also treat outliers as missing values. But then these missing values also have to be filled. So to fill missing values you can use any of the methods as discussed above in this article.

### Outlier capping
This procedure involves capping the maximum and minimum values at a predefined value. This value can be derived from the variable distribution. If a variable is normally distributed we can cap the maximum and minimum values at the mean plus or minus three times the standard deviation. But if the variable is skewed, we can use the inter-quantile range proximity rule or cap at the bottom percentiles.

## Date and Time Feature Engineering

Date variables are considered a special type of categorical variable and if they are processed well they can enrich the dataset to a great extent. From the date we can extract various important information like: Month, Semester, Quarter, Day, Day of the week, Is it a weekend or not, hours, minutes, and many more. Let’s use some dataset and do some coding around it.