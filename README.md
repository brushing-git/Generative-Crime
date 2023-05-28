# Generative Crime

![Dataset2Scatter](Images/dataset2scatter.png)

Machine learning techniques are frequently classied into three buckets:  supervised learning, unsupervised learning, and reinforcement learning.  In the supervised case, we generally know the hypotheses we want our algorithms to predict.  This can be things like predicting income based on socio-economic features in regression or predicting categorical features like object type in image recognition.  However, we do not always know what we want our models to predict.  In that case, we can appeal to unsupervised learning where algorithms learn the hypotheses themselves.  We might want to go even further and desire not only for our models to be able to use those hypotheses to register predictions but to be able to recreate the data from those hypotheses.  In this latter case, we aim for what is called *generative models*.  Here I explore both unsupervised and generative modeling techniques.  **This project demonstrates my ability to use powerful unsupervised learning techniques for building generative models useful for predicting and modeling data.**

Unsupervised learning and generative modeling can be useful in a number of applications.  One such application is the application of limited resource application.  Manpower intensive activities like policing or social services are often resource constrained.  Consequently, governments would like to know where should available resources be best deployed.  This requires having a model (in the general sense) about possible hypotheses correlated with social symptoms like crime.  Unsupervised learning can help here by *learning latent hypotheses predictive of those social symptoms and the probability of those hypotheses*.  From Chicago crime report counts, sums, and averages, I construct a series of models that learn geographically locatable hypotheses useful for predicting and recreating those Chicago crime reports.  This can be very informative for where to allocate various government services.

I analyzed three datasets containing geographical information and statistics about Chicago crime reports.  This data required grooming and cleaning using **pandas**.  From this data, I constructed a sequence of generative models via the unsupervised learning technique of **expectation maximization** to learn **Gaussian mixture models**.  The code for these algorithms was custom, though I am familiar with **scikitlearn**'s functions to do the same task.  I then evaluated those models on separate test data using a powerful Bayesian measure of predictive performance with an Occam's prior, the **Bayesian Information Criterion (BIC)**.

## Datasets

The primary data sets were taken from the UCI Machine Learning Repo found [here](https://archive-beta.ics.uci.edu/dataset/493/query+analytics+workloads+dataset).  There were 3 data sets.  Each one consisted of an x and y coordinate that gives the center of a circle that characterizes a geographic area.  In addition, the radius of that circle was included in each data set.  According to the documentation and the **pandas-profile** report, both the center of the cirlces and the radii were sampled from a Gaussian distribution.  The other items in the datasets dependend on the specific set.  Data set 2 had the number of police crime reports from the sampled geographic area, while data 3 had the number of crime reports plus the sum and averages of those reports.

The initial data sets had a few issues.  First, included in the data sets were the columns and rows labels.  These had to be dropped from the groomed data set.  In addition, data set 3 had a number of NaN values for the sums and average of crime reports.  After identifying the culprit entries, those entries were dropped from the cleaned data set.

Once cleaned, the data sets were then visualized in a Python notebook.  I created histograms for the crime counts in both datasets 2 and 3 as well as the sums and averages for dataset 3.  The results indicated an exponential distribution for the counts across both datasets.

![Dataset2countdistr](Images/dataset2countdistr.png) ![Dataset3averagedistr](Images/dataset3averagedistr.png)

I also performed a series of scatter plots to clue myself into the clustering behavior of the different data sets.  Dataset2 showed clear clustering behavior, with a high crime area appearing to occur at the intersection of two clusters.  Dataset3 showed no clear clustering behavior on crime report counts, but it did show clustering behavior on the crime averages.

![Dataset2scatter](Images/dataset2scatter.png) ![Dataset3scatter](Images/dataset3scatteraverage.png)

Both of these results suggested a type of clustering method would be appropriate.

## Statistical Methods

## Machine Learning Methods

## Experiments

## Results

## Discussion
