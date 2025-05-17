Predicting Credit Card Default Risk

Ranjith Kumar Saila,
Ganesh Sai Uttej Kayala,
Michele Perina,
Robert Mikkelson,
Sharifur Rahman,

Department of Business, University of Central Oklahoma,

MSBA 5314
Dr. Ho-Chang Chae
December 12th, 2024

# Abstract

Understanding factors that influence credit card default rates is critical for financial institutions aiming to manage risk effectively while fostering inclusivity. This study focuses on assessing the impact of migrant worker status on credit card default rates, a demographic that remains largely unexplored in existing literature. Using a dataset of American Express customers, we analyzed variables such as demographics, financial behaviors, employment, and socio-economic attributes to uncover patterns and relationships.

Advanced machine learning models, including decision trees, logistic regression, and neural networks, were employed to evaluate default likelihood, with and without credit-related variables. Migrant worker status emerged as a significant predictor, with migrant workers showing marginally higher odds of default. This demographic’s distinct financial challenges, such as income instability and limited credit history, may contribute to this trend. Models incorporating migrant worker status provided novel insights, enhancing our understanding of this group’s credit behavior.

While credit-related factors like credit score and limit utilization remained dominant predictors, focusing on non-credit variables highlighted actionable opportunities for financial inclusivity. These findings suggest that tailored strategies addressing migrant workers’ unique financial circumstances can reduce default risks while opening avenues for expanding credit access responsibly. This research bridges a critical gap, guiding banks to balance financial risk with social equity.

# Introduction

Predicting credit card defaults is crucial for banks and financial institutions, as default rates directly impact their bottom line and strategic planning. Despite the significant resources allocated to this task, American Express reported losses of approximately $1.4 billion in the last quarter of 2023 (CITATION). This highlights the ongoing challenge of predicting defaults while expanding the customer base. Traditional prediction models have relied on financial indicators and demographic variables such as age, income, credit utilization, and payment history, but these predictors often lack the depth needed to capture the diverse customer groups in the evolving demographics of credit card holders. This gap has led to increased interest in behavioral factors like payment history and credit limit usage, as well as a need for more inclusive credit risk assessments.

Our research differs from previous studies, which have largely relied on datasets like the Taiwan Credit Data from the UCI Machine Learning Repository, leading to similar results. Instead, we utilize a novel dataset specifically about American Express customers in the United States, accounting for the influence of local financial behaviors, protection laws, and credit reporting practices. We propose integrating migrant worker status as a predictive factor—an underexplored variable in credit default prediction. Migrant workers face unique financial challenges, including employment instability and inconsistent credit histories, and may therefore present additional risks to credit card companies. By investigating whether migrant workers contribute significantly to credit card default prediction, we aim to determine if they represent an underserved demographic, offering an opportunity for American Express to expand into this market and promote financial inclusion.

Finally, our study emphasizes the importance of non-financial predictors, recognizing that financial variables alone are insufficient. While credit scores have traditionally been used to predict defaults, we believe that analyzing overlooked factors can enhance credit risk models, improve prediction accuracy, and help credit card providers refine their methodologies to better assess risk and target underserved demographics.

# Literature Review

The existing literature on credit card default prediction predominantly relies on financial indicators, such as income, credit history, and traditional demographic variables. Yeh and Lien (2009) utilized the Taiwan Credit Data from the UCI Machine Learning Repository, establishing a baseline for machine learning models in credit prediction (Arora et al., 2022; Sayjadah et al., 2018; Subasi & Cankurt, 2019; Teng & Lee, 2019; Yu, 2020). This dataset and similar ones from Germany, Belgium, and China have frequently been used in predictive studies (Alam et al., 2020; Li et al., 2019). Arora et al. (2022) also utilized this dataset to evaluate the efficacy of machine learning models, noting that attributes like income and credit limit were among the most influential predictors.

A notable literature gap remains concerning the predictive power of non-traditional demographic variables, especially for groups like migrant workers, who present unique financial behaviors and risks. Prior research has often neglected or undervalued these and other demographic indicators, limiting the scope of credit risk prediction and creating a significant blind spot for lenders who aim to support diverse and underserved populations. While most studies incorporate a blend of financial and non-financial variables, credit-related factors have consistently dominated the models due to their strong predictive accuracy (Dunn & Kim, 1999; Guerra, 2023; Jain & Jayabalan, 2022; Yeh & Lien, 2009). This approach, while effective for individuals with established credit histories, lacks inclusivity for clients like migrant workers, who may not have a substantial credit history in the host country or may hold records in foreign credit systems.

The originality of our project lies in its dual-model approach, assessing predictive models that include traditional credit-related variables for optimal accuracy and models that exclude these to highlight the value of demographic factors that are often overlooked. Moreover, this analysis is made possible by a unique American Express dataset, rarely accessible in existing research, which includes variables such as migrant worker status, car ownership, and detailed employment data like occupation and days employed. In contrast, traditional datasets generally contain only basic demographic factors like age, gender, education, and homeownership, which limits their utility for fully understanding diverse credit behaviors (Dunn & Kim, 1999).

By evaluating this more comprehensive dataset, this study seeks to uncover insights that could lead to more inclusive and accurate credit risk assessments, especially for clients with atypical credit backgrounds, thus supporting American Express’s goal of expanding responsibly into emerging demographics.

Data cleaning and preparation strategies play a critical role in ensuring accurate and unbiased predictions, especially in studies involving imbalanced datasets or substantial missing data. In this study, missing values were handled through imputation: mean or median imputation for interval variables and mode for nominal variables. Most credit card default datasets are imbalanced, with far fewer defaults than non-defaults. Studies such as those by Kiarie et al. (2015) and Dunn and Kim (1999) did not account for this imbalance before implementing their predictive models, which can lead to biased models that favor the majority class and often underperform on minority classes. Most prior studies addressed this imbalance through oversampling (Guerra, 2023; Jain & Jayabalan, 2022; Subasi et al., 2019). In our research, however, we will use undersampling, as we have sufficient data in the minority class and wish to avoid synthetic observations. Of the studies we reviewed, only Alam et al. (2020) also used undersampling.

Outlier handling is another crucial step. Log transformations will be applied in our research to high-skew variables, creating a more stable basis for prediction. Dunn and Kim (1999) also applied log transformations to normalize income distribution, which improved their model’s accuracy by 7%. In addition, entries with implausible data, such as excessively high income figures, will be removed to prevent distortion of results. The same approach was also employed by Jain and Jayabalan (2022) and Teng and Lee (2019). By meticulously preparing the dataset, our study will ensure robust predictions that accurately reflect the impact of non-financial indicators.

The literature review reveals a spectrum of machine learning techniques applied to credit default prediction, with each model offering distinct advantages. Logistic regression, decision trees, and Random Forest models are widely employed across studies for their interpretability and robustness in handling classification problems. For example, Sayjadah et al. (2018) and Yeh and Lien (2009) used logistic regression and decision trees to classify default risks, achieving accuracy levels of 89% and 91%, respectively.. However, these models fall short when capturing nonlinear interactions, limiting their effectiveness in predicting behaviors among diverse demographic groups To solve this problem, Goodman and Bai (2016) created multiple interaction terms between different variables before running the regression. This resulted in a more accurate model, but harder to interpret predictions.

More recent studies have explored advanced machine learning models to enhance prediction accuracy, particularly in handling large datasets and complex relationships. For instance, Jain and Jayabalan (2022) achieved 95.84% accuracy using a Random Forest model, which outperformed simpler models like logistic regression in handling imbalanced data. Alam et al. (2020) applied Gradient Boosting Decision Trees (GBDT) across multiple datasets and reported accuracy levels exceeding 90%, demonstrating GBDT’s efficacy in capturing intricate patterns in credit data. This study builds on these approaches by including SVM, LASSO, LARS, Adaptive LASSO, Random Forest, PCA, and Variable Clustering, enabling us to address high-dimensional data and optimize variable selection. Another model that has shown strong performance in predicting credit default in the literature is neural networks. Jain and Jayabalan (2022) achieved an accuracy of 93.76%, while Neema and Soibam (2017) and Sayjadah et al. (2018) reached accuracies around 87%.

Evaluation techniques across studies were tailored to address common challenges in credit default prediction, particularly the need to balance accuracy with sensitivity and specificity due to the class imbalance in default cases. Yeh and Lien (2009) employed confusion matrices and classification metrics like sensitivity and specificity to measure model performance, while Jain and Jayabalan (2022) applied SMOTE to enhance Random Forest’s recall, achieving a precision of 94.87% and recall of 85.85%. In our study, we applied a cost-sensitive approach to handle the imbalance in our dataset, weighting the minority class to mitigate bias toward non-default cases.

Several studies employed under-sampling to address imbalance, as seen in Alam et al. (2020), who matched rare outcomes with secondary cases to achieve a 50-50 split. However, our study used class weighting, which is particularly beneficial in cost-sensitive contexts. A comparison of these methods shows that under-sampling, while effective in balancing datasets, can reduce the number of training samples, potentially impacting model robustness. Conversely, class weighting, as applied in our study, allowed us to retain all samples, preserving model stability and accuracy.

Metrics like F1 score, sensitivity, and specificity were used to evaluate the model comprehensively, given the complexity of predicting rare events. Dunn and Kim (1999) emphasized the importance of sensitivity in evaluating default risk models, as failing to detect a default risk can have greater financial implications than a false positive.


**Table of Literature:**


Modeling Methods:

CREATE and ADD variables table

# Data Understanding

Our project data was sourced from Kaggle. The data was created for the AmExpert 2021 – Code Lab competition, a hackathon organized jointly by American Express and HackerEarth and open to participants across India. A contestant from the competition uploaded the dataset to Kaggle for educational use.

The training dataset consists of following variables:


**Table 1: Data Dictionary**

![image](https://github.com/user-attachments/assets/3512e049-a983-4ed7-af5b-753752be5817)
![image](https://github.com/user-attachments/assets/f3837baf-114b-428f-b122-bbb3d858da90)
![image](https://github.com/user-attachments/assets/7bf50355-242a-47a0-8327-ac73e89e1906)



The initial dataset contains 45,528 observations and 19 variables. There are 10 interval variables, 4 nominal variables, and 5 binary variables including the target variable which is credit card default.

## Data Preprocessing

The first step in processing our dataset was to address and remove observations with data entry errors and inconsistencies. When applying for an American Express card, customers fill out a form without any data validation checks, allowing them to enter invalid or illogical information. This resulted in errors within the dataset.

One major issue we identified was the inconsistency between a person’s age and the number of days they had been employed. In some cases, individuals were recorded as having worked longer than they had been alive. In the United States, the minimum legal working age, with parental consent, is 14 years. Based on this assumption, we removed all customers who appeared to have started working before this age. This issue affected multiple rows, leading to the removal of 16,051 rows.

Another significant data entry error was a row containing extreme outliers for net income and credit limit. This record reported a net income of 140 million and a credit limit of 31 million, while the average values for these variables are approximately 200,000 and 40,000, respectively.

The final data entry issue involved the gender column. While it is normal to include more than two gender options for individuals who prefer not to specify, the dataset contained only one distinct anomalous value, "X," which appeared to be a data entry error. We treated "X" as missing and replaced it with the most frequent value, "female," using a Replacement Node. The new value assigned was "N."


**Table 2: Gender**

![image](https://github.com/user-attachments/assets/9ea1f3c4-181c-4c40-bdf3-f24790f614ee)

These were the only observations removed due to data entry errors. Other variables required cleaning, but we addressed these issues without removing rows entirely. After cleaning, the final dataset contained 29,475 observations, which were used for predictive modeling and exploratory data analysis.

The second step we took in the process of cleaning the dataset was checking for missing values. Many variables in our dataset contain missing values. The following tables provides a detailed overview of the number of missing values for the numeric variables:


**Table 3: Missing Numeric Values**

![image](https://github.com/user-attachments/assets/05b57f45-dfea-4ace-b7c9-bcf4fdd5b4f2)
![image](https://github.com/user-attachments/assets/68d421f8-aab6-4010-b8be-4f446695cbf0)
![image](https://github.com/user-attachments/assets/6787ecda-394f-4cbb-be31-305c77512f1b)
![image](https://github.com/user-attachments/assets/7e77637c-5733-4920-86bc-537b68ac283e)

Numeric variables that contain missing values are: no_of_children, total_family_members, yearly_debt_payments and credit_score. The percentage of missing values is relatively low, with a maximum of 1.71% across all variables. Given this low rate, one option could have been to remove rows with missing values. However, we chose to impute these values to keep all the observations. In SAS Miner 15.2, we utilized the Impute Node to complete this step, using the mean.

Among the categorical variables, the only one with missing values is migrant worker.


**Table 4: Missing Nominal Values**

![image](https://github.com/user-attachments/assets/4e48acba-038f-405c-aaab-98093e5572dc)

Similarly to the numeric variables discussed above, we are not planning on deleting rows containing missing values for this variable. We have opted to impute it using the mode (most frequently occurring value). Also in this case, the Impute Node was utilized to deal with missing values and the Default Input Method was set to Count for class variables.

The last step we took in the data preprocessing phase was looking at the distribution plots and to look for distribution and possible outliers. For numeric variables below, we can see how multiple variables have right skewed distribution and have outliers:

Chart 1: Numeric Variable Distribution Plots and Skewness
![image](https://github.com/user-attachments/assets/ab4a5596-b162-4a77-8515-b24ab74eb900)
![image](https://github.com/user-attachments/assets/6a3ae3d8-8db4-4399-8ef9-c1eacfdabce3)
![image](https://github.com/user-attachments/assets/4108457f-c973-4c00-b028-dae0bf9f153e)
![image](https://github.com/user-attachments/assets/344ef915-6331-4b9d-9da6-a3aebcde2f67)
![image](https://github.com/user-attachments/assets/26cbeda5-75ba-4360-921a-37550a470f61)
![image](https://github.com/user-attachments/assets/6c01eda7-0e82-4a89-aa56-77edb47d976a)
![image](https://github.com/user-attachments/assets/0ff256ae-dd18-4a37-8775-d8bd9d602ec8)
![image](https://github.com/user-attachments/assets/94bfe32e-72e4-4907-b332-b71ecc713590)

To deal with outliers, we decide to use the Transform node in SAS Miner and use the log transformation on all the variables that have skewness higher than 3. In our case, the only variables with skewness higher than 3 are previous defaults, credit limit and net yearly income. After applying the log transformation, all the variables have skewness of less than 1, improving their suitability for predictive modeling.

All the numeric variables not shown above have no outliers and normal distributions.

## Exploratory Data Analysis

As mentioned above, our dataset contains 10 interval variables. The following table contains summary statistics for these variables.


**Table 5: Summary Statistics for Interval Variables**

![image](https://github.com/user-attachments/assets/3d3d7375-3c9c-46c3-878a-5caf75794821)

The tables below are the frequency tables for categorical variables. The last frequency table represents our target binary variable, Credit Card Default.


**Table 7: Frequency Tables for Categorical Variables**

![image](https://github.com/user-attachments/assets/adbc6631-ecf0-4955-b7c9-f0cc5545baa1)
![image](https://github.com/user-attachments/assets/b11af048-e63b-4339-8491-b8e313e054f0)
![image](https://github.com/user-attachments/assets/0f4b9e1b-b9c2-42f4-b28f-4cb91d157433)
![image](https://github.com/user-attachments/assets/7573e458-4bc5-44e6-a847-7aaebedf4c84)
![image](https://github.com/user-attachments/assets/a0d0c6a8-431c-4614-9818-4f62b450c311)
![image](https://github.com/user-attachments/assets/7ab63241-384b-4086-b2fa-1659ea4b198a)
![image](https://github.com/user-attachments/assets/aea91300-3065-4312-99c5-163e58a9e4d0)
![image](https://github.com/user-attachments/assets/89b69bc3-e2c5-49f6-af59-efbaeb04d17f)

Our target variable is credit_card_default. The percentage of individuals defaulting is 9.15%, meaning that the dataset is imbalanced. We plan to address this issue by under sampling the non-defaulting target class to restore equilibrium to the dataset. We will select all the rare primary outcomes (default) and match them with a secondary outcome case (non-default) to create a true 50-50 split of the data.

The existing literature has shown that credit card default is influenced by additional variables and factors like age and gender. Li et al. (2019) found that individuals under 30 are more likely to default on credit card debt. To examine if this finding holds in our dataset, we grouped our data into two age categories: under 30 and over 30. We then compared default rates between these groups. Our analysis showed that the default rate is nearly identical across both groups, with 8.30% of customers under 30 and 8.07% of customers over 30 defaulting. This result differs from previous research, likely because our dataset, sourced from American Express, represents a typically wealthier clientele, which may reduce the age-related likelihood of default. A chi-square statistic of 0.22, at a significance level of 0.05, confirms no significant difference between the age groups in default rates.


**Table 8: Credit Card Default By Age Over/Under 30**

![image](https://github.com/user-attachments/assets/eacd48b5-ec75-4cab-917f-645c66693240)
![image](https://github.com/user-attachments/assets/b27d5efa-d209-459c-a508-15e20acc0dec)

We also wanted to determine if gender plays a role in the expected likelihood of defaulting. Two studies (Achsan et al., 2022; Godman et al., 2016) highlight that males, especially younger ones, tend to struggle more with credit card payments compared to females. Our analysis showed that females are indeed less likely to default: 10.30% of males defaulted, compared to only 6.99% of females. The Chi-Square test further confirms this result as statistically significant, with a p-value below 0.0001. Our findings appear to align with the existing literature on this topic.


**Table 9: Credit Card Default by Gender**

![image](https://github.com/user-attachments/assets/31bfa6e5-fe49-469f-a53c-784660fec9c3)

Multicollinearity is one of the biggest issues when creating predictive models. For this reason, we analyzed the correlation between the interval variables.


**Table 10: Correlation Matrix Between Interval Variables**

![image](https://github.com/user-attachments/assets/80d1a78c-66fe-4033-a43f-555981ee7c8f)

The variables with the highest correlation are net_yearly_income and credit_limit with a very high correlation of (0.91). The variables with the second highest correlation are no_of_children and total_family_members (0.89). The variables with the third highest correlation are prev_defaults and credit_limit_used(%). The strongest negative correlation is between credit_score and credit_limit_used(%). This makes sense because when the percentage of credit limit used goes up, the credit score tends to go down.

When creating machine learning models, it is important to pay close attention to these variables with high correlation to avoid multicollinearity issues. Some of the models we plan to use, such as tree-based models (decision trees and random forests), are unaffected by multicollinearity. However, other models, like logistic regression, may produce biased results as they lack internal methods to handle multicollinearity. Therefore, we decided to exclude the variables credit_limit and no_of_children to minimize the risk of biased predictions.


**Table 11: Excluded Variables by Correlation**

![image](https://github.com/user-attachments/assets/694dd694-24e8-4edc-b2f3-2764941daf36)

Credit limit and number of children were not the only variables that will not be used in building predictive models. Name, of course, was rejected because it does not provide any predictive value, especially in this case where names were changed because of privacy. ID is another variable that will not be used, and will be given the ID role in SAS Miner.

Two more variables that we decided to remove after analyzing them more in detail are Prev_Defaults and Default_In_Last_6Monthsl. In the dataset, every individual who defaulted at least once before has defaulted on its credit card debt. Similarly, among those who defaulted in the last 6 months, all have defaulted on their credit card debt. These two variables are evidently too strong in predicting credit card default, which could result in extremely high odds ratios and standardized estimates that lack practical insight. This issue arises during model creation, and for this reason, excluded both variables in SAS Miner to avoid incorporating them into the predictive models.


**Table 12: Previous Defaults by Credit Card Default**
![image](https://github.com/user-attachments/assets/cf487ef5-e96f-4416-85c5-658d9a5973ad)


**Table 13: Previous Defaults by Credit Card Default**
![image](https://github.com/user-attachments/assets/c0d1fa91-2cfb-46cd-aa11-fe17c147cb3f)


Another variable with similar behavior is Credit_Score. After analyzing it closely, an interesting insight was discovered. In our dataset, no individual with a credit score above 699 has defaulted, underscoring the credit score's importance in predicting defaults. We attempted to create a new categorical variable for credit scores based on industry standards—800+ as exceptional, 740-799 as very good, 670-739 as good, 580-669 as fair, and below 580 as poor (MyFICO, 2004). However, the top categories, exceptional and very good, contained only non-defaulting individuals. If we adopted this approach, we would incur in the same complete separation issue that we had above with default last 6 months and previous defaults. For this reason, this approach was not used to create predictive models.

We also performed feature engineering and created two additional columns.

The first column involved converting the "number of days employed" into "number of years employed." This transformation makes it easier to interpret the model results.

The second column was derived from the "occupation type" variable. We divided the dataset into two groups: professional and non-professional jobs. Professional jobs included roles such as accountant, HR staff, high-skilled tech staff, IT staff, manager, medical staff, realty agents, sales staff, and secretary. All other roles, typically requiring less formal education or specialized training and often categorized as manual, support, or service jobs, were grouped as non-professional.

# Statistical Analysis

Multiple T-Tests, ANOVA tests and Chi-Square tests were performed to assess the relationships among variables.

T-test: Mean Credit Score vs Default


**Table 14: T-Test of Mean Credit Score vs. Default**

![image](https://github.com/user-attachments/assets/eab8ab38-b7e7-49d7-87db-59cac57a8c10)



An examination of the test for the equality of variance (folded F test) shows that the p-value is smaller than 0.05. Consequently, there is enough evidence to reject H0, suggesting that the variances of the two groups are statistically different. For this reason, the Satterthwaite was used.

The p-value of the Satterthwaite is less than 0.0001, and, for this reason, we can reject H0. At a significance level of 0.05, there is a statistically significant difference between the average credit score of people who defaulted and those who didn’t. It appears that the credit score impacts whether a person defaults on their credit card debt or not. More precisely, people with lower credit scores are more likely to default.

T-test: Mean Credit Limit Used (%) vs Default


**Table 15: Mean Credit Limit Used (%) vs. Default**

![image](https://github.com/user-attachments/assets/3f5cde35-4620-4e1a-9581-b90f6fc7acac)



An examination of the test for the equality of variance (folded F test) shows that the p-value is smaller than 0.05. Consequently, there is enough evidence to reject H0, suggesting that the variances of the two groups are statistically different. For this reason, the Satterthwaite was used.

The p-value of the Satterthwaite is less than 0.0001, and, for this reason, we can reject H0. At a significance level of 0.05, there is a statistically significant difference between the average percentage of credit used of people who defaulted and those who didn’t. It appears that the percentage of credit limit used impacts whether a person defaults on their credit card debt or not. More precisely, people who default on average have higher percentages of credit limit used. Credit_limit_used (%) is likely to be an important variable when building models to predict credit card default

T-test: Mean Number of Years Employed vs Default


**Table 17: T-Test of Mean Number of Days Employed vs Default**

![image](https://github.com/user-attachments/assets/e0d64ee2-5eb9-4cec-8a40-a2bdcfe03347)



An examination of the test for the equality of variance (folded F test) shows that the p-value is smaller than 0.05. Consequently, there is enough evidence to reject H0, suggesting that the variances of the two groups are statistically different. For this reason, the Satterthwaite was used.

The p-value of the Satterthwaite is less than 0.0001, and, for this reason, we can reject H0. At a significance level of 0.05, there is a statistically significant difference between the average number of years an individual has been employed for those who defaulted and those who didn’t. It appears that the number of years a person was employed for impacts whether a person defaults on their credit card debt or not. More precisely, people with a lower number of days employed are more likely to default.

ANOVA test: Occupation Type vs Mean Credit Score


**Table 18: ANOVA Test Occupation Type vs Mean Credit Score**

![image](https://github.com/user-attachments/assets/23267e67-d972-4a66-9051-b79cb3edc33d)
![image](https://github.com/user-attachments/assets/6b054538-9e6f-425b-ada4-bc03680de617)



To perform ANOVA, since we need a categorical variable with more than 2 levels, we utilized the occupation type before being aggregated into the professional and non professional group. The p-value of the ANOVA is less than 0.0001. At a significance level of 0.05, we can reject H0. Consequently, there is a statistical difference between the average credit score and at least one of the occupation types. IT staff has the highest while realty agents have the lowest one.

Chi-Square Test: Default vs. Migrant Worker


**Table 19: Chi-Square Test Credit Card Default vs Migrant Worker**
![image](https://github.com/user-attachments/assets/7f7be43a-03c5-41e3-a037-862e53fa0ebe)
![image](https://github.com/user-attachments/assets/c028e65a-baf4-42cc-9972-205d737a8a27)




The p-value of the chi-square (<0.0001) is smaller than the significance level of 0.05. Therefore, we can reject the null hypothesis (Ho: whether or not a person defaults is independent of their status as migrant worker). This suggests that the two variables are not independent, and the fact that a person is a migrant worker or not does have a statistically significant influence on whether or not a person defaults. As a consequence, being a migrant worker will likely be an important variable when predicting credit card default.

After conducting data exploration and statistical analysis, we have a clear understanding of the dataset. It is obvious that credit-related values have a much higher predictive power than the other variables. This is great because we are able to build models that can predict credit card defaults very precisely. However, at the same time, the predominance of credit-related variables can inhibit the influence that other variables have on credit card default. To obviate this problem, we created two different datasets, one containing credit-related variables and the other without. Every predictive model will be estimated twice, once for each dataset. The final data dictionary of the two datasets is as follows:


**Table 20: Data Dictionary Final Datasets**
![image](https://github.com/user-attachments/assets/6074da27-054e-4141-84e5-70aab647df83)



# Modeling

Before running the models, we plan to address the issue of having an imbalance dataset by under sampling the non-defaulting target class to restore equilibrium to the dataset. We will select all the rare primary outcomes (default) and match them with a secondary outcome case (non-default) to create a true 50-50 split of the data. However, employing separate sampling presents certain challenges. The majority of model fit statistics may exhibit bias. For this reason, the cost-sensitive approach is implemented to allocate varying weights to different classes. This ensures that the minority target class (Default) receives a higher weight, consequently incurring a higher misclassification cost. By doing so, bias toward the majority class (non-Default) can be mitigated. We applied the inverse of the class distribution, giving a weight of 11.11 to the minority class. The sample is then divided using the Data Partition node such that 50% of the data would be utilized for training and the other 50% for validation.

For each dataset analyzed, a total of 26 models were developed, consisting of 3 tree-based models, 11 logistic regression models, 8 neural networks, and 4 support vector machines (SVMs).

The analysis began with the three tree-based models. Decision trees and random forests, which are capable of handling missing values, were connected directly to the Data Partition node without requiring prior imputation or transformation. The decision tree models were constructed in SAS Miner 15.2 using Misclassification as the Subtree Assessment Measure. Two random forests were also developed: the first using SAS Miner’s default settings with 100 trees, and the second employing a larger configuration with 200 trees.

Next, 11 logistic regression models were built using various variable selection techniques. The first model employed stepwise selection, implemented by setting the Model Selection option to Stepwise in SAS Miner, with Validation Misclassification as the selection criterion. The second model utilized the Variable Selection node, which identifies variables that maximize the model's R-squared value, with the AOV16 Variables, Group Variables, and Interactions parameters set to "Yes." The third and fourth models applied partial least squares regression, combining elements of multiple and principal components regression to create linear combinations of inputs accounting for both the target and predictors. The fifth regression model selected variables using a decision tree, with the Number of Surrogate Rules parameter set to 1 and the Subtree Method set to "Largest" in SAS Miner. For the sixth model, a Principal Component Analysis (PCA) node was used to create components, which were then used as inputs for the regression. Additionally, two models utilized the Variable Clustering node: one incorporated the clusters directly, while the other selected the best variables from each cluster. The final three logistic regression models used advanced machine learning variable selection methods, LARS, LASSO, and Adaptive LASSO, which, as highlighted in the literature review, have not been widely used in previous studies.

The neural network models were developed using the same variable selection techniques applied to the logistic regression models.

Finally, the last four models were SVMs, a robust method for classification tasks. These models explored four kernel functions—linear, polynomial, radial basis function (RBF), and sigmoid—each offering unique strengths and addressing different aspects of the data structure in the context of credit default prediction. This diversity of kernel functions allowed for a comprehensive evaluation of SVM performance across various scenarios.

Before analyzing the models, it is essential to note that accuracy was prioritized as the primary metric for evaluating model performance, aligning with the approach commonly adopted in previous studies. However, additional metrics such as sensitivity, specificity, precision, and F1 score were included to provide a more comprehensive comparison of the models and their results. Sensitivity, in particular, holds critical importance in credit card default prediction, as failing to identify potential defaulters (false negatives) can lead to substantial financial losses by extending credit to high-risk individuals. By emphasizing sensitivity, the models aim to identify the majority of defaulters, even at the expense of accepting some false positives, which generally pose a lower financial risk to lenders.

## Modeling with Credit Variables

The table below presents the metrics for the 26 models developed using the dataset, which includes credit-related variables. The primary objective of this analysis was to identify the most accurate model, rather than interpreting specific variable parameters, to help American Express minimize financial risk. The chosen champion model for this dataset is the larger random forest, which was constructed using 200 trees. This model achieved an accuracy of 95%, matching the stepwise neural network. However, given the importance of specificity in predicting credit card defaults, the random forest was selected due to its higher specificity (0.96 compared to 0.93).
![image](https://github.com/user-attachments/assets/49d475d1-d20d-4acf-9796-43c61c8e3c00)
![image](https://github.com/user-attachments/assets/312a5511-2ce8-4860-aca8-80ca38c1f906)



While the model's accuracy is notably high, it is important to acknowledge potential limitations stemming from the data cleaning process, particularly with credit-specific variables such as credit scores. Nonetheless, our results align closely with prior research, where models incorporating credit variables achieved accuracies ranging from 0.89 to 0.96.

An analysis of the variables selected by the 26 models reveals that credit score and credit limit utilization (%) were consistently chosen across all models. This outcome aligns with expectations based on the exploratory data analysis. Consequently, the second part of this paper focuses on a dataset that excludes credit-related variables to explore alternative predictors.
![image](https://github.com/user-attachments/assets/ee72a8e0-7706-4e2c-bb42-7d5fdbf9fe4e)


Although interpretability was not the primary focus of this part of the project, examining the importance of variables and their effects on the target variable provides valuable insights. The table below presents parameter estimates and odds ratios for the partial least squares regression, one of the most accurate regression models. The standardized estimates indicate that credit score is approximately 35 times more significant than the first non-credit-related interval variable, years of employment. Similarly, credit limit utilization (%) is four times more significant than years of employment.

The odds ratios of the two credit-related variables offer further insights. For credit score, the odds ratio estimate is 0.94, indicating that for each additional unit of credit score, the odds of defaulting decrease by 6%. For credit limit utilization (%), the odds ratio estimate is 1.11, meaning that each 1% increase in credit limit usage raises the odds of defaulting by 11%. Interestingly, the odds ratio for migrant worker status is approximately 1 (0.99), suggesting that the likelihood of default is nearly identical for migrant and non-migrant workers.
![image](https://github.com/user-attachments/assets/bb5b5a09-60df-42cb-98c9-fb8b925a5907)


Conversely, the decision tree, one of the most interpretable models, did not offer any new insights. It produced only two splits, based on credit score and credit limit utilization (%), which were already consistently used by all other models.
![image](https://github.com/user-attachments/assets/5c977dee-bb92-47f8-a099-6ac9bdffe05c)


## Modeling without Credit Variables

The second part of this project focused on predicting credit card default using only non-credit-related variables. While credit variables are highly predictive, they tend to overshadow the effects of less significant variables, such as demographic factors. To address this, we developed models that excluded credit-related variables.

These models serve two purposes: highlighting the impact of other variables and addressing scenarios where customers lack a credit history, making credit decisions more challenging. For instance, migrant workers, a focus of this research, are individuals who temporarily relocate to another country for work. Despite possibly having a credit history in their home country, they may lack one in the new country, complicating the process of obtaining credit cards.

As expected, the table below shows a significant reduction in model accuracy without credit-related variables, underscoring their critical role in predicting credit card default. In this context, where accuracy is not the primary objective, the best-performing model was a logistic regression using partial least squares for variable selection, chosen based on accuracy and sensitivity.
![image](https://github.com/user-attachments/assets/9073fb49-a859-4067-9506-9aa5f3ee2258)
![image](https://github.com/user-attachments/assets/98da6616-59f2-40a1-8dbe-d3bf84c6e810)


With credit-related variables removed, additional variables gained prominence in many models. Car ownership, previously insignificant in most models, has now become the most frequently used variable in the dataset. Other variables that appear more often include years of employment, gender, income, and occupation type. Migrant worker status also appears in more models than before, ranking among the top five most utilized variables.
![image](https://github.com/user-attachments/assets/ad41ce42-71b8-4321-9036-337aa9e9842a)

As before, we can analyze the odds ratios and parameter estimates from the partial least squares regression. The most significant numeric variable, based on standardized estimates, is years of employment, which is approximately 2.5 times more influential than the log of net income.

For car ownership (0 vs. 1), the odds ratio is 1.42, indicating that individuals without a car have 1.42 times higher odds of defaulting compared to those with a car. For gender (F vs. M), the odds ratio is 0.66, meaning that males have 1.52 times higher odds of defaulting compared to females. For occupation type, the odds ratio is 1.16, suggesting that individuals in non-professional roles have 1.16 times higher odds of defaulting compared to those in professional jobs.

Regarding migrant worker status, the odds ratio has become more distinct from 1, though its effect remains smaller relative to other variables. Migrant workers have 1.06 times higher odds of defaulting compared to non-migrant workers.
![image](https://github.com/user-attachments/assets/9ab10176-696f-4243-bf79-d1ba84ab15ee)

For this dataset, an additional regression model was included. All variables were selected, but the three interval variables were transformed into categorical variables using the quantile method to examine how odds ratios differ across groups. The odds ratios for age and family members are close to 1, indicating minimal differences between categories. This aligns with their lack of significance in previous models.

For net yearly income, the highest odds ratio is 1.17, suggesting that individuals in the lowest income category have 1.17 times higher odds of defaulting compared to those in the highest income group. For years employed, the highest odds ratio is 1.55, indicating that individuals with 0–2 years of employment have 1.55 times higher odds of defaulting compared to those employed for 7 or more years.
![image](https://github.com/user-attachments/assets/9df80423-babd-4519-80e5-5fce0fd39e9d)


# Additional Models

After analyzing models with and without credit-related variables, we attempted to segment the dataset to uncover additional insights. Initially, we combined the Variable Clustering Node with the Cluster Node, but this approach did not yield any significant results.

Next, we divided the dataset into two groups based on income, using the median to separate "poor" and "rich" customers. Models were rerun for each subset, both with and without credit-related variables. However, the metrics and selected variables remained largely consistent across both groups.

![image](https://github.com/user-attachments/assets/4354cd52-aa1d-430c-8321-07820c2fc5e2)
![image](https://github.com/user-attachments/assets/8c40da9e-8cc8-40ec-bbfa-ecc570ae86a2)
![image](https://github.com/user-attachments/assets/9f0c0e7e-a617-4b93-9020-6c676259f64c)


Finally, we segmented the dataset based on occupation type, separating observations into "professional" and "non-professional" categories. While this segmentation had little effect when credit variables were included, it produced noticeable differences when using only non-credit variables. The two tables below (professional on top, non-professional on bottom) display the accuracy and other metrics for the top five models. The results indicate that accuracy and other metrics were slightly higher for the professional group. Although the differences are minimal, they could serve as a valuable starting point for future research.
