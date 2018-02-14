# HousePricePredict
Category : Data Science - Subject : Predict a House Price

### Project Story:
This is my first experience in data science area. In Business Intelligence subject in the last course of the Computer Science
degree, I got the oportunity to make my first experience at data science area. I was so lucky so I got [Paco Herrera][1] as a profesor 
for this subject, so I tried to learn the most of him.

### Project Description:
It's a regression problem. It's a [Kaggle competiton][2] with the following description:
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home

### Followed Steps:
1. Study the problem(problem's natural, general view over the problem from a normal user's "not data scientist" perspective)
2. Study the variables (the most correlated ones, using a minimum umbral value to select the variables)
3. Make an iterative characteristics's selection process.
   - Select the most selected variables + the most correlated variables.
4. Detect the missing values.
5. Delete the variables with a high percentage of missing values (using an umbral of 90%)
6. Missing values imputation (using different ways)
7. Complete the predictive model by fitting different regression algorithms, using weighted mean strategy.
8. Try different weight for each algorithm, different algorithms, different algorithm's parameters values.

### Versions:
In the project src directory, I put a folder for each effective version. "effective version: gives higher ranking"

### Project Structure
- In src directoy, we can find a folder for each effective commit, with the submission file and the obtained ranking.
- In Data directory, you can find the train and test data sets.

[1]:http://decsai.ugr.es/~herrera/
[2]:https://www.kaggle.com/c/house-prices-advanced-regression-techniques
