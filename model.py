# building a beginner machine learning project. This includes creating a hypothesis, setting up the model, and measuring error

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error
import numpy as np

teams = pd.read_csv("teams.csv")
available_columns = ["team", "country", "year", "athletes", "age", "prev_medals", "medals"]
existing_columns = [col for col in available_columns if col in teams.columns]
teams = teams[existing_columns]

#Exclude non-numeric columns
numeric_teams = teams.select_dtypes(include=['number'])

#check if 'medals' exists in numeric data
if "medals" in numeric_teams.columns:
    correlation = numeric_teams.corr()["medals"]
    print(correlation)
else: 
    print ("The 'medals' column is missing or not numeric")

#plot the chart 
sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True,ci=None)
sns.lmplot(x="age",y="medals",data=teams,fit_reg=True,ci=None)

#histogram 
teams.plot.hist(y="medals")

#cleaning the missing values(missing values are not zero, those are null spaces mainly )

#detecting the null values
teams[teams.isnull().any(axis=1)]
#drop the null value column
teams = teams.dropna()


train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

print("The training shape ",train.shape)
print("The test shape ",test.shape)

# #Show the charts
plt.show()


print(teams)


reg = LinearRegression()
predictions = ["athletes","prev_medals"]
target = "medals"


#training the model reg.fit(training data , target)
reg.fit(train[predictions],train["medals"])

predictions = reg.predict(test[predictions])
test["predictions"] = predictions

test.loc[test["predictions"] < 0 ,"predictions"] = 0
test["predictions"] = test["predictions"].round()

error = mean_absolute_error(test["medals"],test["predictions"])
teams.describe()["medals"]

print(test[test["team"] == "USA"])
print(test[test["team"] == "IND"])

error = (test["medals"]- predictions).abs()
print(error)

error_by_team = error.groupby(test["team"]).mean()
print(error_by_team)

medals_by_team = test.groupby("team")["medals"].mean()

# When dividing two Series, their indexes must match
error_by_team, medals_by_team = error_by_team.align(medals_by_team, join='inner')

error_ratio = error_by_team / medals_by_team
error_ratio = error_ratio.replace([np.inf, -np.inf], np.nan).dropna()



# error_ratio[~pd.isnull(error_ratio)]

# Clean up the error ratio
error_ratio = error_ratio.replace([np.inf, -np.inf], np.nan).dropna()
error_ratio.plot.hist(bins=20, edgecolor='black')

error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()

error_ratio.sort_values()

print(error_ratio)
