🏅 Beginner Machine Learning Project - Predicting Olympic Medals

📘 Overview
This project predicts the number of medals a team might win in the Olympics using historical data. The process involves formulating a hypothesis, training a linear regression model, and evaluating the model’s accuracy.

📂 Project Structure
Data Loading and Preprocessing:
Load the teams.csv dataset.
Select relevant columns: team, country, year, athletes, age, prev_medals, medals.
Handle missing values by dropping rows with null entries.

Exploratory Data Analysis (EDA):
Plot correlations between features (e.g., athletes, age) and medals.
Visualize data using scatter plots and histograms.

Model Training:
Split the data into training (year < 2012) and test sets (year >= 2012).
Train a linear regression model using athletes and prev_medals as predictors.

Predictions and Error Measurement:
Predict the number of medals for the test set.
Round negative predictions to zero.
Calculate the Mean Absolute Error (MAE) and error ratio by team.
Visualize the error distribution using histograms.

🛠️ Key Functions and Libraries

Pandas: Data manipulation and cleaning.
Matplotlib & Seaborn: Data visualization.
Scikit-learn: Model training and error calculation.
NumPy: Numerical computations.

▶️ Running the Project
Place your dataset (teams.csv) in the project directory.
Install necessary libraries:
pip install pandas matplotlib seaborn scikit-learn numpy

Run the script:
python your_script.py

Check the terminal output for error metrics and predictions.

📊 Results and Interpretation
The model predicts medal counts based on the number of athletes and previous medals won.
The error ratio histogram helps assess prediction accuracy across teams.

🚀 Next Steps
Add more features (e.g., GDP, training facilities).
Experiment with other models (e.g., Random Forest, XGBoost).
Fine-tune hyperparameters to improve accuracy.
