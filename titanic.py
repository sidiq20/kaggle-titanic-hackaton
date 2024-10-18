import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import re

# Step 1: Load the data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/tested.csv')  # Ensure the filename is correct

# Step 2: Feature Engineering
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# Fill missing Embarked values with 'S' and convert to numeric
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train_data['Embarked'] = train_data['Embarked'].fillna('S').map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].fillna('S').map(embarked_mapping)

# Create FamilySize and Title features
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\n.', name)
    if title_search:
        return title_search.group(1)
    return ""

train_data['Title'] = train_data['Name'].apply(extract_title)
test_data['Title'] = test_data['Name'].apply(extract_title)

title_mapping = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 5,
    'Countess': 5, 'Ms': 5, 'Lady': 5, 'Jonkheer': 5, 'Don': 5, 'Dona': 5, 'Mme': 5, 'Capt': 5, 'Sir': 5}
train_data['Title'] = train_data['Title'].map(title_mapping).fillna(5)
test_data['Title'] = test_data['Title'].map(title_mapping).fillna(5)

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = train_data[features].copy()
y = train_data['Survived']
X_test = test_data[features].copy()

# Handle missing values for numerical features
numerical_features = ['Age', 'Fare']
imputer = SimpleImputer(strategy='median')
X.loc[:, numerical_features] = imputer.fit_transform(X[numerical_features])
X_test.loc[:, numerical_features] = imputer.transform(X_test[numerical_features])

# Step 3: Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Step 4: Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Hyperparameter tuning with GridSearchCV for individual models

# Logistic Regression
log_reg_params = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
log_reg = GridSearchCV(LogisticRegression(max_iter=200), log_reg_params, cv=5)
log_reg.fit(X_train, y_train)

# Random Forest
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [4, 6, 8]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf.fit(X_train, y_train)

# Gradient Boosting
gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5)
gb.fit(X_train, y_train)

# Step 6: Ensemble Method - Voting Classifier (Majority Voting)
voting_clf = VotingClassifier(
    estimators=[
        ('log_reg', log_reg.best_estimator_),
        ('rf', rf.best_estimator_),
        ('gb', gb.best_estimator_)
    ], voting='hard')

voting_clf.fit(X_train, y_train)

# Step 7: Validation
for name, model in zip(['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Voting Classifier'],
                       [log_reg, rf, gb, voting_clf]):
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f"\n{name} Validation Accuracy: {accuracy:.4f}")

# Step 8: Make predictions on the test data with the best model (Voting Classifier)
test_predictions = voting_clf.predict(X_test)

# Step 9: Save predictions to a CSV file for Kaggle submission
output = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
output.to_csv('submission.csv', index=False)
print("Your submission file has been saved!")
