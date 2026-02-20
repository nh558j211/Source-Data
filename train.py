from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold

# 1. Load and preprocess data
data = pd.read_csv('./data_features.csv')
columns_to_drop = ["formula", "p Band Center"]
X = data.drop(columns_to_drop, axis=1)
y = data['p Band Center']

# 2. Split data into training and test sets (7:3 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, shuffle=True
)

# 3. Standardization: Fit scaler only on training set (avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform test set, do not refit
dump(scaler, 'feature_scaler.joblib')

# 4. Define models (optimize some hyperparameters)
models = [
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=1)),
    ('Linear Regression', LinearRegression()),
    ('AdaBoost', AdaBoostRegressor(n_estimators=50, random_state=1, learning_rate=0.1)),
    ('KNN', KNeighborsRegressor(n_neighbors=5)),
    ('MLP', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                         max_iter=1000, random_state=1, verbose=False))
]

# 5. Model training and evaluation (train on training set, evaluate on test set)
predictions = {'True_Value': y_test}  # Keep true values for comparison
for model_name, model in models:
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)  # Train only on training set
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train_scaled)
    r2_train = round(r2_score(y_train, y_train_pred), 3)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Evaluate on test set (core: check generalization ability)
    y_test_pred = model.predict(X_test_scaled)
    r2_test = round(r2_score(y_test, y_test_pred), 3)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f'{model_name} - Train R2: {r2_train}, Train RMSE: {rmse_train:.3f}')
    print(f'{model_name} - Test R2: {r2_test}, Test RMSE: {rmse_test:.3f}')
    
    predictions[model_name] = y_test_pred
    dump(model, f'{model_name.replace(" ", "_")}_model.joblib')

# 6. Save prediction results (including true values)
pd.DataFrame(predictions).to_csv('model_predictions.csv', index=False)

# 7. Cross-validation + hyperparameter tuning (take RF as example)
cv = KFold(n_splits=5, shuffle=True, random_state=1)
param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=1),
    param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Evaluate the best model
best_rf = grid_search.best_estimator_
test_r2 = r2_score(y_test, best_rf.predict(X_test_scaled))
test_rmse = np.sqrt(mean_squared_error(y_test, best_rf.predict(X_test_scaled)))

print('\nOptimal RF model (after cross-validation tuning):')
print(f'Best parameters: {grid_search.best_params_}')
print(f'Test set R2: {test_r2:.3f}, Test set RMSE: {test_rmse:.3f}')
dump(best_rf, 'best_random_forest_model.joblib')