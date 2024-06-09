import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

# Tạo thư mục lưu biểu đồ và mô hình nếu chưa tồn tại
output_folder = 'resources'
model_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Tải dữ liệu
datasets = {
    'FPT': pd.read_csv('data/merged_FPT_data.csv'),
    'HPG': pd.read_csv('data/merged_HPG_data.csv'),
    'VCB': pd.read_csv('data/merged_VCB_data.csv'),
    'VIC': pd.read_csv('data/merged_VIC_data.csv'),
    'VNM': pd.read_csv('data/merged_VNM_data.csv')
}

# Cấu hình tinh chỉnh mô hình
tuning_params = {
    'Ridge Regression': {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]},
    'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'Random Forest Regressor': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]},
    'XGBoost Regressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1]}
}

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Hàm tinh chỉnh và đánh giá mô hình
def tune_and_evaluate(X, y, model, params):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_params = {}
    if params:
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=kf, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = model
        best_model.fit(X, y)
        best_params = model.get_params()
    
    # Cross-validation scores
    cv_results = {
        'neg_mean_squared_error': -cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_squared_error').mean(),
        'neg_mean_absolute_error': -cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_absolute_error').mean(),
        'r2': cross_val_score(best_model, X, y, cv=kf, scoring='r2').mean()
    }
    
    # Calculate additional metrics
    cv_rmse = (cv_results['neg_mean_squared_error'])**0.5
    cv_mae = cv_results['neg_mean_absolute_error']
    cv_mse = cv_results['neg_mean_squared_error']
    cv_r2 = cv_results['r2']

    return best_model, cv_rmse, cv_mae, cv_mse, cv_r2, best_params

# Ánh xạ mô hình vào bộ datasets
results = []
for dataset_name, data in datasets.items():
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    target_column = [col for col in data.columns if 'close' in col.lower()][0]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, model in models.items():
        params = tuning_params.get(model_name, {})
        best_model, cv_rmse, cv_mae, cv_mse, cv_r2, best_params = tune_and_evaluate(X_train_scaled, y_train, model, params)

        # Lưu kết quả và dự đoán
        y_pred = best_model.predict(X_test_scaled)

        # Lưu kết quả trực quan hóa
        plt.figure()
        plt.plot(y_test.values, label='Thực tế')
        plt.plot(y_pred, label='Dự đoán', linestyle='dashed')
        plt.xlabel('Ngày')
        plt.ylabel('Giá cổ phiếu')
        plt.title(f'{model_name} Dự đoán vs Thực tế - {dataset_name}')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'{model_name}_{dataset_name}_pred_vs_actual.png'))
        plt.close()

        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'CV_RMSE': cv_rmse,
            'CV_MAE': cv_mae,
            'CV_MSE': cv_mse,
            'CV_R^2': cv_r2,
            'Best_Params': best_params
        })
        
        # Save the tuned model
        joblib.dump(best_model, os.path.join(model_folder, f'{model_name}_{dataset_name}_tuned.joblib'))

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Lưu kết quả vào CSV
results_df.to_csv('model_comparison_results.csv', index=False)

# Tạo biểu đồ so sánh các chỉ số hiệu suất
metrics = ['CV_RMSE', 'CV_MAE', 'CV_MSE', 'CV_R^2']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model_name in models.keys():
        subset = results_df[results_df['Model'] == model_name]
        plt.bar(subset['Dataset'], subset[metric], label=model_name)
    plt.xlabel('Dataset')
    plt.ylabel(metric)
    plt.title(f'So sánh {metric} giữa các mô hình')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'Comparison_{metric}.png'))
    plt.close()
