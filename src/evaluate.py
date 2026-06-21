import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    print(f"\n{name} Evaluation:")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  R2   : {r2:.4f}")
    
    return {'RMSE': round(rmse, 2), 
            'MAE':  round(mae, 2), 
            'R2':   round(r2, 4)}

def compare_models(results):
    results_df = pd.DataFrame(results).T
    print("\nFull Model Comparison:")
    print(results_df)
    results_df.to_csv('outputs/model_comparison.csv')
    print("Saved to outputs/model_comparison.csv")
    return results_df

def plot_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['RMSE', 'MAE', 'R2']
    colors  = ['coral', 'steelblue', 'green']

    for i, metric in enumerate(metrics):
        axes[i].bar(results_df.index, 
                    results_df[metric], 
                    color=colors[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=15)

    plt.suptitle('Model Performance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png')
    plt.show()
    print("Plot saved to outputs/")

def plot_predictions(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{model_name} — Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_predictions.png')
    plt.show()
    print(f"Plot saved!")

def feature_importance(model, X_train, top_n=15):
    importance_df = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', 
                data=importance_df, palette='viridis')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.show()

    print("\nTop 5 Features:")
    print(importance_df.head())
    return importance_df

if __name__ == "__main__":
    print("Evaluate module ready!")
    print("Functions available:")
    print("  evaluate_model(name, model, X_test, y_test)")
    print("  compare_models(results)")
    print("  plot_comparison(results_df)")
    print("  plot_predictions(model, X_test, y_test, model_name)")
    print("  feature_importance(model, X_train)")