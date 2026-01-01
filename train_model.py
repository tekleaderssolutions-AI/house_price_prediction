# Main training script using OOP classes
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_saver import ModelSaver

def main():
    """Main function to orchestrate the training pipeline using OOP classes."""
    print("="*60)
    print("House Price Prediction - Model Training Pipeline")
    print("="*60)
    
    # Step 1: Preprocess data
    preprocessor = DataPreprocessor('enhanced_house_price_dataset.csv')
    X_encoded, y, label_encoders = preprocessor.preprocess()
    feature_names = preprocessor.get_feature_names()
    
    # Step 2: Train models
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X_encoded, y)
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Step 3: Display comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    comparison_df = trainer.get_comparison_dataframe()
    print("\n", comparison_df.to_string(index=False))
    
    # Step 4: Get best model
    best_result = trainer.get_best_model()
    print(f"\n{'='*60}")
    print(f"Best Model: {best_result.name}")
    print(f"R-squared Score: {best_result.r2_score:.4f}")
    print(f"Mean Absolute Error: ₹{best_result.mae:,.2f}")
    print(f"{'='*60}")
    
    # Step 5: Save models
    saver = ModelSaver(models_dir='models')
    model_info = {
        'model_name': best_result.name,
        'r2_score': best_result.r2_score
    }
    saver.save_all(
        model=best_result.model,
        label_encoders=label_encoders,
        feature_names=feature_names,
        model_info=model_info
    )
    
    print("\nModel training completed successfully!")

if __name__ == '__main__':
    main()
