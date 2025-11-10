# src/models/training.py
"""
ML model training pipeline with multiple algorithms and hyperparameter optimization
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
# from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
import json
import os
from src.config.settings import settings
from src.utils.logging import setup_logging
from sklearn.utils.class_weight import compute_sample_weight
logger = setup_logging(__name__)

class ModelTrainer:
    """Class for training and evaluating ML models for materials property prediction"""
    
    def __init__(self, experiment_name: str = "credit_default_prediction",  
                #  data: pd.DataFrame, 
                #  target_column: str, model_dir: Path
                 ):
        self.experiment_name = experiment_name
        self.models = self._setup_models()
        self.scalers = self._setup_scalers()
        self.trained_models = {}
        self.results = {}

        # Setup MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        # Setup data, assign an aribrary month to each row and split into monthly files if not present
        logger.info("Setting up data...")
        try:
            data_dir = Path(settings.DATA_DIR)
            expected = [f"data_{i:02d}.csv" for i in range(1, 13)]
            files = set(os.listdir(data_dir))
            missing = [f for f in expected if f not in files]
            all_present = not missing
            if not all_present:    
                data = pd.read_excel(data_dir / 'default of credit card clients.xls', header=1)
                data.insert(1, column='MONTH', value = np.random.randint(1, 13, size=(data.shape[0])))
                for month in range(1, 13):
                    monthly_data = data[data['MONTH'] == month]
                    monthly_data.to_csv(data_dir / f"data_{month:02d}.csv", index=False)
            logger.info("Data setup complete.")
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise

        # self.data = data
        # self.target_column = target_column
        # self.model_dir = model_dir
        # self.models: Dict[str, Any] = {}
        # self.scalers: Dict[str, Any] = {}
        # self.results: Dict[str, Dict[str, float]] = {}
        
        # self.X = self.data.drop(columns=[self.target_column])
        # self.y = self.data[self.target_column]
        
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     self.X, self.y, test_size=0.2, random_state=42
        # )
    def _setup_models(self) -> Dict[str, Any]:
        """Setup ML models with hyperparameter grids"""
        models = {
            'logistic_regression': {
                'model': LogisticRegression(),
                'params': {}
            },
            'ridge': {
                'model': RidgeClassifier(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'svc': {
                'model': SVC(probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
            },
            # # 'lasso': {
            # #     'model': Lasso(),
            # #     'params': {
            # #         'alpha': [0.01, 0.1, 1.0, 10.0]
            # #     }
            # # },
            # # 'elastic_net': {
            # #     'model': ElasticNet(),
            # #     'params': {
            # #         'alpha': [0.01, 0.1, 1.0],
            # #         'l1_ratio': [0.1, 0.5, 0.9]
            # #     }
            # },
            'random_forest': {
                'model': RandomForestClassifier(random_state=settings.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=settings.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=settings.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=settings.RANDOM_STATE, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100]
                }
            },
            # 'neural_network': {
            #     'model': MLPRegressor(random_state=settings.RANDOM_STATE, max_iter=1000),
            #     'params': {
            #         'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
            #         'alpha': [0.0001, 0.001, 0.01],
            #         'learning_rate_init': [0.001, 0.01]
            #     }
            # }
        }
        
        
        logger.info(f"Setup {len(models)} ML models")
        return models
    
    
    def _setup_scalers(self) -> Dict[str, Any]:
        """Setup feature scalers"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        return scalers

    def _scale_data(self, scaler_type: str = "standard") -> None:
        """Scale features using specified scaler"""
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.scalers[scaler_type] = scaler


    def prepare_data(
        self, 
        features_df: pd.DataFrame, 
        target_column: str = 'default payment next month',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training
        
        Args:
            features_df: DataFrame with features
            target_column: Target variable column name
            test_size: Test set size
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training")
        
        # Select feature columns (exclude target and metadata)
        exclude_columns = [
            target_column, 'ID', 'MONTH'  # Exclude categorical and IDs
        ]
        
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        # Remove non-numeric columns
        X = features_df[feature_columns].select_dtypes(include=[np.number])
        y = features_df[target_column]
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")
        logger.info(f"Selected {len(feature_columns)} feature columns")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=settings.RANDOM_STATE
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        use_hyperparameter_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Train a single model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            use_hyperparameter_optimization: Whether to use hyperparameter optimization
            
        Returns:
            Training results
        """
        logger.info(f"Training {model_name} model")
        
        model_config = self.models[model_name]
        base_model = model_config['model']
        param_grid = model_config['params']
        # compute per-sample weights to account for class imbalance
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            
            if use_hyperparameter_optimization and param_grid:
                logger.info(f"Performing hyperparameter optimization for {model_name}")
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=settings.CV_FOLDS, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1,
                    
                )
                
                grid_search.fit(X_train, y_train, sample_weight=sample_weight)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_  # Convert back to positive MAE
                
                # Log best parameters
                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)
                
                mlflow.log_metric("cv_auc", best_score)
                
                logger.info(f"Best parameters for {model_name}: {best_params}")
                logger.info(f"Best CV auc: {best_score:.4f}")
                
            else:
                logger.info(f"Training {model_name} with default parameters")
                best_model = base_model
                best_model.fit(X_train, y_train, sample_weight=sample_weight)
                best_params = {}
                best_score = None
            
            # Store trained model
            self.trained_models[model_name] = {
                'model': best_model,
                'params': best_params,
                'cv_score': best_score
            }
            
            # Log model
            if 'xgboost' in model_name.lower():
                mlflow.xgboost.log_model(best_model, f"{model_name}_model")
            elif 'lightgbm' in model_name.lower():
                mlflow.lightgbm.log_model(best_model, f"{model_name}_model")
            else:
                mlflow.sklearn.log_model(best_model, f"{model_name}_model")
            
            logger.info(f"Completed training {model_name}")
            
            return {
                'model': best_model,
                'best_params': best_params,
                'cv_score': best_score
            }    


    def evaluate_model(
        self, 
        model_name: str, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics        
        labels = np.unique(y_test)
        average = 'binary' if labels.size == 2 else 'macro'

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_test, y_pred, average=average, zero_division=0)
        }

        # Attempt to compute ROC AUC where applicable
        roc_auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if labels.size == 2:
                    roc_auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
            elif hasattr(model, "decision_function") and labels.size == 2:
                scores = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, scores)
        except Exception:
            roc_auc = None

        if roc_auc is not None:
            metrics['roc_auc'] = float(roc_auc)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            for metric, value in metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
        
        logger.info(f"Evaluation metrics for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        use_hyperparameter_optimization: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            use_hyperparameter_optimization: Whether to use hyperparameter optimization
            
        Returns:
            Results for all models
        """
        logger.info("Training all models")
        
        all_results = {}
        
        for model_name in self.models.keys():            
            try:
                if model_name in ['logistic_regression','ridge', 'svc']:
                    logger.info(f"Scaline data for model: {model_name}")
                    X_train_scaled = self.scalers['standard'].fit_transform(X_train)
                    X_test_scaled = self.scalers['standard'].transform(X_test)
                    
                    # Train model
                    training_result = self.train_model(
                        model_name, X_train_scaled, y_train, use_hyperparameter_optimization
                    )
                    # Evaluate model
                    evaluation_result = self.evaluate_model(model_name, X_test_scaled, y_test)

                    # Combine results
                    all_results[model_name] = {
                    'training': training_result,
                    'evaluation': evaluation_result
                    }
                else:
                    # Train model
                    training_result = self.train_model(
                        model_name, X_train, y_train, use_hyperparameter_optimization
                    )
                    
                    # Evaluate model
                    evaluation_result = self.evaluate_model(model_name, X_test, y_test)
                
                    # Combine results
                    all_results[model_name] = {
                        'training': training_result,
                        'evaluation': evaluation_result
                    }
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.results = all_results
        logger.info(f"Completed training {len(all_results)} models")
        
        return all_results
    
    def get_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """
        Get the best performing model based on test score
        
        Returns:
            Model name, model object, and metrics
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_model_name = None
        best_score = float('-inf')
        
        for model_name, results in self.results.items():
            score = results['evaluation']['f1']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        best_model = self.trained_models[best_model_name]['model']
        best_metrics = self.results[best_model_name]['evaluation']
        
        logger.info(f"Best model: {best_model_name} (F1: {best_score:.4f})")
        
        return best_model_name, best_model, best_metrics
    
    def save_models(self, output_dir: Optional[Path] = None) -> Path:
        """Save all trained models"""
        if output_dir is None:
            output_dir = settings.MODELS_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_info in self.trained_models.items():
            model_path = output_dir / f"{model_name}_model.joblib"
            joblib.dump(model_info['model'], model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        # save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_dir / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {scaler_name} scaler to {scaler_path}")

        # Save results summary
        results_path = output_dir / "training_results.json"
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = results['evaluation']
        
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Models saved to {output_dir}")
        return output_dir
        