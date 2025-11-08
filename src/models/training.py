# src/models/training.py
"""
ML model training pipeline with multiple algorithms and hyperparameter optimization
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'elastic_net': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
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
                'model': GradientBoostingRegressor(random_state=settings.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=settings.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=settings.RANDOM_STATE, verbose=-1),
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
    
    def _train_model(self, model_name: str, model: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Train a given model with optional hyperparameter tuning"""
        if params:
            grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error')
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(self.X_train, self.y_train)
        
        self.models[model_name] = best_model   


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
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = -grid_search.best_score_  # Convert back to positive MAE
                
                # Log best parameters
                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)
                
                mlflow.log_metric("cv_mae", best_score)
                
                logger.info(f"Best parameters for {model_name}: {best_params}")
                logger.info(f"Best CV MAE: {best_score:.4f}")
                
            else:
                logger.info(f"Training {model_name} with default parameters")
                best_model = base_model
                best_model.fit(X_train, y_train)
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
        