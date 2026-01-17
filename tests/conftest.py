"""
Pytest configuration and shared fixtures
"""
import tempfile
import warnings
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# Suppress convergence warnings for models that fail to converge on random test data
# This is expected behavior with synthetic data
warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", module="sklearn.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample credit default data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame(
        {
            "ID": range(1, n_samples + 1),
            "LIMIT_BAL": np.random.uniform(10000, 500000, n_samples),
            "SEX": np.random.choice([1, 2], n_samples),
            "EDUCATION": np.random.choice([1, 2, 3, 4], n_samples),
            "MARRIAGE": np.random.choice([1, 2, 3], n_samples),
            "AGE": np.random.randint(20, 80, n_samples),
            "PAY_0": np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
            "PAY_2": np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
            "PAY_3": np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
            "PAY_4": np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
            "PAY_5": np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
            "PAY_6": np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
            "BILL_AMT1": np.random.uniform(0, 500000, n_samples),
            "BILL_AMT2": np.random.uniform(0, 500000, n_samples),
            "BILL_AMT3": np.random.uniform(0, 500000, n_samples),
            "BILL_AMT4": np.random.uniform(0, 500000, n_samples),
            "BILL_AMT5": np.random.uniform(0, 500000, n_samples),
            "BILL_AMT6": np.random.uniform(0, 500000, n_samples),
            "PAY_AMT1": np.random.uniform(0, 100000, n_samples),
            "PAY_AMT2": np.random.uniform(0, 100000, n_samples),
            "PAY_AMT3": np.random.uniform(0, 100000, n_samples),
            "PAY_AMT4": np.random.uniform(0, 100000, n_samples),
            "PAY_AMT5": np.random.uniform(0, 100000, n_samples),
            "PAY_AMT6": np.random.uniform(0, 100000, n_samples),
            "default payment next month": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )
    return data


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def data_dir_with_files(temp_dir: Path, sample_data: pd.DataFrame) -> Path:
    """
    Create a data directory with monthly CSV files that ModelTrainer expects.
    This simulates the data setup that happens in ModelTrainer.__init__
    
    ModelTrainer.__init__ checks for files like data_01.csv through data_12.csv.
    If they don't exist, it tries to read from "default of credit card clients.xls".
    By creating these files here, we ensure the initialization doesn't fail.
    """
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create monthly data files (data_01.csv through data_12.csv)
    # Use the sample_data and split it across months
    np.random.seed(42)  # Ensure reproducibility
    sample_data_with_month = sample_data.copy()
    sample_data_with_month.insert(1, column="MONTH", value=np.random.randint(1, 13, size=len(sample_data)))
    
    # Ensure each month has at least some data
    for month in range(1, 13):
        monthly_data = sample_data_with_month[sample_data_with_month["MONTH"] == month].copy()
        
        # If a month has no data, use a subset of the sample data
        if len(monthly_data) == 0:
            # Take a small subset and assign it to this month
            monthly_data = sample_data.head(10).copy()
            monthly_data.insert(1, column="MONTH", value=month)
        
        # Ensure MONTH column is present and set correctly
        if "MONTH" not in monthly_data.columns:
            monthly_data.insert(1, column="MONTH", value=month)
        else:
            monthly_data["MONTH"] = month
        
        # Save to CSV file
        monthly_data.to_csv(data_dir / f"data_{month:02d}.csv", index=False)
    
    # Verify all expected files were created
    expected_files = [f"data_{i:02d}.csv" for i in range(1, 13)]
    created_files = [f.name for f in data_dir.iterdir() if f.is_file()]
    for expected_file in expected_files:
        assert expected_file in created_files, f"Failed to create {expected_file}"
    
    return data_dir


@pytest.fixture
def sample_train_test_data(sample_data: pd.DataFrame):
    """Create train/test split from sample data"""
    from sklearn.model_selection import train_test_split
    
    X = sample_data.drop(columns=["default payment next month", "ID"])
    y = sample_data["default payment next month"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
