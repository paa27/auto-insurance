from setuptools import setup, find_packages

setup(
    name="auto-insurance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "optuna>=3.3.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "openpyxl>=3.1.0",
        "category_encoders>=2.0.0",
        "pymoo>=0.5.0"
    ],
    python_requires=">=3.10",
)
