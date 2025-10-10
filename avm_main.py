# Imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sodapy import Socrata
import warnings
import os
import requests
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


# Config
# Defining variables for socrata client
CURRENT_YEAR = '2026'
DOMAIN = "data.winnipeg.ca"
MAX_ROWS = 1000000
TARGET_COL = "total_assessed_value"
DATASET_ID = "d4mq-wa44"
# This is the odata endpoint
ODATA_URL = "https://data.winnipeg.ca/api/odata/v4/d4mq-wa44"

# Log_file location to be changed accordingly
LOG_FILE = "winnipeg_avm.log"

#os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Remove existing handlers and reconfigure
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info(" Logging started")

class WinnipegAVM:
    """
    Objective: To create an Automated Valuation Model for Winnipeg Assessment Parcels
    which predict the total_assessed_value based on property features
    """

    def __init__(self):
        self.preprocessor = None
        self.best_model = None
        self.best_score = float('-inf')
        self.results = {}

    # This function tries to load data from two sources: Socrata client and Odata endpoint
    # The reason is that socrata client fails to load data sometimes
    def load_dataset(self):
        logging.info("Starting dataset load")
        try:
            logging.info("Trying Socrata client")
            client = Socrata(DOMAIN, None, timeout=60)
            results = client.get(DATASET_ID, limit=MAX_ROWS)
            df = pd.DataFrame.from_records(results)
            return df
        except Exception as e:
            logger.warning("Socrata fetch failed:", {e})
    
            try:
                
                logging.info("Trying OData fetch:")
                resp = requests.get(ODATA_URL, params={"$top": MAX_ROWS}, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("value", [])
                return pd.DataFrame.from_records(data)
            except Exception as e2:
              # If Socrata and Odata fail, the reason might be that you are offline. Please connect to internet and try again
                raise RuntimeError("Failed to load dataset") from e2

    """
    This function involves exploring the characteristics of data and visualizing the data
    """
    def exploratory_data_analysis(self,df):
        # Making sure column names are consistent with underscore and lower case
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        logging.info("Data Summary:")
        # Dataset shape and data types
        logging.info(f"Dataset Shape {df.shape}")
        logging.info("Data Types:")
        logging.info(df.dtypes)
        
        # Finding out missing value % for each feature
        missing_pct = df.isnull().mean() * 100
        logging.info("Missing Values Percentage per Column:")
        logging.info(missing_pct[missing_pct > 0].sort_values(ascending=False))


        # Convert to numeric columns for further analysis
        candidate_numeric = [
            'total_living_area','assessed_land_area','year_built','rooms',
            'number_floors_condo','dwelling_units','water_frontage_measurement',
            'sewer_frontage_measurement','centroid_lat','centroid_lon',
            'total_assessed_value'
        ]
        numeric_cols = [c for c in candidate_numeric if c in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

        # target analysis
        null_count_target = df["total_assessed_value"].isna().sum()
        min_c = df["total_assessed_value"].min()
        max_c = df["total_assessed_value"].min()
        skewness = df["total_assessed_value"].skew()

        logging.info(f"Target Null Count:{null_count_target}")
        logging.info(f"Min Value for target variable{min_c}")
        logging.info(f"Max Value for target variable{max_c}")
        logging.info(f"Skewness of target variable: {skewness:.2f}")

        # Skewness plot for target variable
        plt.figure(figsize=(8, 4))
        sns.histplot(df["total_assessed_value"].dropna(), bins=50, kde=True, color="skyblue")
        plt.title("Distribution of Total Assessed Value\n(skewness = {:.2f})".format(skewness))
        plt.xlabel("Total Assessed Value ($)")
        plt.ylabel("Count")
        plt.xscale("log") 
        plt.show()

        # Binary features analysis
        bin_features = ['basement_finish',
                'basement',
                'air_conditioning',
                'fire_place',
                'attached_garage',
                'detached_garage',  
                'pool'    
                ]
        for x in bin_features:
          df[x] = df[x].map({'Yes': 1, 'No': 0})

        
        # Plotting to see the relationship between total assessed value and the binary features
        assessed_value = 'total_assessed_value'

        existing_cols = [col for col in bin_features if col in df.columns and assessed_value in df.columns]
        
        fig, axes = plt.subplots(nrows=len(existing_cols), ncols=1, figsize=(8, 5 * len(existing_cols)))

        if len(existing_cols) == 1:
            sns.boxplot(x=existing_cols[0], y=assessed_value, data=df, ax=axes)
            axes.set_title(f'Effect of {existing_cols[0]} on {assessed_value}')
        else:
            for i, col in enumerate(existing_cols):
                sns.boxplot(x=col, y=assessed_value, data=df, ax=axes[i])
                axes[i].set_title(f'Effect of {col} on {assessed_value}')
                axes[i].set_xticks([0, 1])
                axes[i].set_xticklabels(['No', 'Yes'])

        plt.tight_layout()
        plt.show()



        # Histogram of total_living_area, total_assessed_value
        key_numeric = ['total_living_area', 'total_assessed_value']
        for col in key_numeric:
            if col in df.columns:
                plt.figure(figsize=(10,4))
                sns.histplot(df[col].dropna(), bins=50, kde=True)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.show()

        # Bar plot counts of categorical variables to see the distribution
        cat_cols = ['neighbourhood_area', 'building_type', 'market_region']
        for col in cat_cols:
            if col in df.columns:
                plt.figure(figsize=(12,4))
                order = df[col].value_counts().index[:15]
                sns.countplot(data=df, x=col, order=order)
                plt.xticks(rotation=45)
                plt.title(f'Count of Properties by {col}')
                plt.show()

        
        
        # Violin plot for total_assessed_value v/s top 5 neighbourhoods
        if 'total_assessed_value' in df.columns and 'neighbourhood_area' in df.columns:
            top_neigh = df['neighbourhood_area'].value_counts().index[:5]
            plt.figure(figsize=(12,6))
            sns.violinplot(x='neighbourhood_area', y='total_assessed_value', data=df[df['neighbourhood_area'].isin(top_neigh)])
            plt.title('Violin Plot of Total Assessed Value by Top 5 Neighbourhoods')
            plt.xticks(rotation=45)
            plt.show()
        return df



    def load_and_prepare_data(self, df):
        """
        This function loads data and select relevant features for property valuation
        """
        logging.info("Preparing data and selecting features")
        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        # Replace spaces with underscores in column names
        df.columns = df.columns.str.replace(' ', '_')

        """
        Selecting the most informative features—that is those with low rates of missing data, 
        clear relevance to property value, and strong correlation with the target variable.
        """
        selected_features = [
            # Categorical variables, location features, these are very crucial for property valuation
            'neighbourhood_area', 'market_region',

            # These features help understand the size and structure and are primary value drivers
            'total_living_area', 'building_type', 'year_built', 'rooms',
            'assessed_land_area', 'number_floors_condo',

            # These are amenity featues
            'basement', 'basement_finish', 'air_conditioning', 'fire_place',
            'attached_garage', 'detached_garage', 'pool',

            # These help understand property characteristics 
            'property_use_code', 'zoning', 'multiple_residences', 'dwelling_units',
            'water_frontage_measurement', 'sewer_frontage_measurement', 'property_influences',

            # Target variable
            'total_assessed_value'
        ]

        # Keeping only selected features
        available_features = [col for col in selected_features if col in df.columns]

        logging.info(f"Selected {len(available_features)} features for AVM modeling")
        logging.info(f"Features: {', '.join(available_features)}")

        return df[available_features].copy()

    def engineer_features(self, df):

        """
        Creating new features to improve model performance
        """
        
        
        df = df.copy()

        logging.info("Starting Feature Engineering")

        # Converting binary features to numeric 
        binary_mappings = {
            'basement': {'Yes': 1, 'No': 0},
            'air_conditioning': {'Yes': 1, 'No': 0},
            'fire_place': {'Yes': 1, 'No': 0},
            'attached_garage': {'Yes': 1, 'No': 0},
            'detached_garage': {'Yes': 1, 'No': 0},
            'pool': {'Yes': 1, 'No': 0},
            'multiple_residences': {'Yes': 1, 'No': 0}
        }

        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # basement_finish
        if 'basement_finish' in df.columns:
            finish_mapping = {'Yes': 1, 'No': 0}
            df['basement_finish'] = df['basement_finish'].map(finish_mapping).fillna(0)

        # Creating new feature: Property Age (Current year -  Year built)
        
        current_year = 2026  
        if 'year_built' in df.columns:
            df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
            df['property_age'] = current_year - df['year_built']
            df['property_age'] = df['property_age'].clip(lower=0)  

        # Converting all amenity features into one feature. 
        # This will be a weighted feature, according to the amenity's importance to the property
        # Total score of 10

        amenity_weights = {
            'basement': 1.5,           # Adds value to the house
            'basement_finish': 1.5,    # Adds more value
            'air_conditioning': 1.5,   # Adds value in summers
            'fire_place': 0.5,         # Non essential but nice to have
            'attached_garage': 2.0,    # Highly valuable for Winnipeg winters
            'detached_garage': 1.0,    # Less valauable than attached garage
            'pool': 2.0                # Adds value to the property
        }

        df['amenity_score'] = 0
        for col, weight in amenity_weights.items():
            if col in df.columns:
                df['amenity_score'] += df[col] * weight

        # New feature: has_garage: 1/0
        if 'attached_garage' in df.columns and 'detached_garage' in df.columns:
            df['has_garage'] = ((df['attached_garage'] == 1) |
                              (df['detached_garage'] == 1)).astype(int)

        # New feature: Luxury_score. These amenities are usually found in high value homes
        luxury_features = ['pool', 'fire_place']
        df['luxury_score'] = 0
        for feature in luxury_features:
            if feature in df.columns:
                df['luxury_score'] += df[feature]

        # Removing binary columns since new features are now created
        columns_to_remove = ['basement', 'basement_finish', 'air_conditioning', 'fire_place',
                           'attached_garage', 'detached_garage', 'pool', 'year_built','water_frontage_measurement', 'sewer_frontage_measurement' ]

        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
                logger.info(f" Removed redundant columns: {col}")

        new_features = ['property_age', 'amenity_score', 'has_garage', 'luxury_score']

        available_new_features = [f for f in new_features if f in df.columns]

        logger.info(f"Created {len(available_new_features)} new features:")
        for feature in available_new_features:
            logger.info(f" {feature}")

        return df

    def preprocess_data(self, df):
        """
        This function has the data preprocessing pipeline which handles missing values,removes outliers,
        encodes categorical variables and performs scaling on numeric variables
        """
        logging.info("Performing Data Preprocessing:")

        # Perform feature engineering
        df = self.engineer_features(df)

        # to make sure df is a DataFrame, not Series
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # check for target column
        if 'total_assessed_value' not in df.columns:
            raise ValueError("Target column 'total_assessed_value' not found in dataset")

        # cleaning all numeric columns to remove commas
        def clean_numeric_columns(df, numeric_columns):
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '').replace('', np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        # cleaning columns
        df['total_assessed_value'] = df['total_assessed_value'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['total_assessed_value'] = pd.to_numeric(df['total_assessed_value'], errors='coerce')


        numeric_cols = ['total_living_area', 'rooms', 'assessed_land_area', 'dwelling_units',
                        'number_floors_condo',
                       'property_age', 'amenity_score', 'has_garage', 'luxury_score']
        df = clean_numeric_columns(df, numeric_cols)

        df = df.dropna(subset=['total_assessed_value'])

        # seperating features and target
        X = df.drop('total_assessed_value', axis=1)
        y = df['total_assessed_value'].copy()

        logger.info(f"Initial dataset: {X.shape[0]} properties, {X.shape[1]} features")
        logger.info(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
        logger.info(f"Target mean: ${y.mean():,.0f}")

        # Removing outliers using percentile trimming 1st–99th percentiles
        # IQR method cannot be used since it trims the dataset to 1- 700k value properties
        lower_pct, upper_pct = 0.01, 0.99

        lower_bound = y.quantile(lower_pct)
        upper_bound = y.quantile(upper_pct)

        outlier_mask = y.between(lower_bound, upper_bound)
        X_clean = X[outlier_mask].copy()
        y_clean = y[outlier_mask].copy()

        outliers_removed = len(y) - len(y_clean)
        logger.info(f"Removed {outliers_removed} outliers ({outliers_removed/len(y)*100:.1f}%)")
        logger.info(f"Clean dataset: {X_clean.shape[0]} properties")


        # Convert the 'dwelling_units' column from categorical strings to numeric values
        # to enable more accurate modeling and mathematical operations

        if 'dwelling_units' in X_clean.columns:
            dwelling_mapping = {'1': 1, '2': 2, '3': 3, '4+': 4}
            X_clean['dwelling_units'] = X_clean['dwelling_units'].map(dwelling_mapping).fillna(1).astype(int)

        # feature types for preprocessing
        numeric_features = []
        categorical_features = []

        for col in X_clean.columns:
            if col in ['total_living_area', 'rooms', 'assessed_land_area', 'number_floors_condo',
                       'multiple_residences',
                      'dwelling_units', 'property_age', 'amenity_score', 'has_garage', 'luxury_score']:
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

        # preprocessing pipelines
        # using KNN imputer for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        # using most frequent strategy for categorical imputation
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # combining transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        return X_clean, y_clean

    def train_models(self, X, y):
        """
        This function trains multiple regression models: Random forest regressor, XGB regressor with hyperparameter tuning
        to find the best performing model for property valuation
        """
        logger.info("Performing Model Training:")

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"Training: {X_train.shape[0]} properties")
        logger.info(f"Testing: {X_test.shape[0]} properties")

        # Using only two models due to compute limitations
        models_and_params = {
            'Random Forest': {
                'model': RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'regressor__n_estimators': [100 ],
                    'regressor__max_depth': [10 ],
                    'regressor__min_samples_split': [ 5]
                }
            },
            'XGBoost': {
                'model': XGBRegressor(
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1,
                    verbosity=0
                ),
                'params': {
                    'regressor__n_estimators': [200],
                    'regressor__learning_rate': [0.1],
                    'regressor__max_depth': [10]
                }
            }
        }

        best_models = {}

        for name, model_config in models_and_params.items():
            logger.info(f"Training {name}...")

            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model_config['model'])
            ])

            # Hyperparameter tuning
            if model_config['params']:
                search = RandomizedSearchCV(
                    pipeline,
                    model_config['params'],
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_iter=5,
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                pipeline.fit(X_train, y_train)
                best_model = pipeline
                best_params = {}

            # Model evaluation
            y_pred = best_model.predict(X_test)

            # Calculating metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # Calculating Mean Absolute Percentage Error
            # computed on y>100 to avoid division issues
            mask = y_test > 100  
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                mape = float('inf')

            # results
            best_models[name] = {
                'model': best_model,
                'params': best_params,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'predictions': y_pred,
                'y_test': y_test
            }

            logger.info(f"  R² Score: {r2:.4f}")
            logger.info(f"  MAE: ${mae:,.0f}")
            logger.info(f"  MAPE: {mape:.2f}%")

            # Tracking best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = best_model

        self.results = best_models
        return best_models

    def evaluate_best_model(self):
        """
        This provides a detailed evaluation of the best performing model
        """
        logger.info("Model evaluation:")

        # Find best model
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_result = self.results[best_name]

        logger.info(f"Best Model: {best_name}")
        logger.info(f"Best Parameters: {best_result['params']}")
        logger.info(f"Performance Metrics:")
        logger.info(f"  R² Score: {best_result['r2']:.4f} ({best_result['r2']*100:.1f}% variance explained)")
        logger.info(f"  MAPE: {best_result['mape']:.2f}%")
        logger.info(f"  MAPE: {best_result['mape']:.2f}%")

        # prediction accuracy 
        predictions = best_result['predictions']
        actual = best_result['y_test']

        within_5_percent = np.mean(np.abs(predictions - actual) / actual <= 0.05) * 100
        within_10_percent = np.mean(np.abs(predictions - actual) / actual <= 0.10) * 100
        within_15_percent = np.mean(np.abs(predictions - actual) / actual <= 0.15) * 100

        logger.info(f"Prediction Accuracy:")
        logger.info(f"  Within 5% of actual value: {within_5_percent:.1f}%")
        logger.info(f"  Within 10% of actual value: {within_10_percent:.1f}%")
        logger.info(f"  Within 15% of actual value: {within_15_percent:.1f}%")

        # To find where the model is performing best: low, mid and high quantile range
        low_value = actual <= actual.quantile(0.33)
        mid_value = (actual > actual.quantile(0.33)) & (actual <= actual.quantile(0.67))
        high_value = actual > actual.quantile(0.67)

        logger.info(f"Performance by Value Range:")
        logger.info(f"  Low value (≤${actual.quantile(0.33):,.0f}): MAE = ${mean_absolute_error(actual[low_value], predictions[low_value]):,.0f}")
        logger.info(f"  Mid value (${actual.quantile(0.33):,.0f}-${actual.quantile(0.67):,.0f}): MAE = ${mean_absolute_error(actual[mid_value], predictions[mid_value]):,.0f}")
        logger.info(f"  High value (>${actual.quantile(0.67):,.0f}): MAE = ${mean_absolute_error(actual[high_value], predictions[high_value]):,.0f}")

        return best_name, best_result


    def model_summary(self):
        """
        To print model summary
        """
        
        logger.info("AVM summary :")
       

        logger.info(f"\n{'Model':<20} {'R² Score':<10} {'RMSE':<15} {'MAE':<15} {'MAPE (%)':<10}")
     
        for name, results in self.results.items():
            logger.info(f"{name:<20} {results['r2']:<10.4f} ${results['rmse']:<14,.0f} ${results['mae']:<14,.0f} {results['mape']:<10.2f}")

        best_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_result = self.results[best_name]

        logger.info(f"BEST MODEL: {best_name}")
        logger.info(f" Explains {best_result['r2']:.1%} of property value variance")
        logger.info(f" Average prediction error: ${best_result['mae']:,.0f}")
        logger.info(f" Average percentage error: {best_result['mape']:.1f}%")

        # Accuracy metrics
        predictions = best_result['predictions']
        actual = best_result['y_test']
        within_10_percent = np.mean(np.abs(predictions - actual) / actual <= 0.10) * 100

        logger.info(f"{within_10_percent:.1f}% of predictions within 10% of actual value")


# Usage:


# Initialize and train AVM
avm = WinnipegAVM()
df = avm.load_dataset()

# eda
avm.exploratory_data_analysis(df)
# Prepare data
df_clean = avm.load_and_prepare_data(df)
X, y = avm.preprocess_data(df_clean)

# Train models
model_results = avm.train_models(X, y)

# Evaluate best model
best_model_name, best_results = avm.evaluate_best_model()


# Print summary
avm.model_summary()

