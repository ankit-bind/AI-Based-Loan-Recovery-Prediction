import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException

logger = get_logger(__name__)

class FeatureEngineering:
    def __init__(self, config):
        self.config = config

    def clean_application_data(self, df):
        """Replicate data cleaning from notebook 02."""
        logger.info("Cleaning application data")
        
        # Remove low-information document features
        remove_docs = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7', 
                       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17']
        df = df.drop(columns=remove_docs, errors='ignore')
        
        # Remove near-constant mobile feature
        df = df.drop(columns=['FLAG_MOBIL'], errors='ignore')
        
        # Create employment anomaly flag (365243 is a sentinel value)
        df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        
        # Replace anomaly with NaN for ratio calculations
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
        
        # CODE_GENDER XNA fix — replace with F (mode)
        if 'CODE_GENDER' in df.columns:
            df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')

        # OWN_CAR_AGE imputation (Section 2.5 of cleaning and Section 8.3 of feature engineering notebooks)
        if 'OWN_CAR_AGE' in df.columns and 'FLAG_OWN_CAR' in df.columns:
            df.loc[df['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'] = df.loc[df['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'].fillna(0)
            car_owner_median = df.loc[df['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR_AGE'].median()
            if not np.isnan(car_owner_median):
                df.loc[df['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR_AGE'] = df.loc[df['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR_AGE'].fillna(car_owner_median)
            df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)
        
        logger.info(f"Cleaned application shape: {df.shape}")
        return df

    def aggregate_bureau_data(self, bureau_df):
        """Aggregate bureau data by SK_ID_CURR."""
        logger.info("Aggregating bureau data")
        
        # Precompute boolean columns to speed up aggregation
        bureau_df = bureau_df.copy()
        bureau_df['is_active'] = (bureau_df['CREDIT_ACTIVE'] == 'Active').astype(int)
        bureau_df['is_closed'] = (bureau_df['CREDIT_ACTIVE'] == 'Closed').astype(int)
        
        # Advanced and basic aggregates matching notebook logic
        bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(
            # Basic aggregates
            SK_ID_BUREAU_count=('SK_ID_BUREAU', 'count'),
            AMT_CREDIT_SUM_sum=('AMT_CREDIT_SUM', 'sum'),
            AMT_CREDIT_SUM_mean=('AMT_CREDIT_SUM', 'mean'),
            AMT_CREDIT_SUM_DEBT_sum=('AMT_CREDIT_SUM_DEBT', 'sum'),
            AMT_CREDIT_SUM_DEBT_mean=('AMT_CREDIT_SUM_DEBT', 'mean'),
            CREDIT_DAY_OVERDUE_max=('CREDIT_DAY_OVERDUE', 'max'),
            # Advanced aggregates
            bureau_total_loans=('SK_ID_BUREAU', 'count'),
            bureau_active_loans=('is_active', 'sum'),
            bureau_closed_loans=('is_closed', 'sum'),
            bureau_total_credit=('AMT_CREDIT_SUM', 'sum'),
            bureau_total_debt=('AMT_CREDIT_SUM_DEBT', 'sum'),
            bureau_total_overdue=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
            bureau_total_prolonged=('CNT_CREDIT_PROLONG', 'sum')
        ).reset_index()
        
        logger.info(f"Bureau aggregated shape: {bureau_agg.shape}")
        return bureau_agg

    def create_financial_ratios(self, df):
        """Create financial ratio features from notebook 03."""
        logger.info("Creating financial ratio features")
        
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
        df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        
        return df

    def create_family_features(self, df):
        """Create family-related features."""
        logger.info("Creating family features")
        
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1)
        
        return df

    def create_external_credit_features(self, df):
        """Create external credit score features."""
        logger.info("Creating external credit features")
        
        df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['EXT_SOURCE_1_MISSING'] = df['EXT_SOURCE_1'].isnull().astype(int)
        
        return df

    def create_collateral_feature(self, df):
        """Create collateral ownership feature."""
        logger.info("Creating collateral feature")
        
        df['HAS_COLLATERAL'] = ((df['FLAG_OWN_CAR'] == 'Y') | (df['FLAG_OWN_REALTY'] == 'Y')).astype(int)
        
        return df

    def create_bureau_features(self, df):
        """Create features from bureau aggregates."""
        logger.info("Creating bureau-derived features")
        
        df['BUREAU_DEBT_RATIO'] = df['AMT_CREDIT_SUM_DEBT_sum'] / (df['AMT_CREDIT_SUM_sum'] + 1)
        df['AVG_CREDIT_PER_LOAN'] = df['AMT_CREDIT_SUM_sum'] / (df['SK_ID_BUREAU_count'] + 1)
        df['DEBT_PER_LOAN'] = df['AMT_CREDIT_SUM_DEBT_sum'] / (df['SK_ID_BUREAU_count'] + 1)
        
        # Advanced bureau features
        df['ACTIVE_LOAN_RATIO'] = df['bureau_active_loans'] / (df['bureau_total_loans'] + 1)
        df['CLOSED_LOAN_RATIO'] = df['bureau_closed_loans'] / (df['bureau_total_loans'] + 1)
        df['OVERDUE_PER_LOAN'] = df['bureau_total_overdue'] / (df['bureau_total_loans'] + 1)
        df['PROLONGED_LOAN_RATIO'] = df['bureau_total_prolonged'] / (df['bureau_total_loans'] + 1)
        df['CREDIT_UTILIZATION_RATIO'] = df['bureau_total_debt'] / (df['bureau_total_credit'] + 1)
        
        return df

    def create_age_employment_features(self, df):
        """Create age and employment features."""
        logger.info("Creating age/employment features")
        
        df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365
        df['EMPLOYMENT_YEARS'] = np.abs(df['DAYS_EMPLOYED']) / 365
        df['EMPLOYMENT_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
        
        return df

    def create_document_features(self, df):
        """Aggregate remaining document flags."""
        logger.info("Creating document aggregate features")
        
        doc_flags = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
        if doc_flags:
            df['TOTAL_DOCUMENTS_SUBMITTED'] = df[doc_flags].sum(axis=1)
        
        return df

    def create_inquiry_features(self, df):
        """Create credit bureau inquiry features."""
        logger.info("Creating inquiry features")
        
        inquiry_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
        
        available_cols = [col for col in inquiry_cols if col in df.columns]
        if available_cols:
            df['TOTAL_INQUIRIES'] = df[available_cols].sum(axis=1)
            
            recent_cols = ['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 
                          'AMT_REQ_CREDIT_BUREAU_MON']
            available_recent = [col for col in recent_cols if col in df.columns]
            if available_recent:
                df['RECENT_INQUIRY_RATIO'] = df[available_recent].sum(axis=1) / (df['TOTAL_INQUIRIES'] + 1)
        
        return df

    def create_social_features(self, df):
        """Create social circle features."""
        logger.info("Creating social features")
        
        df['TOTAL_SOCIAL_DEFAULTS'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] + df['DEF_60_CNT_SOCIAL_CIRCLE']
        df['TOTAL_SOCIAL_OBS'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] + df['OBS_60_CNT_SOCIAL_CIRCLE']
        
        return df

    def apply_binary_encoding(self, df):
        """Apply binary encoding to specific categorical columns."""
        logger.info('Applying binary encoding')
        
        binary_mappings = {
            'CODE_GENDER': 'M',
            'FLAG_OWN_CAR': 'Y',
            'FLAG_OWN_REALTY': 'Y',
            'NAME_CONTRACT_TYPE': 'Cash loans'
        }
        
        for col, true_val in binary_mappings.items():
            if col in df.columns:
                df[col] = (df[col] == true_val).astype(int)
        
        logger.info('Binary encoding completed successfully')
        return df

    def apply_frequency_encoding(self, df):
        """Apply frequency encoding to high-cardinality categorical columns."""
        logger.info('Applying frequency encoding')
        
        freq_cols = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
        encoding_maps = {}
        
        for col in freq_cols:
            if col in df.columns:
                freq_map = df[col].value_counts().to_dict()
                encoding_maps[col] = freq_map
                df[f'{col}_FREQ'] = df[col].map(freq_map)
        
        # Save frequency encoding maps
        output_dir = os.path.dirname(self.config.output_path) if hasattr(self.config, 'output_path') else 'artifacts/feature_engineering'
        artifact_dir = self.config.artifact_dir if hasattr(self.config, 'artifact_dir') else output_dir
        os.makedirs(artifact_dir, exist_ok=True)
        
        joblib.dump(
            encoding_maps,
            os.path.join(artifact_dir, 'frequency_encoding_maps.pkl')
        )
        logger.info('Frequency encoding maps saved successfully')
        
        return df

    def apply_onehot_encoding(self, df):
        """Apply one-hot encoding to low-cardinality categorical columns."""
        logger.info('Applying one-hot encoding')
        
        ohe_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
        available_cat = [col for col in ohe_cols if col in df.columns]
        
        if not available_cat:
            logger.info('One-hot encoding completed successfully')
            return df
        
        # One-hot encode using drop_first=True and integer dtype to align with notebook
        df_encoded = pd.get_dummies(df, columns=available_cat, drop_first=True, dtype=int)
        
        # Sanitize column names for LightGBM/XGBoost compatibility
        import re
        clean_columns = {}
        for col in df_encoded.columns:
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
            clean_col = re.sub(r'_+', '_', clean_col)
            clean_col = clean_col.rstrip('_')
            clean_columns[col] = clean_col
        
        df_encoded.rename(columns=clean_columns, inplace=True)
        logger.info('Column names sanitized successfully')
        logger.info('One-hot encoding completed successfully')
        
        return df_encoded

    def handle_missing_values(self, df):
        """Handle missing values as done in notebooks."""
        logger.info("Handling missing values")
        
        # Fill specific columns with 0 as per notebook
        zero_fill_cols = ['DAYS_EMPLOYED', 'EMPLOYMENT_YEARS', 'EMPLOYMENT_AGE_RATIO',
                          'TOTAL_SOCIAL_DEFAULTS', 'TOTAL_SOCIAL_OBS', 'EXT_SOURCE_STD',
                          'OCCUPATION_TYPE_FREQ']
        
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill median for specific ratio columns
        median_cols = ['CNT_FAM_MEMBERS', 'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO',
                       'INCOME_PER_PERSON', 'CHILDREN_RATIO', 'ACTIVE_LOAN_RATIO',
                       'CLOSED_LOAN_RATIO', 'OVERDUE_PER_LOAN', 'PROLONGED_LOAN_RATIO',
                       'CREDIT_UTILIZATION_RATIO']
        
        for col in median_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill remaining numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df

    def initiate_feature_engineering(self, application_path: str, bureau_path: str):
        logger.info("Starting comprehensive feature engineering")
        try:
            # Load raw data
            app_df = pd.read_csv(application_path)
            bureau_df = pd.read_csv(bureau_path)
            logger.info(f"Loaded application: {app_df.shape}, bureau: {bureau_df.shape}")
            
            # Target Column Validation
            if self.config.target_column not in app_df.columns:
                raise ValueError(
                    f'Target column '
                    f'{self.config.target_column} '
                    f'not found'
                )

            # Step 1: Clean application data
            app_df = self.clean_application_data(app_df)
            
            # Step 1b: Drop high missing and low importance columns as per notebook
            notebook_drops = [
                # From Section 2.3 & 2.4 of 02_data_cleaning.ipynb
                'COMMONAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI',
                'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_MEDI',
                'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI',
                'FLOORSMIN_AVG', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI',
                'YEARS_BUILD_AVG', 'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI',
                'FONDKAPREMONT_MODE',
                
                # From Section 2.4 (additional drops) of 02_data_cleaning.ipynb
                'LANDAREA_AVG', 'LANDAREA_MODE', 'LANDAREA_MEDI',
                'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE', 'BASEMENTAREA_MEDI',
                'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_MEDI',
                'ELEVATORS_AVG', 'ELEVATORS_MODE', 'ELEVATORS_MEDI',
                'APARTMENTS_AVG', 'APARTMENTS_MODE', 'APARTMENTS_MEDI',
                'ENTRANCES_AVG', 'ENTRANCES_MODE', 'ENTRANCES_MEDI',
                'LIVINGAREA_AVG', 'LIVINGAREA_MODE', 'LIVINGAREA_MEDI',
                'FLOORSMAX_AVG', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI',
                'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI',
                'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE',
                
                # Other low importance/missing drops
                'TOTALAREA_MODE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 
                'WALLSMATERIAL_MODE'
            ]
            app_df.drop(columns=[col for col in notebook_drops if col in app_df.columns], inplace=True, errors='ignore')
            logger.info("Dropped high-missing and low-importance columns as per notebook")
            
            # Step 2: Aggregate bureau data
            bureau_agg = self.aggregate_bureau_data(bureau_df)
            
            # Step 3: Merge bureau aggregates
            app_df = app_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
            
            # Bureau Merge Validation
            missing_bureau = app_df['SK_ID_BUREAU_count'].isnull().sum()
            logger.info(f'Missing bureau records: {missing_bureau}')

            # Duplicate Row Handling
            duplicate_count = app_df.duplicated().sum()
            logger.info(f'Duplicate rows found: {duplicate_count}')
            if duplicate_count > 0:
                app_df = app_df.drop_duplicates()
                logger.info('Duplicate rows removed successfully')

            # Step 4: Create all engineered features
            app_df = self.create_financial_ratios(app_df)
            app_df = self.create_family_features(app_df)
            app_df = self.create_external_credit_features(app_df)
            app_df = self.create_collateral_feature(app_df)
            app_df = self.create_bureau_features(app_df)
            app_df = self.create_age_employment_features(app_df)
            app_df = self.create_document_features(app_df)
            app_df = self.create_inquiry_features(app_df)
            app_df = self.create_social_features(app_df)
            
            # Step 5: Binary Encoding
            app_df = self.apply_binary_encoding(app_df)
            
            # Step 6: Frequency Encoding
            app_df = self.apply_frequency_encoding(app_df)
            
            # Drop original columns after frequency encoding to prevent OHE
            freq_drop_cols = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
            app_df.drop(columns=[col for col in freq_drop_cols if col in app_df.columns], inplace=True, errors='ignore')
            logger.info("Dropped original high-cardinality categorical columns after frequency encoding")
            
            # Step 7: Handle missing values before OHE
            app_df = self.handle_missing_values(app_df)
            
            # Step 8: One-Hot Encoding
            app_df = self.apply_onehot_encoding(app_df)
            
            # ============================================================
            # REMOVE EXTRA RAW BUREAU FEATURES
            # ============================================================

            raw_bureau_cols = [
                'bureau_active_loans',
                'bureau_closed_loans',
                'bureau_total_loans',
                'bureau_total_overdue',
                'bureau_total_prolonged',
                'bureau_total_credit',
                'bureau_total_debt'
            ]

            app_df.drop(
                columns=[
                    col for col in raw_bureau_cols
                    if col in app_df.columns
                ],
                inplace=True,
                errors='ignore'
            )

            logger.info(
                'Dropped raw bureau aggregate columns successfully'
            )

            # ============================================================
            # KEEP IMPORTANT PREDICTIVE FEATURES
            # ============================================================

            drop_high_corr = [
                'FLAG_EMP_PHONE',
                'NAME_INCOME_TYPE_Pensioner'
            ]

            app_df.drop(
                columns=[
                    col for col in drop_high_corr
                    if col in app_df.columns
                ],
                inplace=True,
                errors='ignore'
            )

            logger.info(
                'Dropped highly correlated columns successfully'
            )
            
            # Remove ID Column Before Modeling
            if 'SK_ID_CURR' in app_df.columns:
                app_df.drop(columns=['SK_ID_CURR'], inplace=True)
                logger.info('Identifier columns removed successfully')

            # Final Missing Value Check
            remaining_nulls = app_df.isnull().sum().sum()
            logger.info(f'Remaining null values: {remaining_nulls}')

            logger.info(f"Final engineered dataset shape: {app_df.shape}")
            
            # Save final engineered dataset
            output_dir = os.path.dirname(self.config.output_path) if hasattr(self.config, 'output_path') else 'artifacts/feature_engineering'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save Feature Names
            feature_names = list(app_df.columns)
            if self.config.target_column in feature_names:
                feature_names.remove(self.config.target_column)

            feature_path = os.path.join(output_dir, 'feature_names.json')
            with open(feature_path, 'w') as f:
                json.dump(feature_names, f)
            logger.info('Feature names saved successfully')

            output_path = os.path.join(output_dir, "final_feature_engineered_dataset.csv")
            app_df.to_csv(output_path, index=False)
            logger.info(f"Saved engineered dataset to {output_path}")
            
            logger.info(
                f'Total engineered features: '
                f'{app_df.shape[1]}'
            )

            return output_path
            
        except Exception as e:
            raise LoanRecoveryException(e, sys)
