#!/usr/bin/env python
# coding: utf-8
"""
Generate a synthetic dataset with intentional data drift for testing retraining pipeline.
This script creates a modified version of the original dataset with significant drift.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def introduce_drift(df, drift_type='moderate'):
    """
    Introduce data drift to the dataset.
    
    Types of drift introduced:
    1. Numerical drift: Shift means and increase variance
    2. Categorical drift: Change distribution of categories
    3. Pattern drift: Change relationships between features
    """
    df_drift = df.copy()
    
    print(f"\n{'='*60}")
    print(f"Introducing {drift_type} data drift...")
    print(f"{'='*60}\n")
    
    # Define drift intensity based on type
    drift_multipliers = {
        'mild': 1.1,
        'moderate': 1.3,
        'severe': 1.6
    }
    multiplier = drift_multipliers.get(drift_type, 1.3)
    
    # ==================== NUMERICAL DRIFT ====================
    
    # 1. Tenure - simulate aging customer base (increase mean)
    if 'Tenure' in df_drift.columns:
        original_mean = df_drift['Tenure'].mean()
        df_drift['Tenure'] = df_drift['Tenure'] * multiplier + np.random.normal(5, 2, len(df_drift))
        df_drift['Tenure'] = df_drift['Tenure'].clip(0, 61)  # Keep within reasonable bounds
        new_mean = df_drift['Tenure'].mean()
        print(f"✓ Tenure drift: {original_mean:.2f} → {new_mean:.2f} (shift: {new_mean-original_mean:.2f})")
    
    # 2. WarehouseToHome - simulate expansion to remote areas (increase mean + variance)
    if 'WarehouseToHome' in df_drift.columns:
        original_mean = df_drift['WarehouseToHome'].mean()
        df_drift['WarehouseToHome'] = df_drift['WarehouseToHome'] * multiplier + np.random.normal(10, 5, len(df_drift))
        df_drift['WarehouseToHome'] = df_drift['WarehouseToHome'].clip(5, 127)
        new_mean = df_drift['WarehouseToHome'].mean()
        print(f"✓ WarehouseToHome drift: {original_mean:.2f} → {new_mean:.2f} (shift: {new_mean-original_mean:.2f})")
    
    # 3. HourSpendOnApp - simulate increased engagement (increase mean)
    if 'HourSpendOnApp' in df_drift.columns:
        original_mean = df_drift['HourSpendOnApp'].mean()
        df_drift['HourSpendOnApp'] = df_drift['HourSpendOnApp'] * multiplier + np.random.normal(1, 0.5, len(df_drift))
        df_drift['HourSpendOnApp'] = df_drift['HourSpendOnApp'].clip(0, 5)
        new_mean = df_drift['HourSpendOnApp'].mean()
        print(f"✓ HourSpendOnApp drift: {original_mean:.2f} → {new_mean:.2f} (shift: {new_mean-original_mean:.2f})")
    
    # 4. OrderAmountHikeFromlastYear - simulate inflation (increase variance)
    if 'OrderAmountHikeFromlastYear' in df_drift.columns:
        original_mean = df_drift['OrderAmountHikeFromlastYear'].mean()
        df_drift['OrderAmountHikeFromlastYear'] = df_drift['OrderAmountHikeFromlastYear'] * multiplier + np.random.normal(5, 3, len(df_drift))
        df_drift['OrderAmountHikeFromlastYear'] = df_drift['OrderAmountHikeFromlastYear'].clip(11, 26)
        new_mean = df_drift['OrderAmountHikeFromlastYear'].mean()
        print(f"✓ OrderAmountHikeFromlastYear drift: {original_mean:.2f} → {new_mean:.2f} (shift: {new_mean-original_mean:.2f})")
    
    # 5. CashbackAmount - simulate changed cashback policy (shift distribution)
    if 'CashbackAmount' in df_drift.columns:
        original_mean = df_drift['CashbackAmount'].mean()
        df_drift['CashbackAmount'] = df_drift['CashbackAmount'] * multiplier + np.random.normal(50, 20, len(df_drift))
        df_drift['CashbackAmount'] = df_drift['CashbackAmount'].clip(0, 324)
        new_mean = df_drift['CashbackAmount'].mean()
        print(f"✓ CashbackAmount drift: {original_mean:.2f} → {new_mean:.2f} (shift: {new_mean-original_mean:.2f})")
    
    # 6. DaySinceLastOrder - simulate reduced frequency (increase mean)
    if 'DaySinceLastOrder' in df_drift.columns:
        original_mean = df_drift['DaySinceLastOrder'].mean()
        df_drift['DaySinceLastOrder'] = df_drift['DaySinceLastOrder'] * multiplier + np.random.normal(3, 1.5, len(df_drift))
        df_drift['DaySinceLastOrder'] = df_drift['DaySinceLastOrder'].clip(0, 46)
        new_mean = df_drift['DaySinceLastOrder'].mean()
        print(f"✓ DaySinceLastOrder drift: {original_mean:.2f} → {new_mean:.2f} (shift: {new_mean-original_mean:.2f})")
    
    # ==================== CATEGORICAL DRIFT ====================
    
    # 7. PreferredLoginDevice - simulate mobile adoption
    if 'PreferredLoginDevice' in df_drift.columns:
        # Increase mobile phone usage
        mask = df_drift['PreferredLoginDevice'] == 'Computer'
        switch_to_mobile = np.random.choice([True, False], size=mask.sum(), p=[0.4, 0.6])
        df_drift.loc[mask, 'PreferredLoginDevice'] = np.where(
            switch_to_mobile, 
            'Mobile Phone', 
            df_drift.loc[mask, 'PreferredLoginDevice']
        )
        mobile_pct = (df_drift['PreferredLoginDevice'] == 'Mobile Phone').sum() / len(df_drift) * 100
        print(f"✓ PreferredLoginDevice drift: Mobile Phone now {mobile_pct:.1f}% (increased mobile adoption)")
    
    # 8. PreferredPaymentMode - simulate payment preference changes
    if 'PreferredPaymentMode' in df_drift.columns:
        # Shift towards digital payments
        mask = df_drift['PreferredPaymentMode'] == 'Cash on Delivery'
        num_cod = mask.sum()
        if num_cod > 0:
            # Generate switch decisions
            switch_to_digital = np.random.choice([True, False], size=num_cod, p=[0.5, 0.5])
            # Generate new payment modes for those who switch
            new_modes = np.random.choice(['Credit Card', 'Debit Card', 'UPI'], size=num_cod)
            # Apply changes
            current_values = df_drift.loc[mask, 'PreferredPaymentMode'].values
            df_drift.loc[mask, 'PreferredPaymentMode'] = np.where(switch_to_digital, new_modes, current_values)
        cod_pct = (df_drift['PreferredPaymentMode'] == 'Cash on Delivery').sum() / len(df_drift) * 100
        print(f"✓ PreferredPaymentMode drift: COD now {cod_pct:.1f}% (shift to digital payments)")
    
    # 9. PreferedOrderCat - simulate category popularity changes
    if 'PreferedOrderCat' in df_drift.columns:
        # Increase laptop & accessory orders (simulate tech boom)
        mask = df_drift['PreferedOrderCat'].isin(['Fashion', 'Grocery'])
        num_fashion_grocery = mask.sum()
        if num_fashion_grocery > 0:
            switch_to_tech = np.random.choice([True, False], size=num_fashion_grocery, p=[0.35, 0.65])
            current_values = df_drift.loc[mask, 'PreferedOrderCat'].values
            df_drift.loc[mask, 'PreferedOrderCat'] = np.where(switch_to_tech, 'Laptop & Accessory', current_values)
        laptop_pct = (df_drift['PreferedOrderCat'] == 'Laptop & Accessory').sum() / len(df_drift) * 100
        print(f"✓ PreferedOrderCat drift: Laptop & Accessory now {laptop_pct:.1f}% (tech boom)")
    
    # 10. SatisfactionScore - simulate declining satisfaction
    if 'SatisfactionScore' in df_drift.columns:
        original_mean = df_drift['SatisfactionScore'].mean()
        # Shift distribution towards lower scores
        df_drift['SatisfactionScore'] = df_drift['SatisfactionScore'] - np.random.choice([0, 1], size=len(df_drift), p=[0.6, 0.4])
        df_drift['SatisfactionScore'] = df_drift['SatisfactionScore'].clip(1, 5)
        new_mean = df_drift['SatisfactionScore'].mean()
        print(f"✓ SatisfactionScore drift: {original_mean:.2f} → {new_mean:.2f} (declining satisfaction)")
    
    # 11. Complain - simulate increased complaints
    if 'Complain' in df_drift.columns:
        original_rate = df_drift['Complain'].mean()
        # Increase complaint rate
        mask = df_drift['Complain'] == 0
        num_no_complaints = mask.sum()
        if num_no_complaints > 0:
            new_complaints = np.random.choice([1, 0], size=num_no_complaints, p=[0.15, 0.85])
            df_drift.loc[mask, 'Complain'] = new_complaints
        new_rate = df_drift['Complain'].mean()
        print(f"✓ Complain drift: {original_rate:.3f} → {new_rate:.3f} (complaint rate: {new_rate*100:.1f}%)")
    
    # ==================== PATTERN DRIFT ====================
    
    # 12. Create interaction effects - simulate changing customer behavior patterns
    # Customers with high tenure but low satisfaction (unusual pattern)
    if 'Tenure' in df_drift.columns and 'SatisfactionScore' in df_drift.columns:
        high_tenure_mask = df_drift['Tenure'] > df_drift['Tenure'].quantile(0.75)
        df_drift.loc[high_tenure_mask, 'SatisfactionScore'] = np.maximum(
            1,
            df_drift.loc[high_tenure_mask, 'SatisfactionScore'] - np.random.choice([0, 1, 2], size=high_tenure_mask.sum(), p=[0.5, 0.3, 0.2])
        )
        print(f"✓ Pattern drift: Introduced high-tenure + low-satisfaction pattern")
    
    print(f"\n{'='*60}")
    print(f"Drift introduction complete!")
    print(f"{'='*60}\n")
    
    return df_drift

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset with intentional data drift'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='E_Commerce_Dataset.xlsx',
        help='Path to original dataset (default: E_Commerce_Dataset.xlsx)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for drifted dataset (default: E_Commerce_Dataset_Drifted_<timestamp>.xlsx)'
    )
    parser.add_argument(
        '--sheet',
        type=str,
        default='E Comm',
        help='Sheet name in Excel file (default: E Comm)'
    )
    parser.add_argument(
        '--drift-type',
        type=str,
        choices=['mild', 'moderate', 'severe'],
        default='moderate',
        help='Intensity of drift to introduce (default: moderate)'
    )
    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=1.0,
        help='Fraction of original data to use (default: 1.0 = all data)'
    )
    
    args = parser.parse_args()
    
    # Generate output filename with timestamp if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'E_Commerce_Dataset_Drifted_{timestamp}.xlsx'
    
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DRIFT DATASET GENERATOR")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Drift type: {args.drift_type}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"{'='*60}\n")
    
    # Load original dataset
    print("Loading original dataset...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Sample if requested
    if args.sample_fraction < 1.0:
        n_samples = int(len(df) * args.sample_fraction)
        df = df.sample(n=n_samples, random_state=42)
        print(f"✓ Sampled {len(df)} rows ({args.sample_fraction*100:.1f}%)")
    
    # Display original statistics
    print(f"\nOriginal dataset statistics:")
    if 'Churn' in df.columns:
        churn_rate = df['Churn'].mean()
        print(f"  Churn rate: {churn_rate*100:.2f}%")
    print(f"  Total customers: {len(df)}")
    
    # Introduce drift
    df_drifted = introduce_drift(df, drift_type=args.drift_type)
    
    # Display drifted statistics
    print(f"\nDrifted dataset statistics:")
    if 'Churn' in df_drifted.columns:
        churn_rate_drifted = df_drifted['Churn'].mean()
        print(f"  Churn rate: {churn_rate_drifted*100:.2f}%")
    print(f"  Total customers: {len(df_drifted)}")
    
    # Save drifted dataset
    print(f"\nSaving drifted dataset to: {args.output}")
    df_drifted.to_excel(args.output, sheet_name=args.sheet, index=False)
    print(f"✓ Saved successfully!")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ DRIFT DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nYou can now use this drifted dataset to test retraining:")
    print(f"  python preprocess.py --dataset {args.output}")
    print(f"  python training_MLFlow.py --dataset {args.output}")
    print(f"\nThe model should detect drift and trigger retraining!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()