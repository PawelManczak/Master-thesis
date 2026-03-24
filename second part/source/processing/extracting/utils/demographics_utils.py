#!/usr/bin/env python3
"""
Tools for handling demographic data (age, gender).
"""
import pandas as pd

# Standard age categories across the project
AGE_BINS = [20, 25, 30, 35, 40, 100]
AGE_LABELS = ['20-24', '25-29', '30-34', '35-39', '40+']

def get_age_group(age) -> str:
    """
    Categorizes age into standard age groups for analysis.
    Returns a string with the group name.
    """
    if pd.isna(age):
        return 'Unknown'
    
    try:
        age = int(float(age))
    except (ValueError, TypeError):
        return 'Unknown'
        
    if age < 20: 
        return '19-19'
        
    for i in range(len(AGE_BINS)-1):
        if AGE_BINS[i] <= age < AGE_BINS[i+1]:
            return AGE_LABELS[i]
            
    return 'Unknown'
