#!/usr/bin/env python3
"""
Demographic Data Extraction Script.
Extracts survey demographic data with calculated percentages.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Define demographic categories we want to extract
DEMOGRAPHIC_CATEGORIES = [
    'GENDER', 
    'REGION', 
    'URBANRURAL', 
    'EDUCATION',
    'HHINCOME',
    'ETHNICITYROLL23',
    'VIZMINROLL23',
    'PMARITALSTATUS'
]

# Define survey questions to analyze
SURVEY_QUESTIONS = {
    "Q1": "Is a hot dog a sandwich?",
    "Q2": "If the bottom of a hot dog bun rips, resulting in two separate pieces of bread containing the sausage, would it now be considered a sandwich?"
}

def calculate_percentages(counts):
    """
    Calculate percentages from count data.
    
    Args:
        counts: List of counts
        
    Returns:
        List of percentages (rounded to 1 decimal place)
    """
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    
    return [(count / total * 100) for count in counts]

def process_survey_data(data_file):
    """
    Process a survey data file to extract demographic data.
    
    Args:
        data_file: Path to the data file (CSV, Excel, or JSON)
        
    Returns:
        Dictionary with demographic data results
    """
    print(f"Processing survey data: {data_file}")
    
    # Load the data
    file_ext = Path(data_file).suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(data_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(data_file)
    elif file_ext == '.json':
        df = pd.read_json(data_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Initialize the results dictionary
    demographic_data = {}
    
    # Process each question
    for question_id in SURVEY_QUESTIONS:
        # Find columns related to this question
        question_cols = [col for col in df.columns if question_id.lower() in col.lower()]
        
        if not question_cols:
            print(f"Warning: No columns found for question {question_id}")
            continue
        
        # Find the primary response column for this question
        response_col = None
        for col in question_cols:
            # Skip columns with too many unique responses or text context columns
            if "_with_context" in col or df[col].nunique() > 10:
                continue
            response_col = col
            break
        
        if not response_col:
            print(f"Warning: No suitable response column found for question {question_id}")
            continue
        
        # Initialize question data
        demographic_data[question_id] = {}
        
        # Process overall responses
        all_responses = df[response_col].value_counts().to_dict()
        sorted_responses = sorted(all_responses.items(), key=lambda x: x[1], reverse=True)
        
        labels = []
        data = []
        
        for response, count in sorted_responses:
            resp_str = str(response)
            data.append(count)
            
            # Calculate percentage for the label
            percentage = (count / df[response_col].count()) * 100
            labels.append(f"{resp_str} ({percentage:.1f}%)")
        
        demographic_data[question_id]['ALL'] = {
            'labels': labels,
            'data': data
        }
        
        # Process each demographic category
        for category in DEMOGRAPHIC_CATEGORIES:
            # Check if the category column exists
            if category not in df.columns:
                print(f"Warning: Category {category} not found in the dataset")
                continue
            
            demographic_data[question_id][category] = {}
            
            # Get unique values for this category
            unique_values = df[category].dropna().unique()
            
            for value in unique_values:
                # Filter data for this demographic value
                subset = df[df[category] == value]
                
                # Get response counts
                value_responses = subset[response_col].value_counts().to_dict()
                sorted_value_responses = sorted(value_responses.items(), key=lambda x: x[1], reverse=True)
                
                value_labels = []
                value_data = []
                
                for response, count in sorted_value_responses:
                    value_labels.append(str(response))
                    value_data.append(count)
                
                # Store in the result dictionary
                demographic_data[question_id][category][str(value)] = {
                    'labels': value_labels,
                    'data': value_data
                }
    
    return demographic_data

def demographic_survey_analysis(data_file, output_file):
    """
    Analyze survey data and save demographic results to a JSON file.
    
    Args:
        data_file: Path to the survey data file
        output_file: Path to the output JSON file
        
    Returns:
        Dictionary with analysis results
    """
    # Process the data
    demographic_results = process_survey_data(data_file)
    
    # Save to output file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(demographic_results, f, indent=2)
        print(f"Demographic results saved to {output_file}")
    
    # Print summary
    print("\n===== DEMOGRAPHIC SURVEY ANALYSIS =====")
    print(f"Processed file: {data_file}")
    print(f"Questions analyzed: {len(demographic_results)}")
    
    # Print summary of categories
    for question_id, question_data in demographic_results.items():
        print(f"\nQuestion {question_id}: {SURVEY_QUESTIONS.get(question_id, 'Unknown')}")
        print(f"  Total response count: {sum(question_data['ALL']['data'])}")
        print(f"  Response distribution: {', '.join(question_data['ALL']['labels'])}")
        print(f"  Demographic breakdowns: {len(question_data) - 1} categories")
    
    return demographic_results

def main():
    parser = argparse.ArgumentParser(description="Demographic Survey Data Extraction")
    parser.add_argument("--file", required=True, help="Path to the survey data file (CSV, Excel, JSON)")
    parser.add_argument("--output", default="demographic_data.json", 
                      help="Path to the output JSON file (default: demographic_data.json)")
    
    args = parser.parse_args()
    
    # Run the analysis
    demographic_survey_analysis(args.file, args.output)

if __name__ == "__main__":
    main()
