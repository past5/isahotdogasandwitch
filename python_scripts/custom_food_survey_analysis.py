#!/usr/bin/env python3
"""
Custom Food Survey Analysis to categorize respondents based on specific food classification questions.
"""

import os
import sys
import json
import pandas as pd
import re
from pathlib import Path

# Add the parent directory to the path to import protos if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def custom_preprocess_survey_data(file_path):
    """
    Preprocess the survey data focusing on Q1, Q2, Q3, Q4, Q7, Q9_text, Q10.
    Classify respondents into specified categories based on their answers.
    """
    ext = Path(file_path).suffix.lower()
    
    # Load the data based on file extension
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Target questions for analysis
    target_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q9_text', 'Q10']
    
    # Create contribution tracking for each respondent
    contributions = []
    for _ in range(len(df)):
        contributions.append({q: 0 for q in target_questions})
    
    # Process Q9_text to extract yes/no
    if 'Q9_text' in df.columns:
        def extract_yes_no(text):
            if not isinstance(text, str):
                return 'Unknown'
            
            text_lower = text.lower()
            # Check for clear yes patterns
            if re.search(r'\byes\b|\bagree\b|\bdefinitely\b|\bcertainly\b|\bsure\b|\bof course\b', text_lower):
                return 'Yes'
            # Check for clear no patterns
            elif re.search(r'\bno\b|\bdisagree\b|\bdefinitely not\b|\bcertainly not\b|\bnot a\b|\bnever\b', text_lower):
                return 'No'
            # Default to unknown
            return 'Unknown'
        
        df['Q9_yes_no'] = df['Q9_text'].apply(extract_yes_no)
    
    # Initialize classification metrics
    sandwich_score = pd.Series(0, index=df.index)
    chef_intent_score = pd.Series(0, index=df.index)
    
    # Process Q1: Is a hot dog a sandwich?
    if 'Q1' in df.columns:
        q1_values = df['Q1'].map({
            '✅ Yes': 1,
            '❌ No': -1,
            '🤔 It depends': 0.5,
            '🚫 I refuse to answer': 0
        }).fillna(-999)
        
        for i, val in enumerate(q1_values):
            contributions[i]['Q1'] = val
            
        sandwich_score += q1_values
    
    # Process Q2: If hot dog bun rips...
    if 'Q2' in df.columns:
        q2_values = df['Q2'].map({
            '✅ Yes': 0.8,
            '❌ No': -0.8
        }).fillna(-999)
        
        for i, val in enumerate(q2_values):
            contributions[i]['Q2'] = val
            
        sandwich_score += q2_values
    
    # Process Q3: Minimum number of ingredients
    if 'Q3' in df.columns:
        q3_values = df['Q3'].map({
            '0 🥖 (bread is a sandwich by itself)': 1,
            '1 🧈 (e.g. buttered toast)': 0.5,
            '2 🥜🍓 (e.g. PB&J)': 0,
            '3 or more 🥪': -1
        }).fillna(-999)
        
        for i, val in enumerate(q3_values):
            contributions[i]['Q3'] = val
            
        sandwich_score += q3_values
    
    # Process Q4: Taco shell type
    if 'Q4' in df.columns:
        q4_values = df['Q4'].map({
            'Hard': 0.8,  # More sandwich-like with hard shell
            'Soft': 0.8,  # Less sandwich-like with soft shell
            '🚫 Under no conditions should a taco be considered a sandwich': -1,
            'Only if the bottom cracks or rips, regardless of shell': 0.4
        }).fillna(-999)
        
        for i, val in enumerate(q4_values):
            contributions[i]['Q4'] = val
            
        sandwich_score += q4_values
    
    # Process Q7: Open-faced sandwich
    if 'Q7' in df.columns:
        q7_values = df['Q7'].map({
            'Much more ⬆️': -1,  # Traditionalists would prefer standard sandwiches
            'A little more': -0.5,
            'It makes no difference': 0,
            'A little less': 0.5,
            'Much less ⬇️': 1  # Strong traditionalists would avoid open-faced
        }).fillna(-999)
        
        for i, val in enumerate(q7_values):
            contributions[i]['Q7'] = val
            
        sandwich_score += q7_values
    
    # Process Q9: Pizza sandwich
    if 'Q9_yes_no' in df.columns:
        q9_values = df['Q9_yes_no'].map({
            'Yes': 1,  # Expansionist view
            'No': -1,  # Traditionalist view
            'Unknown': 0
        })
        
        for i, val in enumerate(q9_values):
            contributions[i]['Q9_text'] = val
            
        sandwich_score += q9_values
    
    # Process Q10: Chef intent - Using key-value mapping
    if 'Q10' in df.columns:
        q10_values = df['Q10'].map({
            '0 - Not at all 🚫': 0.0,
            1: 0.2,
            2: 0.4,
            3: 0.6,
            4: 0.8,
            '5 - It\'s the only thing that matters 🎯': 1.0
        }).fillna(-999)  # Default to middle value if not found
        
        chef_intent_score = q10_values
        
        for i, val in enumerate(chef_intent_score):
            contributions[i]['Q10'] = val
    
    # Classification functions
    def classify_sandwich_perspective(score):
        if score >= 1:
            return 'Sandwich Expansionist'
        elif score <= -1:
            return 'Sandwich Traditionalist'
        else:
            return 'Sandwich Moderate'
    
    def classify_intent_perspective(score):
        if score >= 0.7:
            return 'Creator-Intent Prioritizer'
        elif score <= 0.3:
            return 'Form-Over-Intent Prioritizer'
        else:
            return 'Intent Moderate'
    
    # Apply classifications
    df['sandwich_classification'] = sandwich_score.apply(classify_sandwich_perspective)
    df['intent_classification'] = chef_intent_score.apply(classify_intent_perspective)
    
    # Add contribution details to the dataframe
    df['contribution_details'] = [json.dumps(cont) for cont in contributions]
    
    # Save processed file
    output_file = Path(file_path).with_name('custom_analysis_' + Path(file_path).name)
    df.to_csv(output_file, index=False)
    
    # Prepare results for JSON output
    results = []
    for idx, row in df.iterrows():
        respondent_data = {
            'respondent_id': idx,
            'sandwich_classification': row['sandwich_classification'],
            'intent_classification': row['intent_classification'],
            'question_contributions': json.loads(row['contribution_details']),
            'explanation': {
                'sandwich_score': float(sandwich_score[idx]),
                'chef_intent_score': float(chef_intent_score[idx]),
                'question_weights': {
                    'Q1': 'Hot dog sandwich question (1 point if yes, -1 if no, 0.5 if depends)',
                    'Q2': 'Broken bun question (0.8 points if yes, -0.8 if no)',
                    'Q3': 'Minimum ingredients (1 point if 0, 0.5 if 1, 0 if 2, -1 if 3+)',
                    'Q4': 'Taco question (0.4 point if bottom rips, -1 if neither, 0.8/0.8 for hard/soft)',
                    'Q7': 'Open-faced preference (-1 if much more, 1 if much less)',
                    'Q9_text': 'Pizza sandwich (1 point if yes, -1 if no)',
                    'Q10': 'Chef intent importance (0=Not at all, 5=Only thing that matters, normalized to 0-1 scale)'
                }
            }
        }
        results.append(respondent_data)
    
    # Analysis summary
    analysis_summary = {
        'file_path': str(output_file),
        'total_respondents': len(df),
        'classification_counts': {
            'sandwich_classifications': df['sandwich_classification'].value_counts().to_dict(),
            'intent_classifications': df['intent_classification'].value_counts().to_dict()
        },
        'question_influence': {
            q: {
                'avg_contribution': sum(c[q] for c in contributions) / len(contributions),
                'max_contribution': max(c[q] for c in contributions),
                'min_contribution': min(c[q] for c in contributions)
            } for q in target_questions
        },
        'respondents': results
    }
    
    # Print summary
    print("\n===== CUSTOM FOOD CLASSIFICATION ANALYSIS =====")
    print(f"Data saved to: {output_file}")
    print("\nCLASSIFICATION COUNTS:")
    print("  Sandwich Classifications:")
    for cls, count in analysis_summary['classification_counts']['sandwich_classifications'].items():
        print(f"    - {cls}: {count}")
    
    print("  Intent Classifications:")
    for cls, count in analysis_summary['classification_counts']['intent_classifications'].items():
        print(f"    - {cls}: {count}")
    
    print("\nQUESTION INFLUENCE:")
    for q, stats in analysis_summary['question_influence'].items():
        print(f"  - {q}: avg={stats['avg_contribution']:.2f}, range=[{stats['min_contribution']:.2f}, {stats['max_contribution']:.2f}]")
    
    return analysis_summary

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Custom Food Classification Analysis")
    parser.add_argument("--file", required=True, help="Path to the survey data file (Excel, CSV, JSON)")
    parser.add_argument("--output", help="Output file to save detailed results (JSON format)")
    
    args = parser.parse_args()
    
    # Process the food classification survey
    results = custom_preprocess_survey_data(args.file)
    
    # Save detailed results if requested
    if args.output:
        # Convert NumPy types before JSON serialization
        results = convert_numpy_types(results)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
