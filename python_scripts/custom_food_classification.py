#!/usr/bin/env python3
"""
Custom Food Classification Survey Analysis Script.
Focuses on food classification and visualizations of controversial food categories.
"""

import os
import sys
import json
import argparse
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import re

try:
    from matplotlib_venn import venn2, venn3
    VENN_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib_venn not available. Install with 'pip install matplotlib-venn'")
    VENN_AVAILABLE = False

# Default food classification categories
DEFAULT_FOOD_CATEGORIES = {
    "sandwich_candidates": [
        "sandwich", "burger", "hot dog", "wrap", "taco", "sub", 
        "panini", "toast", "open-faced sandwich", "pizza sandwich"
    ],
    "soup_candidates": [
        "soup", "stew", "chowder", "broth", "curry", "cereal", 
        "cereal with milk", "gazpacho", "bisque"
    ],
    "pizza_types": [
        "pizza", "calzone", "flatbread", "pizza roll", "pizza bagel"
    ],
    "dessert_items": [
        "cake", "pie", "cookie", "brownie", "ice cream", "pudding", 
        "custard", "candy", "chocolate"
    ]
}

# Controversial food items that commonly appear in multiple categories
DEFAULT_CONTROVERSIAL_ITEMS = [
    "hot dog", "taco", "cereal with milk", "curry", "open-faced sandwich", 
    "pizza sandwich", "ice cream sandwich", "stew", "calzone"
]

# Survey questions for vote closeness measurement
SURVEY_QUESTIONS = {
    "Q1": "Is a hot dog a sandwich?",
    "Q2": "If the bottom of a hot dog bun rips, resulting in two separate pieces of bread containing the sausage, would it now be considered a sandwich?",
    "Q6": "Is cereal with milk a type of soup?",
    "Q8": "Would you classify a curry as a type of soup, like stew or chowder? Why or why not?",
    "Q9": "If two pieces of pizza are put together crust-sides out, is that a sandwich? Why or why not?"
}

# Question to food mapping - which foods are relevant to which questions
QUESTION_FOOD_MAPPING = {
    "Q1": ["hot dog", "sandwich"],
    "Q2": ["hot dog", "sandwich", "bun"],
    "Q6": ["cereal", "milk", "cereal with milk", "soup"],
    "Q8": ["curry", "soup", "stew", "chowder"],
    "Q9": ["pizza", "sandwich", "pizza sandwich"]
}

def extract_food_mentions(text, food_terms):
    """
    Extract food mentions from a piece of text.
    
    Args:
        text: The text to search
        food_terms: List of food terms to look for
        
    Returns:
        List of found food terms
    """
    text_lower = text.lower()
    found_terms = []
    
    for term in food_terms:
        if term.lower() in text_lower:
            found_terms.append(term)
    
    return found_terms

def calculate_vote_closeness(df):
    """
    Calculate how close the votes are for specific survey questions.
    Questions with nearly equal yes/no responses are considered more controversial.
    
    Args:
        df: DataFrame containing survey responses
        
    Returns:
        Dictionary mapping question IDs to vote closeness scores (0-1)
    """
    target_questions = list(SURVEY_QUESTIONS.keys())
    vote_closeness = {}
    
    for question_id in target_questions:
        # Try to find columns related to this question
        question_cols = [col for col in df.columns if question_id.lower() in col.lower()]
        
        if not question_cols:
            continue
            
        for col in question_cols:
            # Skip columns that are unlikely to contain yes/no responses
            if "_with_context" in col or df[col].nunique() > 10:
                continue
                
            values = df[col].value_counts(normalize=True)
            
            # Skip columns with too many unique values (not likely yes/no)
            if len(values) < 2 or len(values) > 5:
                continue
                
            # For binary responses
            if len(values) == 2:
                # Calculate how close votes are to 50/50 split
                values_list = values.tolist()
                closeness = 1.0 - abs(values_list[0] - values_list[1])
                vote_closeness[question_id] = closeness
            
            # For questions with multiple options but still representing opposing views
            elif len(values) <= 5:
                # Group similar responses (e.g., "Yes" and "Yes, definitely")
                yes_pattern = r'yes|true|definitely|agree|1|sandwich|soup'
                no_pattern = r'no|false|disagree|never|0|not'
                
                yes_count = sum(values[i] for i, val in enumerate(values.index) 
                              if re.search(yes_pattern, str(val).lower()))
                no_count = sum(values[i] for i, val in enumerate(values.index) 
                             if re.search(no_pattern, str(val).lower()))
                
                # Calculate closeness if we successfully grouped responses
                if yes_count > 0 and no_count > 0:
                    total = yes_count + no_count
                    closeness = 1.0 - abs(yes_count/total - no_count/total)
                    vote_closeness[question_id] = closeness
            
            # If we found valid vote data, no need to check other columns for this question
            if question_id in vote_closeness:
                break
    
    return vote_closeness

def is_food_related_to_question(food, question_id):
    """
    Determine if a food item is related to a specific survey question.
    
    Args:
        food: Food item name
        question_id: Question identifier (e.g., 'Q1')
        
    Returns:
        Boolean indicating if the food is related to the question
    """
    if question_id not in SURVEY_QUESTIONS:
        return False
        
    question_text = SURVEY_QUESTIONS[question_id].lower()
    food_lower = food.lower()
    
    # Check if food is mentioned in the question
    if food_lower in question_text:
        return True
        
    # Check if food is in the question's known relevant foods
    if question_id in QUESTION_FOOD_MAPPING and food_lower in QUESTION_FOOD_MAPPING[question_id]:
        return True
    
    return False

def process_food_survey_data(data_file, custom_categories=None):
    """
    Process a food survey data file to identify food classifications.
    
    Args:
        data_file: Path to the data file (CSV, Excel, or JSON)
        custom_categories: Custom food categories (optional)
        
    Returns:
        Dictionary with food classification results
    """
    print(f"Processing food survey data: {data_file}")
    
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
    
    # Use either custom categories or defaults
    food_categories = custom_categories or DEFAULT_FOOD_CATEGORIES
    
    # Get all food terms to look for
    all_food_terms = set()
    for category_terms in food_categories.values():
        all_food_terms.update(category_terms)
    
    # Find text columns (potential food mentions)
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Extract food mentions
    food_mentions = defaultdict(int)
    
    for column in text_columns:
        for text in df[column].dropna().astype(str):
            found_terms = extract_food_mentions(text, all_food_terms)
            for term in found_terms:
                food_mentions[term] += 1
    
    # Organize into food families
    food_families = {}
    for category, terms in food_categories.items():
        food_families[category] = {}
        for term in terms:
            if term in food_mentions:
                food_families[category][term] = food_mentions[term]
    
    # Identify foods that appear in multiple categories (controversial items)
    food_to_categories = defaultdict(list)
    for category, foods in food_families.items():
        for food in foods:
            food_to_categories[food].append(category)
    
    # Calculate vote closeness for target questions
    vote_closeness_data = calculate_vote_closeness(df)
    target_questions = list(SURVEY_QUESTIONS.keys())
    
    # Build controversial items dictionary
    controversial_items = {}
    total_mentions = sum(food_mentions.values())
    for food, categories in food_to_categories.items():
        if len(categories) > 1:
            # Calculate controversy score using:
            # 1. Number of categories the food appears in (normalized)
            # 2. Food's mention frequency relative to all mentions
            # 3. Category size diversity (foods in more diverse-sized categories are more controversial)
            
            # Base component from category count - foods in more categories are more controversial
            category_factor = min(1.0, (len(categories) - 1) / (len(food_categories) - 1)) if len(food_categories) > 1 else 0.5
            
            # Mention frequency component - more frequently mentioned foods have more statistical significance
            mention_factor = min(1.0, food_mentions[food] / (total_mentions * 0.1))
            
            # Category size diversity - foods spanning categories of different sizes are more controversial
            category_sizes = [len(food_families[cat]) for cat in categories]
            size_diversity = np.std(category_sizes) / max(np.mean(category_sizes), 1) if len(category_sizes) > 1 else 0
            
            # Check if food is related to questions with close votes
            vote_closeness_factor = 0
            related_questions = []
            
            for question_id in target_questions:
                if question_id in vote_closeness_data and is_food_related_to_question(food, question_id):
                    related_questions.append(question_id)
                    vote_closeness_factor = max(vote_closeness_factor, vote_closeness_data[question_id])
            
            # Score components
            score_components = {
                'category_factor': category_factor,
                'mention_factor': mention_factor,
                'size_diversity': min(1.0, size_diversity)
            }

            # Include vote closeness if it exists and is relevant to this food item
            if vote_closeness_factor > 0:
                # Adjust controversy score: 50% standard factors, 50% vote closeness
                controversy_score = 0.5 * ((0.5 * category_factor) + (0.3 * mention_factor) + (0.2 * min(1.0, size_diversity))) + 0.5 * vote_closeness_factor
                score_components['vote_closeness_factor'] = vote_closeness_factor
                score_components['related_questions'] = related_questions
            else:
                # Standard controversy score without vote closeness
                controversy_score = (0.5 * category_factor) + (0.3 * mention_factor) + (0.2 * min(1.0, size_diversity))
            
            controversy_score = min(1.0, controversy_score)
            
            controversial_items[food] = {
                'categories': categories,
                'count': food_mentions[food],
                'controversy_score': controversy_score,
                'score_components': score_components
            }
    
    # Calculate category overlaps
    category_relationships = {}
    categories = list(food_families.keys())
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j:  # Only process each pair once
                cat1_foods = set(food_families[cat1].keys())
                cat2_foods = set(food_families[cat2].keys())
                overlap = cat1_foods.intersection(cat2_foods)
                
                if overlap:
                    key = f"{cat1}_{cat2}"
                    category_relationships[key] = {
                        'categories': [cat1, cat2],
                        'foods': list(overlap),
                        'count': len(overlap)
                    }
    
    # Compile results
    results = {
        'food_families': food_families,
        'food_controversies': controversial_items,
        'category_relationships': category_relationships,
        'statistics': {
            'total_mentions': sum(food_mentions.values()),
            'unique_terms': len(food_mentions),
            'category_counts': {category: len(foods) for category, foods in food_families.items()}
        }
    }
    
    # Include vote closeness data if available
    if vote_closeness_data:
        results['vote_closeness'] = vote_closeness_data
    
    return results

def generate_venn_diagram(food_results, output_path=None):
    """
    Generate a Venn diagram visualization of overlapping food categories.
    
    Args:
        food_results: Dictionary with food classification results
        output_path: Path to save the visualization image (optional)
        
    Returns:
        Base64 encoded image if successful, None otherwise
    """
    if not VENN_AVAILABLE:
        return None
        
    try:
        # Extract data
        food_families = food_results.get('food_families', {})
        controversial_items = food_results.get('food_controversies', {})
        
        if not food_families or not any(food_families.values()):
            return None
            
        # Prepare sets for Venn diagram - use top 3 categories
        category_foods = {}
        for category, foods in food_families.items():
            if foods:
                category_foods[category] = set(foods.keys())
                
        # Select up to 3 largest categories for visualization
        top_categories = sorted(
            [(category, len(foods)) for category, foods in category_foods.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        top_category_names = [cat for cat, _ in top_categories]
        
        if len(top_category_names) < 2:
            print("Not enough food categories for Venn diagram")
            return None
            
        # Create the figure
        plt.figure(figsize=(10, 8))
        
        # Generate sets for the diagram
        sets = [category_foods[cat] for cat in top_category_names]
        
        # Create Venn diagram based on number of sets
        if len(sets) == 2:
            venn = venn2(sets, set_labels=top_category_names)
            # Highlight controversial items in the intersection
            if venn and venn.get_patch_by_id('11'):
                venn.get_patch_by_id('11').set_color('red')
                venn.get_patch_by_id('11').set_alpha(0.5)
                
                # Get the foods in the intersection
                intersection_foods = sets[0].intersection(sets[1])
                if intersection_foods:
                    # Add a note about the controversial items
                    items_text = ", ".join(list(intersection_foods)[:3])
                    if len(intersection_foods) > 3:
                        items_text += f" and {len(intersection_foods) - 3} more"
                    plt.figtext(0.5, 0.01, f"Controversial items: {items_text}", 
                               ha='center', fontsize=9)
        else:  # 3 sets
            venn = venn3(sets, set_labels=top_category_names)
            # Highlight the central intersection as the controversial zone
            if venn and venn.get_patch_by_id('111'):
                venn.get_patch_by_id('111').set_color('red')
                venn.get_patch_by_id('111').set_alpha(0.5)
                
                # Get the foods in the central intersection
                central_intersection = sets[0].intersection(sets[1]).intersection(sets[2])
                if central_intersection:
                    # Add a note about the controversial items
                    items_text = ", ".join(list(central_intersection)[:3])
                    if len(central_intersection) > 3:
                        items_text += f" and {len(central_intersection) - 3} more"
                    plt.figtext(0.5, 0.01, f"Highly controversial items: {items_text}", 
                               ha='center', fontsize=9)
        
        plt.title('Food Category Classifications with Controversy Zones')
        
        # Add a legend for controversial items
        plt.figtext(0.15, 0.05, "Red zones = Items with controversial classification", 
                   color='red', fontsize=10, weight='bold')
        
        # Save image if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        # Save as base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating Venn diagram: {str(e)}")
        return None

def generate_controversy_chart(food_results, output_path=None):
    """
    Generate a visualization of controversial food items.
    
    Args:
        food_results: Dictionary with food classification results
        output_path: Path to save the visualization image (optional)
        
    Returns:
        Base64 encoded image if successful, None otherwise
    """
    try:
        # Extract data
        controversial_items = food_results.get('food_controversies', {})
        
        if not controversial_items:
            return None
            
        # Sort items by controversy score
        sorted_items = sorted(
            [(food, data['controversy_score'], len(data['categories'])) 
             for food, data in controversial_items.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10 most controversial
        
        foods = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        categories = [item[2] for item in sorted_items]
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(foods, scores, color='orangered', alpha=0.7)
        
        # Add category count as text on bars
        for i, (bar, cat_count) in enumerate(zip(bars, categories)):
            plt.text(
                bar.get_width() + 0.02, 
                bar.get_y() + bar.get_height()/2, 
                f"In {cat_count} categories", 
                va='center'
            )
        
        plt.xlabel('Controversy Score')
        plt.title('Most Controversial Food Classifications')
        plt.xlim(0, max(scores) + 0.2)
        plt.tight_layout()
        
        # Save image if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        # Save as base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating controversy chart: {str(e)}")
        return None

def generate_controversy_matrix(food_results, output_path=None):
    """
    Generate a matrix visualization showing how foods are classified across categories.
    
    Args:
        food_results: Dictionary with food classification results
        output_path: Path to save the visualization image (optional)
        
    Returns:
        Base64 encoded image if successful, None otherwise
    """
    try:
        # Extract data
        controversial_items = food_results.get('food_controversies', {})
        
        if not controversial_items or len(controversial_items) < 2:
            return None
        
        # Select top controversial items (by score)
        top_items = dict(sorted(
            controversial_items.items(),
            key=lambda x: x[1]['controversy_score'],
            reverse=True
        )[:8])  # Limit to 8 for readability
        
        # Get all unique categories
        all_categories = set()
        for item_data in top_items.values():
            all_categories.update(item_data['categories'])
        
        all_categories = sorted(list(all_categories))
        foods = list(top_items.keys())
        
        # Create the matrix data (1 if food belongs to category, 0 otherwise)
        matrix = np.zeros((len(foods), len(all_categories)))
        
        for i, food in enumerate(foods):
            food_categories = top_items[food]['categories']
            for j, category in enumerate(all_categories):
                if category in food_categories:
                    matrix[i, j] = 1
        
        # Create the figure
        plt.figure(figsize=(10, 8))
        
        # Plot as a heatmap
        plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Add labels
        plt.yticks(range(len(foods)), foods)
        plt.xticks(range(len(all_categories)), all_categories, rotation=45, ha='right')
        
        # Add grid lines
        plt.grid(False)
        for i in range(len(foods) - 1):
            plt.axhline(i + 0.5, color='gray', linestyle='-', linewidth=0.5)
        for i in range(len(all_categories) - 1):
            plt.axvline(i + 0.5, color='gray', linestyle='-', linewidth=0.5)
        
        # Add a color bar
        plt.colorbar(ticks=[0, 1], label='Belongs to Category')
        
        plt.title('Food Category Membership Matrix')
        plt.tight_layout()
        
        # Save image if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        # Save as base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating controversy matrix: {str(e)}")
        return None

def custom_food_survey_analysis(data_file, output_file, custom_categories=None):
    """
    Analyze food survey data and save results to a single JSON file.
    
    Args:
        data_file: Path to the survey data file
        output_file: Path to the output JSON file
        custom_categories: Custom food categories (optional)
        
    Returns:
        Dictionary with analysis results
    """
    # Process the data
    food_results = process_food_survey_data(data_file, custom_categories)
    
    # Generate all visualizations but don't include them in JSON
    
    # Venn diagram
    venn_image = generate_venn_diagram(food_results)
    if venn_image and output_file:
        venn_path = Path(output_file).with_suffix('.venn.png')
        with open(venn_path, 'wb') as f:
            f.write(base64.b64decode(venn_image))
        print(f"Venn diagram saved to {venn_path}")
    
    # Controversy chart
    controversy_image = generate_controversy_chart(food_results)
    if controversy_image and output_file:
        controversy_path = Path(output_file).with_suffix('.controversy.png')
        with open(controversy_path, 'wb') as f:
            f.write(base64.b64decode(controversy_image))
        print(f"Controversy chart saved to {controversy_path}")
    
    # Controversy matrix
    matrix_image = generate_controversy_matrix(food_results)
    if matrix_image and output_file:
        matrix_path = Path(output_file).with_suffix('.matrix.png')
        with open(matrix_path, 'wb') as f:
            f.write(base64.b64decode(matrix_image))
        print(f"Controversy matrix saved to {matrix_path}")
    
    # Combine results without including visualizations
    complete_results = {
        'food_classification': {
            'food_families': food_results['food_families'],
            'food_controversies': food_results['food_controversies'],
            'category_relationships': food_results['category_relationships'],
            'statistics': food_results['statistics']
        }
    }
    
    # Save to output file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"Analysis results saved to {output_file}")
    
    # Print summary
    print("\n===== FOOD CLASSIFICATION SURVEY ANALYSIS =====")
    print(f"Processed file: {data_file}")
    print(f"Total food mentions: {food_results['statistics']['total_mentions']}")
    print(f"Unique food terms: {food_results['statistics']['unique_terms']}")
    
    print("\n----- FOOD CATEGORIES -----")
    for category, foods in food_results['food_families'].items():
        if foods:
            print(f"  {category}: {len(foods)} foods")
    
    print("\n----- CONTROVERSIAL FOODS -----")
    controversial_count = len(food_results['food_controversies'])
    print(f"  Found {controversial_count} controversial food items")
    
    # List top controversial foods
    sorted_controversies = sorted(
        food_results['food_controversies'].items(),
        key=lambda x: x[1]['controversy_score'],
        reverse=True
    )[:5]  # Show top 5
    
    for food, data in sorted_controversies:
        print(f"  - {food}: In categories {', '.join(data['categories'])} (score: {data['controversy_score']:.2f})")
    
    return complete_results

def load_custom_categories_from_file(file_path):
    """
    Load custom food categories from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary of custom categories
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Custom Food Classification Survey Analysis")
    parser.add_argument("--file", required=True, help="Path to the survey data file (CSV, Excel, JSON)")
    parser.add_argument("--output", default="food_survey_analysis.json", 
                      help="Path to the output JSON file (default: food_survey_analysis.json)")
    parser.add_argument("--categories-file", help="JSON file with custom food categories")
    
    args = parser.parse_args()
    
    # Load custom categories if provided
    custom_categories = None
    if args.categories_file:
        try:
            custom_categories = load_custom_categories_from_file(args.categories_file)
            print(f"Loaded custom categories from {args.categories_file}")
        except Exception as e:
            print(f"Error loading custom categories: {str(e)}")
            print("Using default categories instead")
    
    # Define food survey context for better classification
    food_survey_context = {
        "primary_controversies": [
            "hot dog as sandwich", 
            "cereal as soup", 
            "curry as soup", 
            "pizza as sandwich base"
        ]
    }
    
    # Enhance default categories with specific controversial items
    if not custom_categories:
        custom_categories = DEFAULT_FOOD_CATEGORIES
        custom_categories["controversial_items"] = DEFAULT_CONTROVERSIAL_ITEMS
    
    # Run the analysis
    custom_food_survey_analysis(args.file, args.output, custom_categories)

if __name__ == "__main__":
    main()