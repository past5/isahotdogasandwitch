# Is a Hot Dog a Sandwich?

This repository contains data, analysis, and visualization for a food classification study focused on the age-old debate: Is a hot dog a sandwich? The project includes data analysis of a comprehensive food survey examining how people categorize various food items, with a special focus on controversial classifications like hot dogs, tacos, and cereal with milk.

## Repository Structure

### Python Scripts
- **custom_food_classification.py**: Analyzes survey responses about food classification, generates visualizations of controversial food categories, and creates Venn diagrams of food classification overlaps.
- **custom_food_survey_analysis.py**: Categorizes survey respondents based on their food classification perspectives (Sandwich Expansionists vs. Traditionalists) and their views on creator intent.
- **demographic_data_extraction.py**: Processes demographic information from survey respondents.

### JSON Data
- **food_survey_results.json**: Primary survey data with responses from 501 participants across multiple food classification questions.
- **demographic_data.json**: Demographic information about survey respondents.
- **custom_food_survey_analysis.json**: Processed survey data with classifications of respondents.
- **custom_food_classification.json**: Analysis of how different food items are categorized.
- **nlu_food_results.json** & **focused_nlu_results.json**: Natural language understanding analysis of text responses.
- **question_to_explanations.json**: Mapping of questions to explanations and classifications.

### Hackathon Entry
This repository was created for a data visualization hackathon, as evidenced by the files in `hackathon_rules/` and `hackathon_entry/`.

- **Visual Assets**:
  - hotdog.png, sandwich.png, hotdoghistory.png, hotdogvictory.png, hero.png
  - hotdogvssandwich.mp4: Video presentation of findings
  - insights.html: Interactive web visualization of survey results

### Hackathon Rules
The repository includes the original hackathon datasets and guidelines emphasizing:
- Creative innovation in presentation
- Authentic storytelling based on real data
- Community collaboration
- Integration of AI into projects

## Key Research Questions
The survey explores several contentious food categorization questions:
1. Is a hot dog a sandwich?
2. Does a ripped hot dog bun change its sandwich classification?
3. Is cereal with milk a type of soup?
4. Is curry a type of soup?
5. Would two pieces of pizza put together crust-sides out be considered a sandwich?

## Analysis Approach
The project uses both quantitative analysis of survey responses and natural language processing of free-text explanations to understand how people categorize food items. The analysis classifies respondents on two spectrums:
- **Sandwich Classification**: From Traditionalists to Expansionists
- **Creator Intent**: From Form-Over-Intent to Creator-Intent Prioritizers

## Getting Started
To run the analysis scripts:
```bash
python python_scripts/custom_food_classification.py
python python_scripts/custom_food_survey_analysis.py
```

## License
This project is licensed under the license included in the LICENSE file.
