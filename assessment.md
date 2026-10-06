Machine Learning Engineer
Assessment Instructions:
● Objective
• Use data/train_test.csv as the labeled development data.
• Decide how to split and validate the development data.
• Explore, clean, engineer features, validate, and model the data as you see fit.
• Run your trained model on every load in data/validation.csv.
■ Data/validation.csv contains the 12,000 loads requiring final predictions. Each
row contains the load features and a unique load_id.
■ Run your trained model on every row in data/validation.csv.
■ data/validation_predictions_template.csv contains the same 12,000 load_id
values.
■ Fill the template’s predicted_rate column and save the completed file as
validation_predictions.csv.
• Please refer to the README.md for setup instructions.
● Submit
instructions.
• An accessible GitHub repository containing your solution code, dependencies, and run
• validation_predictions.csv containing exactly:
■ load_id,predicted_rate
• A PDF or DOCX report containing:
■ Your approach to validate test/train data and split.
■ The fixed December prediction chart produced by the provided score.py
• A 2-3 minute loom going over the following:
■ Key findings from exploring the data
■ Any data-quality issues identified and how you addressed them
■ Reasoning behind the chosen model
■ Training and validation approach, including how the data was split
■ A brief walkthrough of the most important parts of your code