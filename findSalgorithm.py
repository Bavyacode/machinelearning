# Training data: (example, label)
# Each example is a tuple of attribute values
"""training_data = [
    [['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],'Yes'],
    [['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'],
    [['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'],
    [['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes']
]

def find_s(training_data):
    # Initialize the most specific hypothesis to the first positive example
    hypothesis = None

    for example, label in training_data:
        if label == 'Yes':  # Consider only positive examples
            if hypothesis is None:
                hypothesis = example  # Start with the first positive example
            else:
                # Update hypothesis to be the least general generalization
                for i in range(len(hypothesis)):
                    if hypothesis[i] != example[i]:
                        hypothesis[i] = '?'  # Generalize the attribute
    
    return hypothesis

# Find the most specific hypothesis
hypothesis = find_s(training_data)
print(f"The most specific hypothesis is: {hypothesis}")"""

import pandas as pd

# Define the training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

# Create the DataFrame
df = pd.DataFrame(data)
print(df)

def find_s(df):
    # Initialize the most specific hypothesis to the first positive example
    hypothesis = None

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        if row['EnjoySport'] == 'Yes':  # Consider only positive examples
            if hypothesis is None:
                hypothesis = list(row[:-1])  # Start with the first positive example (excluding the label)                                                                                                   
            else:
                # Update hypothesis to be the least general generalization
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row[i]:
                        hypothesis[i] = '?'  # Generalize the attribute
    
    return hypothesis

# Find the most specific hypothesis
hypothesis = find_s(df)
print(f"The most specific hypothesis is: {hypothesis}")

