import pandas as pd

def more_general(h1, h2):
    """
    Function to check if hypothesis h1 is more general than h2.
    """
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg)
    return all(more_general_parts)

def fulfills(example, hypothesis):
    """
    Check if a hypothesis is consistent with the example.
    """
    return more_general(hypothesis, example)

def min_generalization(h, x):
    """
    Function to generalize the hypothesis h to make it consistent with example x.
    """
    new_h = list(h)
    for i in range(len(h)):
        if not fulfills(x[i:i+1], h[i:i+1]):
            new_h[i] = '?' if h[i] != "0" else x[i]
    return tuple(new_h)

def min_specialization(h, domains, x):
    """
    Function to specialize the hypothesis h to make it consistent with example x.
    """
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new = h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new = h[:i] + ("0",) + h[i+1:]
            results.append(h_new)
    return results

def candidate_elimination(examples):
    """
    Candidate Elimination algorithm implementation.
    """
    domains = [set(examples.iloc[:, col]) for col in range(len(examples.columns) - 1)]
    G = {("?",) * (len(examples.columns) - 1)}
    S = {("0",) * (len(examples.columns) - 1)}
    
    for index, example in examples.iterrows():
        x, y = tuple(example.iloc[:-1]), example.iloc[-1]
        if y == 'Yes':  # Positive example
            G = {g for g in G if fulfills(x, g)}
            S = {s for s in S if not fulfills(x, s)}
            S = {min_generalization(s, x) for s in S}
        else:  # Negative example
            S = {s for s in S if not fulfills(x, s)}
            G = {g for g in G if fulfills(x, g)}
            G = G.union(set(h for g in G for h in min_specialization(g, domains, x)))
            G = {g for g in G if any(more_general(g, s) for s in S)}
    
    return S, G

# Load your data into a pandas DataFrame
# Replace 'your_data.csv' with the path to your CSV file
df = pd.read_csv('training_data.csv')

# Run the Candidate Elimination algorithm
S, G = candidate_elimination(df)
print("Specific Boundary: ", S)
print("General Boundary: ", G)
