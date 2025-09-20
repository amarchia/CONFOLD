import argparse
import sys

# Add the project root to the Python path to allow imports from other folders
sys.path.append('.')

from datasets import * # Import all dataset functions
from foldrm import Classifier
from utils import split_data

def main(dataset_name, rules_file=None):
    """
    Runs a CON-FOLD experiment on a specified dataset, optionally with expert rules.
    """
    # --- 1. Load Data ---
    try:
        # This dynamically calls the dataset function (e.g., birds(), mushroom())
        dataset_func = globals()[dataset_name]
        model_template, data = dataset_func()
        print(f"Successfully loaded the '{dataset_name}' dataset.")
    except KeyError:
        print(f"Error: Dataset function '{dataset_name}' not found in datasets.py.")
        return

    data_train, data_test = split_data(data, ratio=0.8, shuffle=True)
    X_test = [d[:-1] for d in data_test]
    Y_test = [d[-1] for d in data_test]
    labels = list(set(Y_test))

    # --- 2. Instantiate Classifier ---
    model = Classifier(attrs=model_template.attrs, 
                       numeric=model_template.numeric, 
                       label=model_template.label)

    # --- 3. Add Manual Rules (if provided) ---
    if rules_file:
        print(f"Loading expert rules from: {rules_file}")
        with open(rules_file, 'r') as f:
            for line in f:
                rule_string = line.strip()
                if rule_string and not rule_string.startswith('#'): # Ignore empty lines and comments
                    print(f"  -> Adding rule: '{rule_string}'")
                    model.add_manual_rule(rule_string, model.attrs, model.numeric, labels, instructions=False)
    else:
        print("No expert rules file provided. Training a baseline model.")

    # --- 4. Fit and Predict ---
    model.fit(data_train)
    predictions_tuples = model.predict(X_test)
    predicted_labels = [p[0] for p in predictions_tuples]

    # --- 5. Evaluate and Print Results ---
    accuracy = sum(1 for i in range(len(Y_test)) if predicted_labels[i] == Y_test[i]) / len(Y_test)
    
    print("\n--- Final Ruleset ---")
    model.print_asp(simple=True)
    
    print("\n--- Evaluation ---")
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a CON-FOLD experiment.")
    parser.add_argument("dataset", type=str, help="The name of the dataset function to run (e.g., 'birds', 'mushroom').")
    parser.add_argument("--rules", type=str, default=None, help="Optional path to a .txt file containing expert rules, one per line.")
    
    args = parser.parse_args()
    
    main(args.dataset, args.rules)