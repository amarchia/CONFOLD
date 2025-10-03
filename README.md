# CON-FOLD: An Explainable Learning Algorithm with Confidence and Domain Knowledge
CON-FOLD is a Python implementation of an explainable machine learning algorithm that extends the original FOLD-RM algorithm. It generates human-readable logic rules for classification tasks and introduces two key enhancements:

1.  **Confidence Scores:** Every prediction is accompanied by a confidence value, allowing users to gauge the model's certainty.
2.  **Domain Knowledge Injection:** You can easily inject expert knowledge into the model using simple, human-readable rule strings. The algorithm then learns around this expert knowledge, combining human intuition with machine learning.

This repository is based on the original FOLD-RM work by Huaduo Wang, available [here](https://github.com/hwd404/FOLD-RM).

## Key Features

*   **Explainable by Design:** The model's output is a set of logic rules that are easy for humans to understand and verify.
*   **Confidence-Aware:** Provides confidence scores for its predictions using the Wilson score interval.
*   **Expert-in-the-Loop:** Seamlessly integrate expert knowledge to improve model accuracy and interpretability.
*   **Handles Mixed Data:** Works directly with both numerical and categorical data without requiring one-hot encoding.

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/lachlanmcg123/CONFOLD.git
    cd CONFOLD
    ```

2.  Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```    

## Tutorial: A Complete Walkthrough

For a complete, step-by-step guide on how to use CON-FOLD, please see our detailed Jupyter Notebook tutorial. It provides a hands-on example that covers:
1.  Training a baseline model from scratch.
2.  Injecting expert domain knowledge (with and without pre-defined confidence).
3.  Pruning the model to improve simplicity and prevent overfitting.
4.  Comparing the results to see the benefits.

**[View the Birds of Prey Tutorial](./examples/1_birds_of_prey_tutorial.ipynb)**

## Quick Start: Adding Expert Rules

The core feature of CON-FOLD is adding domain knowledge. You can provide rules with a pre-defined confidence, or let the model learn the confidence for you.

### Option 1: Provide a Rule with a Confidence Score

If you are confident in your rule, you can specify its confidence directly.

```python
# Define and add your expert rule with a specific confidence
expert_rule = "with confidence 0.95 class = 'yes' if 'beak' '==' 'Sharp' except if 'wingspan' '<=' '25'"
model.add_manual_rule(expert_rule, model.attrs, model.numeric, labels=['yes', 'no'])

# The model will use this confidence value directly
model.fit(data_train)
```

### Option 2: Let CON-FOLD Learn the Confidence
If you know a rule is correct but are unsure of its exact confidence, simply leave it out. CON-FOLD will calculate it from the training data when you call .fit().
```python
# 1. Define a rule WITHOUT a confidence value
rule_without_confidence = "class = 'no' if 'beak' '==' 'Curved'"

# 2. Add the rule. It will be assigned a default confidence of 0.5 for now.
model.add_manual_rule(rule_without_confidence, model.attrs, model.numeric, labels=['yes', 'no'])

# 3. Fit the model. 
# CON-FOLD will now calculate the true confidence of your rule from the data.
model.fit(data_train)

# 4. Print the rules to see the newly learned confidence score.
model.print_asp(simple=True)
```

## Advanced Usage: Pruning and Rule Files

CON-FOLD includes methods for pruning the learned ruleset to create simpler, more robust models. This helps to prevent overfitting and can improve the interpretability of the final model.

### Pruning Learned Rules

There are two primary methods for pruning:

**1. Post-Hoc Confidence Pruning (Confidence Threshold Pruning)**

This is the simplest method. After training a model with `.fit()`, you can remove any rules that fall below a certain confidence threshold. This is useful for cleaning up a model by discarding low-confidence, potentially noisy rules.

```python
from algo import prune_rules

# Assume 'model' has been trained with model.fit(data_train)
print("Original number of rules:", len(model.rules))

# Prune rules with confidence less than 0.75
model.rules = prune_rules(model.rules, confidence=0.75)

print("Number of rules after pruning:", len(model.rules))
```

**2. Confidence-Driven Learning with `confidence_fit` (Improvement Threshold Pruning)**

A more advanced and integrated method is to use `.confidence_fit()`. This method only adds exceptions to rules *during the training process* if they improve the rule's confidence by a specified `improvement_threshold`. This prevents the model from learning overly specific exceptions in the first place, often leading to simpler and more general models from the outset.

```python
# Train the model using the confidence-driven method
# Only add exceptions if they improve confidence by at least 2% (0.02)
model.confidence_fit(data_train, improvement_threshold=0.02)

# The resulting model.rules will likely be simpler and more robust
model.print_asp(simple=True)
```

### Loading Rules from a File
For automated experiments, you can load rules from a text file. Our example script examples/run_experiment.py demonstrates this

1. Create a rule file (e.g., my_rules.txt)
The file should contain one rule per line. Lines starting with # are treated as comments and ignored.

```python
# Rules for the mushroom dataset. Lines starting with # are ignored.
with confidence 0.99 class = 'p' if 'odor' '==' 'f'
```

2. Run the experiment script
The script allows you to specify the dataset and an optional rules file from the command line.


```python
# Run on the mushroom dataset with your expert rules
python examples/run_experiment.py mushroom --rules examples/my_rules.txt
```

## Tutorial Walkthrough and Output

The following is a summary of the code and output from the main tutorial, `1_birds_of_prey_tutorial.ipynb`.

### Step 1: Load and Prepare the Data
We load a custom dataset of 20 birds and split it into a 15-bird training set and a 5-bird test set.

```python
# Load the data
model_template, data = birds(data_path='../data/birds/birds.csv')

# Split into training and testing sets
data_train = data[:15]
data_test = data[15:]
```
```code
% birds dataset loaded (20, 3)
Training set size: 15 birds
Testing set size: 5 birds
```

### Step 2: The Baseline Model
We train a standard model to see what it can learn on its own.

```python
# Instantiate and fit the baseline model
baseline_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)
baseline_model.fit(data_train, ratio=0.5)
baseline_model.print_asp(simple=True)
```
```code
--- Rules Learned by the Baseline Model ---
predator(X,'yes') :- not beak(X,'curved'), not ab1(X). [confidence: 0.73529]
predator(X,'no') :- wingspan(X,N0), N0>10.0. [confidence: 0.7]
predator(X,'no') :- wingspan(X,N0), N0<=10.0. [confidence: 0.55]
ab1(X) :- wingspan(X,N0), N0<=20.0.
```
The baseline model achieves 80% accuracy, incorrectly classifying one of the test birds.

### Step 3 & 4: Injecting an Expert Rule
We provide a single, nuanced rule to the model before training it again.

```python
# Instantiate a new model and add our expert rule
expert_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)
expert_rule = "with confidence 0.95 class = 'yes' if 'beak' '==' 'Sharp' except if 'wingspan' '<=' '25'"
expert_model.add_manual_rule(expert_rule, model_template.attrs, model_template.numeric, ['yes', 'no'], instructions=False)

# Fit the model on the same data
expert_model.fit(data_train, ratio=0.75)
expert_model.print_asp(simple=True)
```
```code
--- Final Ruleset from the Expert Model ---
predator(X,'yes') :- beak(X,'sharp'), not ab1(X). [confidence: 0.95]
predator(X,'no') :- wingspan(X,N0), N0>10.0. [confidence: 0.7]
predator(X,'no') :- wingspan(X,N0), N0<=10.0. [confidence: 0.55]
ab1(X) :- wingspan(X,N0), N0<=25.
```

By providing the difficult rule, the expert-guided model achieves 100% accuracy.

### Step 5: Learning Rule Confidence
We show that you can provide rules without confidence scores and let CON-FOLD calculate them from the data.

```python
# Add rules without confidence
rule1 = "class = 'no' if 'beak' '==' 'Curved'"
rule2 = "class = 'yes' if 'beak' '==' 'Sharp' except if 'wingspan' '<=' '25'"
# ... add rules to a new model ...

# Fit the model
learned_confidence_model.fit(data_train, ratio=0.5)
learned_confidence_model.print_asp(simple=True)
```
```code
--- Final Ruleset with Learned Confidence ---
The confidence values have now been updated based on the training data!
predator(X,'no') :- beak(X,'curved'). [confidence: 0.65385]
predator(X,'yes') :- beak(X,'sharp'), not ab1(X). [confidence: 0.73529]
predator(X,'no') :- wingspan(X,N0), N0>10.0. [confidence: 0.59091]
predator(X,'no') :- wingspan(X,N0), N0<=10.0. [confidence: 0.55]
ab1(X) :- wingspan(X,N0), N0<=25.
```
This model also achieves 100% accuracy.

### Step 6: Adding Expert Rules and Learning Their Confidence

A common scenario is that an expert knows a rule is generally true, but doesn't know the exact statistics or confidence.

CON-FOLD is designed for this. We can provide rules without a confidence value, and the algorithm will **calculate the confidence for us** based on the training data.

Let's provide two rules our ornithologist is fairly certain about:
1.  **"Birds with a curved beak are not predators."**
2.  **"A bird with a sharp beak is a predator, UNLESS it is very small (wingspan <= 25cm)."**

```python
# Instantiate a new classifier
learned_confidence_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)

# Define our expert rules as strings, but WITHOUT the 'with confidence' part.
rule1_no_confidence = "class = 'no' if 'beak' '==' 'Curved'"
rule2_no_confidence = "class = 'yes' if 'beak' '==' 'Sharp' except if 'wingspan' '<=' '25'"

# Add the manual rules to the model
learned_confidence_model.add_manual_rule(rule1_no_confidence, model_template.attrs, model_template.numeric, ['yes', 'no'], instructions=False)
learned_confidence_model.add_manual_rule(rule2_no_confidence, model_template.attrs, model_template.numeric, ['yes', 'no'], instructions=False)

print("--- Manual Rules Added (Before Training) ---")
print("Notice the default confidence value of 0.5 assigned to each rule.")
for rule in learned_confidence_model.rules:
    print(rule)
```
```code
--- Manual Rules Added (Before Training) ---
Notice the default confidence value of 0.5 assigned to each rule.
((-1, '==', 'no'), [(1, '==', 'Curved')], [], 0.5)
((-1, '==', 'yes'), [(1, '==', 'Sharp')], [(-1, [(0, '<=', 25)], [], 0)], 0.5)
```

### Learned Confidence from Data
As we saw above, the rules were added with a placeholder confidence of `0.5`. Now, when we call `.fit()`, CON-FOLD will evaluate these rules against the training data and replace the placeholder with a properly calculated confidence score.

```python
# Now, fit the model on the training data.
# The algorithm will calculate the confidence of our provided rules and then learn any additional rules needed.
learned_confidence_model.fit(data_train, ratio=0.5)

# Print the final, combined rule set
print("--- Final Ruleset with Learned Confidence ---")
print("The confidence values have now been updated based on the training data!")
learned_confidence_model.print_asp(simple=True)
#Note that confidence values will be relatively low due to the small size of the training data. 
```
```code
--- Final Ruleset with Learned Confidence ---
The confidence values have now been updated based on the training data!
predator(X,'no') :- beak(X,'curved'). [confidence: 0.65385]
predator(X,'yes') :- beak(X,'sharp'), not ab1(X). [confidence: 0.73529]
predator(X,'no') :- wingspan(X,N0), N0>10.0. [confidence: 0.59091]
predator(X,'no') :- wingspan(X,N0), N0<=10.0. [confidence: 0.55]
ab1(X) :- wingspan(X,N0), N0<=25.
```

```python
# Get predictions from our new model
learned_conf_predictions = learned_confidence_model.predict(X_test)
learned_conf_labels = [p[0] for p in learned_conf_predictions]

# Calculate accuracy
learned_conf_accuracy = sum(1 for i in range(len(Y_test)) if learned_conf_labels[i] == Y_test[i]) / len(Y_test)

print("--- Learned Confidence Model Evaluation ---")
print(f"True Labels:      {Y_test}")
print(f"Predicted Labels: {learned_conf_labels}")
print(f"Accuracy: {learned_conf_accuracy * 100:.2f}%")
```
```code
--- Learned Confidence Model Evaluation ---
True Labels:      ['yes', 'no', 'no', 'yes', 'no']
Predicted Labels: ['yes', 'no', 'no', 'yes', 'no']
Accuracy: 100.00%
```

## CON-FOLD Citation

```code
@article{mcginness2024confold,
  author    = {McGinness, Lachlan and Baumgartner, Peter},
  title     = {CON-FOLD: Explainable Machine Learning with Confidence},
  journal   = {Theory and Practice of Logic Programming},
  volume    = {24},
  number    = {4},
  pages     = {663--681},
  year      = {2024},
  publisher = {Cambridge University Press}
}
```

which can also be viewed as a pre-print here:

```code
@misc{mcginness2024confold_arxiv,
  title         = {CON-FOLD: Explainable Machine Learning with Confidence}, 
  author        = {McGinness, Lachlan and Baumgartner, Peter},
  year          = {2024},
  eprint        = {2408.07854},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## Original FOLD-RM Citations

```code
@misc{wang2022foldrm,
      title={FOLD-RM: A Scalable and Efficient Inductive Learning Algorithm for Multi-Category Classification of Mixed Data}, 
      author={Huaduo Wang and Farhad Shakerin and Gopal Gupta},
      year={2022},
      eprint={2202.06913},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```code
@misc{wang2021foldr,
      title={FOLD-R++: A Scalable Toolset for Automated Inductive Learning of Default Theories from Mixed Data}, 
      author={Huaduo Wang and Gopal Gupta},
      year={2021},
      eprint={2110.07843},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```