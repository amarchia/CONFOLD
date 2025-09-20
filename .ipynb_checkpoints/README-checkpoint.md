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

## CON-FOLD Citation

```code
@article{mcginness2024confold,
  author    = {McGinness, Lachlan and Baumgartner, Peter},
  title     = {{CON-FOLD}: {E}xplainable {M}achine {L}earning with {C}onfidence},
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
  title         = {{CON-FOLD} -- {E}xplainable {M}achine {L}earning with {C}onfidence}, 
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