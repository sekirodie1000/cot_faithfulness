# Mechanistic Interpretability of Chain-of-Thought Reasoning

We investigate the faithfulness of Chain-of-Thought(CoT) prompting by applying activation patching on sparse features extracted via Sparse Autoencoders (SAEs).


## Code Structure

This project builds on top of:

- https://github.com/sekirodie1000/automated-interpretability (forked from OpenAI)
- https://github.com/sekirodie1000/sparse_coding (forked from Cunningham et al.)

We forked and slightly modified the above repositories to support our experiments.  
The custom code is located in the `experiments/` directory.

### Key Scripts

| Script | Description |
|--------|-------------|
| `activated_patching.py` | Perform activation patching between CoT and NoCoT samples |
| `patch_curve.py` | Plot performance as a function of the number of patched features |
| `activated_box_plot.py` | Visualize activation differences |
| `analyze_score.py` | Analyze explaination scores |

We also made minor adjustments to the original SAE codebase to expose internal activation hooks and support log-prob-free evaluation.


## Installation

This project depends on our custom forks of two repositories.  
Please follow these steps before running any experiments:

```bash
# 1. Clone the sparse_coding fork
git clone https://github.com/sekirodie1000/sparse_coding.git
cd sparse_coding

# 2. Install dependencies
# This will also install neuron-explainer from the forked automated-interpretability repo
pip install -r requirements.txt


