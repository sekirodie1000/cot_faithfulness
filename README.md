# Mechanistic Interpretability of Chain-of-Thought Reasoning

We investigate the faithfulness of Chain-of-Thought(CoT) prompting by applying activation patching on sparse features extracted via Sparse Autoencoders (SAEs).


## Code Structure

This project is based on the following open-source repositories:

- [`automated-interpretability`](https://github.com/openai/automated-interpretability) (originally from OpenAI)
- [`sparse_coding`](https://github.com/HoagyC/sparse_coding) (from Cunningham et al.)

We made targeted modifications to both codebases to support our experiments.  
The full modified code is included in this repository.  
All modifications are documented in the file [`changes.diff`](./changes.diff).

Our experiment-specific logic is implemented in the `experiments/` directory.

### Key Scripts

| Script | Description |
|--------|-------------|
| `activated_patching.py` | Perform activation patching between CoT and NoCoT samples |
| `patch_curve.py` | Plot performance as a function of the number of patched features |
| `activated_box_plot.py` | Visualize activation differences |
| `analyze_score.py` | Analyze explaination scores |

We also made minor adjustments to the original SAE codebase to expose internal activation hooks and support log-prob-free evaluation.


## Installation

This project depends on our modified versions of both `sparse_coding` and `automated-interpretability`.  
To set up the environment:

1. **Clone and install `sparse_coding`**:

   ```bash
   git clone https://github.com/sekirodie1000/cot_faithfulness.git
   cd sparse_coding
   pip install -r requirements.txt
   ```

   > Note: The `requirements.txt` will install the original `automated-interpretability` repo as a dependency.

2. **Manually replace the installed `automated-interpretability`**:

   After installation, locate the installed `automated_interpretability/` package inside your Python environment (usually under `site-packages/`), and replace it with the version provided in this repository's `automated-interpretability/` folder.

3. **Run experiments**:

   Once dependencies are installed and the code is replaced, navigate to the `experiments/` directory and execute the relevant scripts.
