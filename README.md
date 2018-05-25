# Overview
### Directory structure
**root**
  - **code**
    - **backpropagation_code**
      - general_purpose_notebook.ipynb 
      the main notebook for backprop networks to train models, evaluate, create figures, etc
      - **figures**
      figures created by the general purpose notebook
      - **model_comparisons**
      contains json files comparing the scores obtained from different inputs
      - **benchmark_code**
      code for running benchmarks as well as the results obtained from them
    - **modules** -- Python scripts with finished functions used by other notebooks/scripts
    - **special_functions** -- Notebooks used to generate non-neural network results, e.g. modify galaxy catalogues
    - **pso_code**
      - several_inputs_outputs_pso_object_oriented.ipynb
      the latest renditions of the code to train models with a pso. has not been updated in a while
  - **models**
  already trained models. So far nothing of importance has been saved here
