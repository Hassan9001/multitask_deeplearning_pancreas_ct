# multitask_deeplearning_pancreas_ct

Update nnUNet:

- A custom network: Residual Encoder UNet + Classifier

- A modified planner

- A data loader that returns classTarget correctly

- A trainer that handles both segmentation and classification losses

- Update trainer class to handle classTarget in train_step and validation_step

- Split the loss into segmentation and classification components

- Modify network’s forward pass to return both outputs

- other 
  - Your dataset must include both segmentation masks and classification labels.
  - You’ll need to modify the DataLoader to return both.
  - Evaluation and metrics
    - Extend the evaluation script to compute classification metrics (accuracy, AUC, etc.).
    - You might want to log both segmentation and classification performance separately.


