# assent_run.py

import os
import torch
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss
# --- The crucial import step ---
# Import the classes and functions you defined in your other file.
from src.learning.assent_model import ASSENT, train_epoch, evaluate_model, calculate_utility_score


def get_pos_weight(y):
    num_positives = torch.sum(y)
    num_negatives = len(y) - num_positives
    return num_negatives / num_positives + 1e-9

CONSOLE_RUN = False
cwd = os.getcwd()
console_run = cwd + '/src/learning' if CONSOLE_RUN else ''

print("--- Starting ASSENT Model Training ---")

# --- 1. Load Data and Prepare Loaders ---
# Load your final, processed graph dataset
print("Loading dataset...")
file_path = os.path.join(console_run, 'final_graph_dataset.pt')
# file_path = os.path.join(console_run, 'dummy_graph_dataset.pt')
full_dataset = torch.load(file_path, weights_only=False)
# milp_objective_values = torch.load('milp_objective_values.pt')  # Assuming you saved this too

# --- NEW STEP: Determine feature dimensions from the data ---
# Inspect the first sample of the dataset to get the shapes
first_sample = full_dataset[0]
node_feature_dims = {
    'ap': first_sample['ap'].x.shape[1],
    'user': first_sample['user'].x.shape[1],
    'target': first_sample['target'].x.shape[1],
}
print(f"Detected node feature dimensions: {node_feature_dims}")

# Split data
train_size = int(0.8 * len(full_dataset))
train_dataset = full_dataset[:train_size]
test_dataset = full_dataset[train_size:]
# test_milp_utilities = milp_objective_values[train_size:]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples.")

# --- OVERFITTING TEST ---
overfit_batch = next(iter(train_loader))
overfit_dataset = overfit_batch.to_data_list()
# 3. Create a new, proper DataLoader from this tiny dataset.
#    The batch size is set to the size of the dataset to ensure it runs in one batch.
overfit_loader = DataLoader(overfit_dataset, batch_size=len(overfit_dataset))

# --- Calculate pos_weight for the 'x' variable ---
print("Calculating positive weights for loss function...")
y_x_all = torch.cat([data['ap', 'serves', 'user'].y_x for data in train_dataset])
pos_weight_x = get_pos_weight(y_x_all)
y_ytx_all = torch.cat([data['ap', 'senses', 'target'].y_ytx for data in train_dataset])
pos_weight_ytx = get_pos_weight(y_ytx_all)
y_yrx_all = torch.cat([data['ap', 'senses', 'target'].y_yrx for data in train_dataset])
pos_weight_yrx = get_pos_weight(y_yrx_all)
y_tau_all = torch.cat([data['ap'].y_tau for data in train_dataset])
pos_weight_tau = get_pos_weight(y_tau_all)
y_s_all = torch.cat([data['target'].y_s for data in train_dataset])
pos_weight_s = get_pos_weight(y_s_all)

print("\n--- Positive Weights for Weighted BCE Loss ---")
print(f"{'Variable':<16} | {'Pos_Weight':>8}")
print("-" * 30)
print(f"{'τ (AP Mode)':<16} | {pos_weight_tau:8.2f}")
print(f"{'s (Tgt Sched)':<16} | {pos_weight_s:>8.2f}")
print(f"{'x (User Assoc)':<16} | {pos_weight_x:>8.2f}")
print(f"{'y_tx (Sens Tx)':<16} | {pos_weight_ytx:>8.2f}")
print(f"{'y_rx (Sens Rx)':<16} | {pos_weight_yrx:>8.2f}")


# --- 2. Initialize Model and Optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate your ASSENT model (from the imported class)
model = ASSENT(node_feature_dims=node_feature_dims, hidden_dim=256, heads=8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

# --- Create the weighted loss functions ---
loss_fns = {
    'tau': BCEWithLogitsLoss(pos_weight=pos_weight_tau.to(device)),
    's':   BCEWithLogitsLoss(pos_weight=pos_weight_s.to(device)),
    'x':   BCEWithLogitsLoss(pos_weight=pos_weight_x.to(device)),
    'ytx': BCEWithLogitsLoss(pos_weight=pos_weight_ytx.to(device)),
    'yrx': BCEWithLogitsLoss(pos_weight=pos_weight_yrx.to(device)),
}
print("\nWeighted loss functions created successfully.")

print("Model and optimizer initialized.")

# --- 3. Run Training and Evaluation Loop ---
TRAIN_EPOCHS = 500
print("\n--- Starting Training Loop ---")
loss_list = []
for epoch in range(1, TRAIN_EPOCHS+1):
    # Call the imported training function
    loss = train_epoch(model, overfit_loader, optimizer, loss_fns, device)
    loss_list.append(loss)

    # Call the imported evaluation function
    f1 = evaluate_model(model, overfit_loader, device)
    scheduler.step(f1)

    print(f'Epoch: {epoch:03d} | Train Loss: {loss:.4f} | '
          f'Train F1 (tau): {f1:.4f}')

print("\n--- Training Complete ---")

import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(1, TRAIN_EPOCHS+1), loss_list)
plt.show()
