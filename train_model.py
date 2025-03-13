import torch
import os
from torch import nn
from torch.optim import Adam
from tqdm import trange
import numpy as np

from keyword_data_loader import Loader_HeySnips
from keyword_spotter import KeyWordSpotter


def training_loop(model, train_loader, val_loader, num_epochs=20):
    """
    Training loop for the keyword spotting model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    optimizer.zero_grad()
    L1_criterion = nn.L1Loss(reduction='none')

    train_batches_per_epoch = train_loader.num_batches
    val_batches_per_epoch = val_loader.num_batches

    print(f"Training batches per epoch: {train_batches_per_epoch}")
    print(f"Validation batches per epoch: {val_batches_per_epoch}")

    weights_dir = "./weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch: {epoch}")
        bar = trange(train_batches_per_epoch)
        avg_epoch_loss = 0.0
        avg_acc = 0.0

        for batch in bar:
            try:
                x_np, y_np = train_loader.get_batch()

                if x_np.size == 0 or y_np.size == 0:
                    print("Skipping empty batch")
                    continue

                x = torch.from_numpy(x_np).float().to(device)
                y = torch.from_numpy(y_np).float().to(device)

                out = model(x)

                loss = weighted_L1_loss(L1_criterion, out, train_batches_per_epoch, y)
                loss.backward()

                pred = (out > 0.5).float()
                acc = (pred == y).float().mean().item()

                avg_epoch_loss += loss.item() / train_batches_per_epoch
                avg_acc += acc / train_batches_per_epoch

                optimizer.step()
                optimizer.zero_grad()

                bar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.6f}, Acc: {acc:.4f}")

            except Exception as e:
                print(f"Error in training batch {batch}: {str(e)}")
                continue

        print(f"Epoch: {epoch}, Training Loss: {avg_epoch_loss:.6f}, Training Acc: {avg_acc:.4f}")

        # Save model weights
        weight_path = os.path.join(weights_dir, f"epoch_{epoch}_{avg_epoch_loss:.6f}_{avg_acc:.6f}.weights")
        torch.save(model.state_dict(), weight_path)
        print(f"Saved model weights to {weight_path}")


def weighted_L1_loss(L1_criterion, out, batches_per_epoch, y):
    eps = (.001 / batches_per_epoch)
    return (L1_criterion(out, y) * (y + y.mean() + eps)).mean()


def train():
    base_path = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_path, "hey_snips_fl_amt")
    train_json_path = os.path.join(data_path, "train.json")
    test_json_path = os.path.join(data_path, "test.json")
    mfcc_path = os.path.join(data_path, "mfcc")

    print(f"Using data path: {data_path}")
    print(f"Train JSON: {train_json_path}")
    print(f"Test JSON: {test_json_path}")
    print(f"MFCC path: {mfcc_path}")

    if not os.path.exists(train_json_path) or not os.path.exists(test_json_path):
        raise FileNotFoundError("Missing train.json or test.json. Check file placement.")

    if not os.path.exists(mfcc_path):
        raise FileNotFoundError("MFCC directory not found. Ensure MFCC processing was done correctly.")

    mfcc_files = [f for f in os.listdir(mfcc_path) if f.endswith('.mfcc.npy')]
    print(f"Found {len(mfcc_files)} MFCC files")

    train_loader = Loader_HeySnips(train_json_path, mfcc_path, batch_size=100)
    val_loader = Loader_HeySnips(test_json_path, mfcc_path, batch_size=100)

    model = KeyWordSpotter(20)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count:,} trainable parameters")

    print("Starting training...")
    training_loop(model, train_loader, val_loader, num_epochs=20)
    print("Training complete!")


if __name__ == '__main__':
    train()
