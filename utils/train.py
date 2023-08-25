from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vivit import FactorizedViViT
from utils.misc import MeanValue, TopKAccuracy, get_pytorch_device


def train_vivit_temporal(config, train_data, val_data):
    device = get_pytorch_device()
    torch.random.manual_seed(42)

    # Set up the dataset.
    train_data = DataLoader(
        train_data, batch_size=config["train_batch_size"], shuffle=True
    )
    val_data = DataLoader(val_data, batch_size=config["val_batch_size"])

    # Load and set up the model.
    model = FactorizedViViT(**(config["model"]))
    model.load_state_dict(torch.load(config["starting_weights"]))
    model = model.to(device)

    # Set up the optimizer.
    optimizer_class = getattr(optim, config["optimizer"])
    optimizer = optimizer_class(
        list(model.temporal_model.parameters()) + list(model.classifier.parameters()),
        **config["optimizer_kwargs"],
    )

    # Set up the loss and metrics.
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = MeanValue()
    top_1 = TopKAccuracy(k=1)
    top_5 = TopKAccuracy(k=5)

    # Set up TensorBoard logging.
    if "tensorboard" in config:
        base_name = config["tensorboard"]
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard = SummaryWriter(f"{base_name}_{now_str}")
    else:
        tensorboard = None

    def log_epoch(tb_key, step):
        value_loss = mean_loss.compute()
        value_1 = top_1.compute()
        value_5 = top_5.compute()
        if tensorboard is not None:
            tensorboard.add_scalars("loss", {tb_key: value_loss}, step)
            tensorboard.add_scalars("top_1", {tb_key: value_1}, step)
            tensorboard.add_scalars("top_5", {tb_key: value_5}, step)
        print(f"Loss: {value_loss:.4f}; Top-1: {value_1:.4f}; Top-5: {value_5:.4f}")

    def train_pass(step):
        model.train()
        mean_loss.reset()
        top_1.reset()
        top_5.reset()
        print("Training pass", flush=True)
        for spatial, label in tqdm(train_data, total=len(train_data), ncols=0):
            label = label.to(device)
            output = model(spatial.to(device))
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss.update(loss.item())
            top_1.update(output, label.unsqueeze(dim=-1))
            top_5.update(output, label.unsqueeze(dim=-1))
        log_epoch("train", step)

    def val_pass(step):
        model.eval()
        mean_loss.reset()
        top_1.reset()
        top_5.reset()
        print("Validation pass", flush=True)
        for spatial, label in tqdm(val_data, total=len(val_data), ncols=0):
            label = label.to(device)
            output = model(spatial.to(device))
            loss = loss_function(output, label)
            mean_loss.update(loss.item())
            top_1.update(output, label.unsqueeze(dim=-1))
            top_5.update(output, label.unsqueeze(dim=-1))
        log_epoch("val", step)

    val_pass(0)
    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}", flush=True)
        train_pass(epoch + 1)
        val_pass(epoch + 1)

    if tensorboard is not None:
        tensorboard.close()

    # Save the final weights.
    weight_path = config["output_weights"]
    torch.save(model.state_dict(), weight_path)
    print(f"Saved weights to {weight_path}", flush=True)
