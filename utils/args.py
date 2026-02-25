import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Predictive Alerting — LSTM Trainer")

    # Data
    parser.add_argument("--rul-threshold", type=int,   default=30,  help="RUL cycles below which an engine is labelled as near-failure (default: 30)")
    parser.add_argument("--window",        type=int,   default=30,  help="Sliding window length in cycles (default: 30)")
    parser.add_argument("--test-size",     type=float, default=0.2, help="Fraction of engines held out for testing (default: 0.2)")
    parser.add_argument("--batch-size",    type=int,   default=64,  help="DataLoader batch size (default: 64)")
    parser.add_argument("--seed",          type=int,   default=42,  help="Random seed for train/test split (default: 42)")

    # Model
    parser.add_argument("--hidden-size",   type=int,   default=64,  help="LSTM hidden state size (default: 64)")
    parser.add_argument("--num-layers",    type=int,   default=2,   help="Number of stacked LSTM layers (default: 2)")
    parser.add_argument("--dropout",       type=float, default=0.3, help="Dropout probability (default: 0.3)")

    # Training
    parser.add_argument("--epochs",        type=int,   default=20,  help="Number of training epochs (default: 20)")
    parser.add_argument("--lr",            type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--threshold",     type=float, default=0.5, help="Alert probability threshold (default: 0.5)")

    return parser.parse_args()
