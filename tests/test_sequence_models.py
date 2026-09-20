"""Unit tests for PyTorch LSTM and GRU deep sequence models."""
import pytest
import numpy as np


def test_sequence_dataset_and_models():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from src.sequence_models import CropYieldSequenceDataset, RiceLSTM, RiceGRU, SequenceTrainer

    B = 8   # batch size
    L = 16  # sequence length
    F_seq = 1
    F_static = 10

    seq_data = np.random.uniform(0, 100, size=(B, L, F_seq)).astype(np.float32)
    static_data = np.random.uniform(0, 1, size=(B, F_static)).astype(np.float32)
    targets = np.random.uniform(1000, 3000, size=(B,)).astype(np.float32)

    dataset = CropYieldSequenceDataset(seq_data, static_data, targets)
    assert len(dataset) == B
    item = dataset[0]
    assert item[0].shape == (L, F_seq)
    assert item[1].shape == (F_static,)
    assert item[2].shape == (1,)

    # Test LSTM forward pass
    lstm = RiceLSTM(seq_input_dim=F_seq, static_input_dim=F_static, hidden_dim=16, num_layers=1)
    seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
    static_tensor = torch.tensor(static_data, dtype=torch.float32)

    out_lstm = lstm(seq_tensor, static_tensor)
    assert out_lstm.shape == (B, 1)
    assert torch.isfinite(out_lstm).all()

    # Test GRU forward pass
    gru = RiceGRU(seq_input_dim=F_seq, static_input_dim=F_static, hidden_dim=16, num_layers=1)
    out_gru = gru(seq_tensor, static_tensor)
    assert out_gru.shape == (B, 1)
    assert torch.isfinite(out_gru).all()

    # Test Trainer step
    loader = DataLoader(dataset, batch_size=4)
    trainer = SequenceTrainer(gru, lr=1e-3)
    loss = trainer.train_epoch(loader)
    assert np.isfinite(loss)
    assert loss > 0
