import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import torch
from live.signal_engine import SignalEngine
from model.architecture import HybridModel

class TestSignalEngine(unittest.TestCase):
    @patch('live.signal_engine.load_scalers')
    @patch('live.signal_engine.HybridModel')
    @patch('live.signal_engine.torch.load')
    def setUp(self, mock_torch_load, mock_model, mock_scalers):
        # Mock scalers
        mid_mock = MagicMock()
        mock_scalers.return_value = (mid_mock, mid_mock)
        
        # Mock model
        self.mock_model_instance = MagicMock(spec=HybridModel)
        mock_model.return_value = self.mock_model_instance
        
        self.engine = SignalEngine()

    @patch('live.signal_engine.fetch_latest_candles')
    @patch('live.signal_engine.prepare_live_window')
    def test_inference_pipeline(self, mock_prepare, mock_fetch):
        """Test that the inference pipeline correctly calls fetch and model."""
        # Mock data
        df = pd.DataFrame({
            "open": [100.0] * 300,
            "high": [105.0] * 300,
            "low": [95.0] * 300,
            "close": [102.0] * 300,
            "volume": [1000.0] * 300
        }, index=pd.date_range("2020-01-01", periods=300, freq="1h", tz="UTC"))
        
        mock_fetch.return_value = df
        
        # Mock preprocessor output
        mock_prepare.return_value = (np.zeros((1, 240, 8)), np.zeros((1, 240, 24)))
        
        # Mock model output
        self.mock_model_instance.return_value = (torch.tensor([[0.005]]), torch.tensor([[1.0]]))
        
        # Run inference
        with patch('data.fetcher.fetch_funding_rates', return_value=pd.DataFrame()), \
             patch('data.fetcher.fetch_open_interest', return_value=pd.DataFrame()), \
             patch('data.fetcher.fetch_latest_candles', return_value=df):
            res = self.engine._infer()
            
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["prediction"], 0.005)
        self.assertEqual(res["close_price"], 102.0)

if __name__ == '__main__':
    unittest.main()
