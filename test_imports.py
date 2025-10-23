#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

print("Testing imports...")

try:
    print("1. Testing core imports...")
    from core.mcts import MCTS
    from core.node import Node
    print("   ✓ Core imports successful")
except Exception as e:
    print(f"   ✗ Core imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Testing agents network imports...")
    from agents.network_enhanced import SiameseNetwork, SiameseNetworkEnhanced
    from agents.network_enhanced import ContrastiveAndRegressionLoss, ContrastiveAndRegressionLossEnhanced
    print("   ✓ Network imports successful")
except Exception as e:
    print(f"   ✗ Network imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing agents data imports...")
    from agents.data_enhanced import EmbeddingPairsDataset
    print("   ✓ Data imports successful")
except Exception as e:
    print(f"   ✗ Data imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Testing trainer imports...")
    from agents.train import SiameseNetworkTrainer
    print("   ✓ Trainer imports successful")
except Exception as e:
    print(f"   ✗ Trainer imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("5. Testing configuration predictor imports...")
    from agents.configuration_predictor import ConfigurationPredictor
    print("   ✓ Configuration predictor imports successful")
except Exception as e:
    print(f"   ✗ Configuration predictor imports failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("6. Testing main.py imports...")
    import sys
    from agents.model_loader import ModelLoader
    from core.mcts import MCTS
    print("   ✓ Main imports successful")
except Exception as e:
    print(f"   ✗ Main imports failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All import tests completed!")
print("="*50)
