"""
Unit tests for QKD simulation module.

Tests the quantum key distribution simulation implementation
for secure key generation between HALE drones.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum_comms.qkd_simulation import (
    QKDSimulation,
    BB84Protocol,
    E91Protocol,
    QuantumChannel,
    ClassicalChannel
)


class TestQKDSimulation:
    """Test cases for QKD simulation implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "protocol": "BB84",
            "key_length": 256,
            "fidelity_threshold": 0.95,
            "error_correction": True,
            "privacy_amplification": True
        }
        self.qkd = QKDSimulation(self.config)
    
    def test_initialization(self):
        """Test QKD simulation initialization."""
        assert self.qkd.config == self.config
        assert self.qkd.protocol == "BB84"
        assert self.qkd.key_length == 256
        assert self.qkd.fidelity_threshold == 0.95
    
    def test_quantum_state_generation(self):
        """Test quantum state generation."""
        states = self.qkd.generate_quantum_states(10)
        
        assert len(states) == 10
        assert all(isinstance(state, dict) for state in states)
        assert all("basis" in state for state in states)
        assert all("bit" in state for state in states)
    
    def test_measurement_simulation(self):
        """Test quantum measurement simulation."""
        quantum_state = {"basis": "Z", "bit": 1}
        
        # Same basis measurement
        result = self.qkd.measure_quantum_state(quantum_state, "Z")
        assert result["bit"] == 1
        assert result["success"] is True
        
        # Different basis measurement
        result = self.qkd.measure_quantum_state(quantum_state, "X")
        assert result["success"] is True
        assert "bit" in result
    
    def test_eavesdropping_detection(self):
        """Test eavesdropping detection."""
        alice_bits = [1, 0, 1, 0, 1]
        bob_bits = [1, 0, 1, 0, 1]  # Perfect correlation
        
        error_rate = self.qkd.calculate_error_rate(alice_bits, bob_bits)
        assert error_rate == 0.0
        
        # Introduce errors
        bob_bits_with_errors = [1, 0, 0, 0, 1]  # One error
        error_rate = self.qkd.calculate_error_rate(alice_bits, bob_bits_with_errors)
        assert error_rate == 0.2
    
    def test_key_reconciliation(self):
        """Test key reconciliation process."""
        alice_key = [1, 0, 1, 0, 1, 0, 1, 0]
        bob_key = [1, 0, 1, 0, 0, 0, 1, 0]  # One error
        
        reconciled_key = self.qkd.reconcile_keys(alice_key, bob_key)
        
        assert len(reconciled_key) <= len(alice_key)
        assert all(bit in [0, 1] for bit in reconciled_key)
    
    def test_privacy_amplification(self):
        """Test privacy amplification."""
        raw_key = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        final_key = self.qkd.privacy_amplification(raw_key)
        
        assert len(final_key) < len(raw_key)
        assert all(bit in [0, 1] for bit in final_key)
    
    def test_complete_qkd_protocol(self):
        """Test complete QKD protocol execution."""
        with patch.object(self.qkd, 'generate_quantum_states') as mock_gen:
            with patch.object(self.qkd, 'measure_quantum_state') as mock_measure:
                mock_gen.return_value = [
                    {"basis": "Z", "bit": 1},
                    {"basis": "X", "bit": 0},
                    {"basis": "Z", "bit": 1}
                ]
                mock_measure.return_value = {"bit": 1, "success": True}
                
                result = self.qkd.execute_protocol()
                
                assert result["success"] is True
                assert "shared_key" in result
                assert "key_length" in result
                assert "error_rate" in result
    
    def test_invalid_protocol(self):
        """Test handling of invalid protocol."""
        invalid_config = self.config.copy()
        invalid_config["protocol"] = "InvalidProtocol"
        
        with pytest.raises(ValueError, match="Unsupported protocol"):
            QKDSimulation(invalid_config)
    
    def test_key_length_validation(self):
        """Test key length validation."""
        invalid_config = self.config.copy()
        invalid_config["key_length"] = 0
        
        with pytest.raises(ValueError, match="Invalid key length"):
            QKDSimulation(invalid_config)
    
    def test_fidelity_threshold_validation(self):
        """Test fidelity threshold validation."""
        invalid_config = self.config.copy()
        invalid_config["fidelity_threshold"] = 1.5  # Invalid threshold
        
        with pytest.raises(ValueError, match="Invalid fidelity threshold"):
            QKDSimulation(invalid_config)


class TestBB84Protocol:
    """Test cases for BB84 protocol implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.bb84 = BB84Protocol()
    
    def test_basis_generation(self):
        """Test basis generation for BB84."""
        bases = self.bb84.generate_bases(10)
        
        assert len(bases) == 10
        assert all(basis in ["Z", "X"] for basis in bases)
    
    def test_state_preparation(self):
        """Test quantum state preparation."""
        state = self.bb84.prepare_state("Z", 1)
        
        assert state["basis"] == "Z"
        assert state["bit"] == 1
        assert "amplitude" in state
    
    def test_measurement(self):
        """Test quantum measurement."""
        state = {"basis": "Z", "bit": 1, "amplitude": [0, 1]}
        
        result = self.bb84.measure(state, "Z")
        assert result["bit"] == 1
        assert result["success"] is True
    
    def test_sifting_process(self):
        """Test sifting process."""
        alice_bases = ["Z", "X", "Z", "X"]
        bob_bases = ["Z", "Z", "X", "X"]
        alice_bits = [1, 0, 1, 0]
        bob_bits = [1, 1, 0, 0]
        
        sifted_data = self.bb84.sift(alice_bases, bob_bases, alice_bits, bob_bits)
        
        assert len(sifted_data["alice_bits"]) == 2
        assert len(sifted_data["bob_bits"]) == 2


class TestE91Protocol:
    """Test cases for E91 protocol implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.e91 = E91Protocol()
    
    def test_entangled_state_generation(self):
        """Test entangled state generation."""
        state = self.e91.generate_entangled_state()
        
        assert "qubit1" in state
        assert "qubit2" in state
        assert "entanglement" in state
    
    def test_bell_state_measurement(self):
        """Test Bell state measurement."""
        entangled_state = {
            "qubit1": {"basis": "Z", "bit": 1},
            "qubit2": {"basis": "Z", "bit": 0},
            "entanglement": True
        }
        
        result = self.e91.measure_bell_state(entangled_state)
        assert "correlation" in result
        assert "success" in result


class TestQuantumChannel:
    """Test cases for quantum channel simulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.channel = QuantumChannel()
    
    def test_channel_initialization(self):
        """Test quantum channel initialization."""
        assert self.channel.fidelity == 0.95
        assert self.channel.loss_rate == 0.1
    
    def test_state_transmission(self):
        """Test quantum state transmission."""
        quantum_state = {"basis": "Z", "bit": 1}
        
        transmitted_state = self.channel.transmit(quantum_state)
        
        assert transmitted_state is not None
        assert "basis" in transmitted_state
        assert "bit" in transmitted_state
    
    def test_channel_noise(self):
        """Test channel noise effects."""
        quantum_state = {"basis": "Z", "bit": 1}
        
        # Test with high noise
        self.channel.fidelity = 0.5
        transmitted_state = self.channel.transmit(quantum_state)
        
        assert transmitted_state is not None


class TestClassicalChannel:
    """Test cases for classical channel simulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.channel = ClassicalChannel()
    
    def test_channel_initialization(self):
        """Test classical channel initialization."""
        assert self.channel.error_rate == 0.01
        assert self.channel.latency == 10  # milliseconds
    
    def test_message_transmission(self):
        """Test message transmission."""
        message = "Test message"
        
        transmitted_message = self.channel.transmit(message)
        
        assert transmitted_message == message
    
    def test_error_injection(self):
        """Test error injection in transmission."""
        message = "Test message"
        
        # Test with high error rate
        self.channel.error_rate = 0.5
        transmitted_message = self.channel.transmit(message)
        
        # Message might be corrupted or lost
        assert transmitted_message is not None


if __name__ == "__main__":
    pytest.main([__file__]) 