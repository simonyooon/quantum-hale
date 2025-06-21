"""
Unit tests for PQC handshake module.

Tests the post-quantum cryptography handshake implementation
for secure communication between HALE drones.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum_comms.pqc_handshake import (
    PQCHandshake,
    KeyExchangeProtocol,
    SignatureScheme,
    HandshakeState
)


class TestPQCHandshake:
    """Test cases for PQC handshake implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "key_encapsulation": "Kyber768",
            "digital_signature": "Dilithium3",
            "hash_function": "SHA3-256",
            "security_level": 3
        }
        self.handshake = PQCHandshake(self.config)
    
    def test_initialization(self):
        """Test PQC handshake initialization."""
        assert self.handshake.config == self.config
        assert self.handshake.state == HandshakeState.INIT
        assert self.handshake.security_level == 3
    
    def test_key_generation(self):
        """Test key pair generation."""
        with patch('quantum_comms.pqc_handshake.oqs') as mock_oqs:
            mock_kem = Mock()
            mock_kem.generate_keypair.return_value = (b"public_key", b"private_key")
            mock_oqs.KeyEncapsulation.return_value = mock_kem
            
            public_key, private_key = self.handshake.generate_keypair()
            
            assert public_key == b"public_key"
            assert private_key == b"private_key"
            mock_kem.generate_keypair.assert_called_once()
    
    def test_encapsulation(self):
        """Test key encapsulation."""
        with patch('quantum_comms.pqc_handshake.oqs') as mock_oqs:
            mock_kem = Mock()
            mock_kem.encap_secret.return_value = (b"ciphertext", b"shared_secret")
            mock_oqs.KeyEncapsulation.return_value = mock_kem
            
            ciphertext, shared_secret = self.handshake.encapsulate_key(b"public_key")
            
            assert ciphertext == b"ciphertext"
            assert shared_secret == b"shared_secret"
            mock_kem.encap_secret.assert_called_once_with(b"public_key")
    
    def test_decapsulation(self):
        """Test key decapsulation."""
        with patch('quantum_comms.pqc_handshake.oqs') as mock_oqs:
            mock_kem = Mock()
            mock_kem.decap_secret.return_value = b"shared_secret"
            mock_oqs.KeyEncapsulation.return_value = mock_kem
            
            shared_secret = self.handshake.decapsulate_key(b"ciphertext", b"private_key")
            
            assert shared_secret == b"shared_secret"
            mock_kem.decap_secret.assert_called_once_with(b"ciphertext", b"private_key")
    
    def test_signature_generation(self):
        """Test digital signature generation."""
        with patch('quantum_comms.pqc_handshake.oqs') as mock_oqs:
            mock_sig = Mock()
            mock_sig.sign.return_value = b"signature"
            mock_oqs.Signature.return_value = mock_sig
            
            signature = self.handshake.sign_message(b"message", b"private_key")
            
            assert signature == b"signature"
            mock_sig.sign.assert_called_once_with(b"message", b"private_key")
    
    def test_signature_verification(self):
        """Test digital signature verification."""
        with patch('quantum_comms.pqc_handshake.oqs') as mock_oqs:
            mock_sig = Mock()
            mock_sig.verify.return_value = True
            mock_oqs.Signature.return_value = mock_sig
            
            is_valid = self.handshake.verify_signature(b"message", b"signature", b"public_key")
            
            assert is_valid is True
            mock_sig.verify.assert_called_once_with(b"message", b"signature", b"public_key")
    
    def test_handshake_initiation(self):
        """Test handshake initiation."""
        with patch.object(self.handshake, 'generate_keypair') as mock_gen_key:
            mock_gen_key.return_value = (b"public_key", b"private_key")
            
            init_message = self.handshake.initiate_handshake()
            
            assert init_message is not None
            assert "public_key" in init_message
            assert "nonce" in init_message
            assert self.handshake.state == HandshakeState.INITIATED
    
    def test_handshake_response(self):
        """Test handshake response."""
        with patch.object(self.handshake, 'generate_keypair') as mock_gen_key:
            with patch.object(self.handshake, 'encapsulate_key') as mock_encap:
                mock_gen_key.return_value = (b"public_key", b"private_key")
                mock_encap.return_value = (b"ciphertext", b"shared_secret")
                
                init_message = {
                    "public_key": b"remote_public_key",
                    "nonce": b"remote_nonce"
                }
                
                response = self.handshake.respond_to_handshake(init_message)
                
                assert response is not None
                assert "public_key" in response
                assert "ciphertext" in response
                assert "nonce" in response
                assert self.handshake.state == HandshakeState.RESPONDED
    
    def test_handshake_completion(self):
        """Test handshake completion."""
        with patch.object(self.handshake, 'decapsulate_key') as mock_decap:
            mock_decap.return_value = b"shared_secret"
            
            response_message = {
                "public_key": b"remote_public_key",
                "ciphertext": b"ciphertext",
                "nonce": b"remote_nonce"
            }
            
            success = self.handshake.complete_handshake(response_message)
            
            assert success is True
            assert self.handshake.state == HandshakeState.COMPLETED
            assert self.handshake.shared_secret == b"shared_secret"
    
    def test_invalid_handshake_state(self):
        """Test handling of invalid handshake states."""
        self.handshake.state = HandshakeState.COMPLETED
        
        with pytest.raises(ValueError, match="Invalid handshake state"):
            self.handshake.initiate_handshake()
    
    def test_handshake_timeout(self):
        """Test handshake timeout handling."""
        self.handshake.handshake_start_time = 0  # Very old timestamp
        
        with pytest.raises(TimeoutError, match="Handshake timeout"):
            self.handshake.check_timeout()
    
    def test_security_level_validation(self):
        """Test security level validation."""
        invalid_config = self.config.copy()
        invalid_config["security_level"] = 6  # Invalid level
        
        with pytest.raises(ValueError, match="Invalid security level"):
            PQCHandshake(invalid_config)


class TestKeyExchangeProtocol:
    """Test cases for key exchange protocol."""
    
    def test_protocol_initialization(self):
        """Test key exchange protocol initialization."""
        protocol = KeyExchangeProtocol("Kyber768")
        assert protocol.algorithm == "Kyber768"
    
    def test_invalid_algorithm(self):
        """Test handling of invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            KeyExchangeProtocol("InvalidAlgorithm")


class TestSignatureScheme:
    """Test cases for signature scheme."""
    
    def test_scheme_initialization(self):
        """Test signature scheme initialization."""
        scheme = SignatureScheme("Dilithium3")
        assert scheme.algorithm == "Dilithium3"
    
    def test_invalid_scheme(self):
        """Test handling of invalid signature scheme."""
        with pytest.raises(ValueError, match="Unsupported signature scheme"):
            SignatureScheme("InvalidScheme")


if __name__ == "__main__":
    pytest.main([__file__]) 