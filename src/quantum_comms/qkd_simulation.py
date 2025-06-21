"""
Quantum Key Distribution (QKD) Simulation Module

This module provides simulation of quantum key distribution protocols
for secure key generation between HALE drones and ground stations.
"""

import time
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.quantum_info import random_statevector, Operator
    from qiskit.providers.aer import QasmSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    logging.warning("Qiskit not available, using mock QKD implementation")
    QISKIT_AVAILABLE = False


class QKDProtocol(Enum):
    """Supported QKD protocols"""
    BB84 = "BB84"
    BBM92 = "BBM92"
    E91 = "E91"


@dataclass
class QKDMetrics:
    """Performance metrics for QKD operations"""
    key_length: int
    raw_key_rate: float  # bits per second
    sifted_key_rate: float  # bits per second
    quantum_bit_error_rate: float
    secret_key_rate: float  # bits per second
    fidelity: float
    generation_time_ms: float
    success: bool


class QuantumChannel:
    """Simulated quantum channel with noise and loss"""
    
    def __init__(self, distance_km: float, loss_db_per_km: float = 0.2):
        self.distance_km = distance_km
        self.loss_db_per_km = loss_db_per_km
        self.total_loss_db = distance_km * loss_db_per_km
        
        # Channel parameters
        self.transmission_efficiency = 10 ** (-self.total_loss_db / 10)
        self.dark_count_rate = 1e-6  # per second
        self.detection_efficiency = 0.8
        
        logging.info(f"Quantum channel initialized: {distance_km}km, loss: {self.total_loss_db:.2f}dB")
    
    def transmit_qubit(self, qubit_state: np.ndarray) -> Optional[np.ndarray]:
        """
        Transmit a qubit through the quantum channel
        
        Args:
            qubit_state: Input qubit state vector
            
        Returns:
            Received qubit state or None if lost
        """
        # Check if qubit is lost due to channel loss
        if np.random.random() > self.transmission_efficiency:
            return None
        
        # Apply channel noise (depolarization)
        noise_probability = 0.01  # 1% depolarization
        if np.random.random() < noise_probability:
            # Apply random Pauli error
            pauli_errors = [
                np.array([[1, 0], [0, 1]]),  # I
                np.array([[0, 1], [1, 0]]),  # X
                np.array([[0, -1j], [1j, 0]]),  # Y
                np.array([[1, 0], [0, -1]])   # Z
            ]
            error = pauli_errors[np.random.randint(0, 4)]
            qubit_state = error @ qubit_state
        
        return qubit_state
    
    def measure_qubit(self, qubit_state: np.ndarray, basis: int) -> int:
        """
        Measure qubit in specified basis
        
        Args:
            qubit_state: Qubit state to measure
            basis: 0 for computational basis, 1 for diagonal basis
            
        Returns:
            Measurement result (0 or 1)
        """
        if basis == 0:
            # Measure in computational basis
            prob_0 = np.abs(qubit_state[0]) ** 2
            return 0 if np.random.random() < prob_0 else 1
        else:
            # Measure in diagonal basis (Hadamard)
            hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            rotated_state = hadamard @ qubit_state
            prob_0 = np.abs(rotated_state[0]) ** 2
            return 0 if np.random.random() < prob_0 else 1


class QKDSimulation:
    """
    Quantum Key Distribution simulation using BB84 protocol
    """
    
    def __init__(self, protocol: QKDProtocol = QKDProtocol.BB84, key_length: int = 256):
        self.protocol = protocol
        self.key_length = key_length
        self.channel = None
        
        if QISKIT_AVAILABLE:
            self.backend = Aer.get_backend('qasm_simulator')
        else:
            self.backend = None
        
        logging.info(f"QKD simulation initialized with {protocol.value} protocol")
    
    def set_channel(self, distance_km: float, loss_db_per_km: float = 0.2):
        """Set quantum channel parameters"""
        self.channel = QuantumChannel(distance_km, loss_db_per_km)
    
    def generate_bb84_key(self) -> Tuple[bytes, QKDMetrics]:
        """
        Generate key using BB84 protocol
        
        Returns:
            Tuple of (key_bytes, metrics)
        """
        start_time = time.time()
        
        if not self.channel:
            raise ValueError("Quantum channel not set")
        
        # Alice's random bits and bases
        alice_bits = np.random.randint(0, 2, self.key_length)
        alice_bases = np.random.randint(0, 2, self.key_length)
        
        # Bob's random measurement bases
        bob_bases = np.random.randint(0, 2, self.key_length)
        
        # Simulate quantum transmission and measurement
        bob_bits = []
        successful_transmissions = 0
        
        for i in range(self.key_length):
            # Prepare qubit state
            if alice_bits[i] == 0:
                qubit_state = np.array([1, 0])  # |0⟩
            else:
                qubit_state = np.array([0, 1])  # |1⟩
            
            # Apply basis transformation
            if alice_bases[i] == 1:
                hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                qubit_state = hadamard @ qubit_state
            
            # Transmit through quantum channel
            received_state = self.channel.transmit_qubit(qubit_state)
            
            if received_state is not None:
                successful_transmissions += 1
                # Bob measures in his chosen basis
                measurement = self.channel.measure_qubit(received_state, bob_bases[i])
                bob_bits.append(measurement)
            else:
                bob_bits.append(-1)  # Lost qubit
        
        # Sift key (keep bits where bases match and transmission succeeded)
        sifted_alice_bits = []
        sifted_bob_bits = []
        
        for i in range(self.key_length):
            if (alice_bases[i] == bob_bases[i] and 
                alice_bits[i] != -1 and bob_bits[i] != -1):
                sifted_alice_bits.append(alice_bits[i])
                sifted_bob_bits.append(bob_bits[i])
        
        # Calculate quantum bit error rate
        if len(sifted_alice_bits) > 0:
            errors = sum(1 for a, b in zip(sifted_alice_bits, sifted_bob_bits) if a != b)
            qber = errors / len(sifted_alice_bits)
        else:
            qber = 1.0
        
        # Error correction and privacy amplification (simplified)
        if qber < 0.11:  # BB84 threshold
            # Apply error correction (simplified)
            corrected_bits = sifted_alice_bits.copy()
            
            # Privacy amplification (reduce key length)
            final_key_length = len(corrected_bits) // 2
            if final_key_length > 0:
                final_key = corrected_bits[:final_key_length]
            else:
                final_key = []
        else:
            final_key = []
        
        # Convert to bytes
        if len(final_key) >= 8:
            key_bytes = bytes([sum(final_key[i:i+8]) for i in range(0, len(final_key), 8)])
        else:
            key_bytes = b""
        
        # Calculate metrics
        generation_time = (time.time() - start_time) * 1000
        raw_key_rate = successful_transmissions / (generation_time / 1000)
        sifted_key_rate = len(sifted_alice_bits) / (generation_time / 1000)
        secret_key_rate = len(final_key) / (generation_time / 1000)
        
        # Calculate fidelity (simplified)
        fidelity = 1.0 - qber if qber < 0.11 else 0.0
        
        metrics = QKDMetrics(
            key_length=len(final_key),
            raw_key_rate=raw_key_rate,
            sifted_key_rate=sifted_key_rate,
            quantum_bit_error_rate=qber,
            secret_key_rate=secret_key_rate,
            fidelity=fidelity,
            generation_time_ms=generation_time,
            success=len(key_bytes) > 0
        )
        
        logging.info(f"BB84 key generation: {len(key_bytes)} bytes, QBER: {qber:.3f}, "
                    f"fidelity: {fidelity:.3f}, time: {generation_time:.2f}ms")
        
        return key_bytes, metrics
    
    def generate_entanglement_key(self) -> Tuple[bytes, QKDMetrics]:
        """
        Generate key using entanglement-based protocol (E91)
        
        Returns:
            Tuple of (key_bytes, metrics)
        """
        start_time = time.time()
        
        if not QISKIT_AVAILABLE:
            logging.warning("Qiskit not available, using mock entanglement")
            return self._mock_entanglement_key()
        
        try:
            # Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            qc = QuantumCircuit(2, 2)
            qc.h(0)  # Hadamard on first qubit
            qc.cx(0, 1)  # CNOT with first qubit as control
            
            # Execute circuit
            job = execute(qc, self.backend, shots=self.key_length)
            result = job.result()
            counts = result.get_counts(qc)
            
            # Extract measurement results
            alice_bits = []
            bob_bits = []
            
            for bitstring, count in counts.items():
                for _ in range(count):
                    alice_bits.append(int(bitstring[0]))
                    bob_bits.append(int(bitstring[1]))
            
            # Calculate QBER
            if len(alice_bits) > 0:
                errors = sum(1 for a, b in zip(alice_bits, bob_bits) if a != b)
                qber = errors / len(alice_bits)
            else:
                qber = 1.0
            
            # Generate final key
            if qber < 0.11:
                final_key = alice_bits[:self.key_length//2]
                key_bytes = bytes([sum(final_key[i:i+8]) for i in range(0, len(final_key), 8)])
            else:
                key_bytes = b""
            
            # Calculate metrics
            generation_time = (time.time() - start_time) * 1000
            fidelity = 1.0 - qber if qber < 0.11 else 0.0
            
            metrics = QKDMetrics(
                key_length=len(final_key) if qber < 0.11 else 0,
                raw_key_rate=len(alice_bits) / (generation_time / 1000),
                sifted_key_rate=len(alice_bits) / (generation_time / 1000),
                quantum_bit_error_rate=qber,
                secret_key_rate=len(final_key) / (generation_time / 1000) if qber < 0.11 else 0,
                fidelity=fidelity,
                generation_time_ms=generation_time,
                success=len(key_bytes) > 0
            )
            
            return key_bytes, metrics
            
        except Exception as e:
            logging.error(f"Entanglement key generation failed: {e}")
            return b"", QKDMetrics(
                key_length=0, raw_key_rate=0, sifted_key_rate=0,
                quantum_bit_error_rate=1.0, secret_key_rate=0,
                fidelity=0, generation_time_ms=(time.time() - start_time) * 1000,
                success=False
            )
    
    def _mock_entanglement_key(self) -> Tuple[bytes, QKDMetrics]:
        """Mock entanglement key generation"""
        start_time = time.time()
        
        # Simulate entangled measurements
        alice_bits = np.random.randint(0, 2, self.key_length)
        bob_bits = alice_bits.copy()  # Perfect correlation
        
        # Add some noise
        noise_indices = np.random.choice(self.key_length, size=self.key_length//10, replace=False)
        for idx in noise_indices:
            bob_bits[idx] = 1 - bob_bits[idx]
        
        # Calculate QBER
        errors = np.sum(alice_bits != bob_bits)
        qber = errors / self.key_length
        
        if qber < 0.11:
            final_key = alice_bits[:self.key_length//2]
            key_bytes = bytes([sum(final_key[i:i+8]) for i in range(0, len(final_key), 8)])
        else:
            key_bytes = b""
        
        generation_time = (time.time() - start_time) * 1000
        
        metrics = QKDMetrics(
            key_length=len(final_key) if qber < 0.11 else 0,
            raw_key_rate=self.key_length / (generation_time / 1000),
            sifted_key_rate=self.key_length / (generation_time / 1000),
            quantum_bit_error_rate=qber,
            secret_key_rate=len(final_key) / (generation_time / 1000) if qber < 0.11 else 0,
            fidelity=1.0 - qber if qber < 0.11 else 0.0,
            generation_time_ms=generation_time,
            success=len(key_bytes) > 0
        )
        
        return key_bytes, metrics
    
    def generate_key(self) -> Tuple[bytes, QKDMetrics]:
        """
        Generate quantum key using the configured protocol
        
        Returns:
            Tuple of (key_bytes, metrics)
        """
        if self.protocol == QKDProtocol.BB84:
            return self.generate_bb84_key()
        elif self.protocol == QKDProtocol.E91:
            return self.generate_entanglement_key()
        else:
            raise ValueError(f"Unsupported QKD protocol: {self.protocol}")


class QKDNetwork:
    """
    Network of QKD nodes for multi-party key distribution
    """
    
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.nodes: Dict[str, QKDSimulation] = {}
        self.channels: Dict[Tuple[str, str], QuantumChannel] = {}
        
        logging.info(f"QKD network initialized: {network_id}")
    
    def add_node(self, node_id: str, protocol: QKDProtocol = QKDProtocol.BB84):
        """Add QKD node to network"""
        self.nodes[node_id] = QKDSimulation(protocol)
        logging.info(f"QKD node added: {node_id}")
    
    def add_channel(self, node1_id: str, node2_id: str, distance_km: float):
        """Add quantum channel between nodes"""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            raise ValueError(f"Nodes not found: {node1_id}, {node2_id}")
        
        channel = QuantumChannel(distance_km)
        self.channels[(node1_id, node2_id)] = channel
        self.channels[(node2_id, node1_id)] = channel  # Bidirectional
        
        # Set channel for both nodes
        self.nodes[node1_id].set_channel(distance_km)
        self.nodes[node2_id].set_channel(distance_km)
        
        logging.info(f"Quantum channel added: {node1_id} <-> {node2_id} ({distance_km}km)")
    
    def generate_shared_key(self, node1_id: str, node2_id: str) -> Tuple[bytes, QKDMetrics]:
        """Generate shared key between two nodes"""
        if (node1_id, node2_id) not in self.channels:
            raise ValueError(f"No channel between {node1_id} and {node2_id}")
        
        # Use node1's QKD simulation to generate key
        return self.nodes[node1_id].generate_key()
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status information"""
        return {
            "network_id": self.network_id,
            "nodes": list(self.nodes.keys()),
            "channels": [(n1, n2, self.channels[(n1, n2)].distance_km) 
                        for (n1, n2) in self.channels.keys() if n1 < n2],
            "total_nodes": len(self.nodes),
            "total_channels": len(self.channels) // 2
        } 