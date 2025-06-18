"""
Quantum-Compatible HALE Drone: Post-Quantum Cryptography Handshake Implementation
==============================================================================

This module implements the core PQC protocols for secure communication between
the HALE drone and ground control station using NIST-standardized algorithms.

Key Components:
- Kyber: Key Encapsulation Mechanism (ML-KEM)
- Dilithium: Digital Signature Algorithm (ML-DSA)  
- Handshake Orchestration: Complete protocol implementation
- Performance Monitoring: Latency and security metrics

Author: Quantum HALE Development Team
License: MIT (for simulation/research purposes)
"""

import time
import hashlib
import secrets
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from enum import Enum

# Post-Quantum Cryptography Libraries
import oqs  # liboqs Python bindings
import numpy as np

# Quantum Simulation (for QKD emulation)
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import random_statevector

# Performance Monitoring
from prometheus_client import Counter, Histogram, Gauge
import json


class HandshakeState(Enum):
    """Handshake protocol states"""
    INIT = "INIT"
    KEY_EXCHANGE = "KEY_EXCHANGE" 
    AUTHENTICATION = "AUTHENTICATION"
    ESTABLISHED = "ESTABLISHED"
    FAILED = "FAILED"


class SecurityLevel(Enum):
    """NIST security categories"""
    CATEGORY_1 = 1  # AES-128 equivalent
    CATEGORY_3 = 3  # AES-192 equivalent  
    CATEGORY_5 = 5  # AES-256 equivalent


@dataclass
class HandshakeMetrics:
    """Performance and security metrics for handshake"""
    start_time: float
    end_time: float
    total_latency_ms: float
    key_generation_time_ms: float
    signature_time_ms: float
    verification_time_ms: float
    bytes_transmitted: int
    security_level: SecurityLevel
    success: bool
    error_message: Optional[str] = None


class PQCConfiguration:
    """Post-Quantum Cryptography configuration parameters"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.CATEGORY_3):
        self.security_level = security_level
        
        # NIST Standardized Algorithms
        if security_level == SecurityLevel.CATEGORY_1:
            self.kem_algorithm = "Kyber512"
            self.sig_algorithm = "Dilithium2"
        elif security_level == SecurityLevel.CATEGORY_3:
            self.kem_algorithm = "Kyber768"  
            self.sig_algorithm = "Dilithium3"
        else:  # CATEGORY_5
            self.kem_algorithm = "Kyber1024"
            self.sig_algorithm = "Dilithium5"
            
        # Protocol parameters
        self.session_timeout_seconds = 3600  # 1 hour
        self.max_retries = 3
        self.handshake_timeout_ms = 5000  # 5 second timeout


class QuantumKeyDistribution:
    """
    Simulated Quantum Key Distribution for future hardware integration
    Currently implements BB84 protocol simulation using Qiskit
    """
    
    def __init__(self, key_length: int = 256):
        self.key_length = key_length
        self.backend = Aer.get_backend('qasm_simulator')
        
    def generate_quantum_key(self) -> Tuple[bytes, float]:
        """
        Simulate BB84 quantum key generation
        
        Returns:
            Tuple of (quantum_key_bytes, fidelity_score)
        """
        start_time = time.time()
        
        # Create quantum circuit for BB84
        qc = QuantumCircuit(self.key_length, self.key_length)
        
        # Alice's random bits and bases
        alice_bits = np.random.randint(0, 2, self.key_length)
        alice_bases = np.random.randint(0, 2, self.key_length)
        
        # Encode qubits based on Alice's choices
        for i in range(self.key_length):
            if alice_bits[i] == 1:
                qc.x(i)  # Bit flip for |1⟩
            if alice_bases[i] == 1:
                qc.h(i)  # Hadamard for diagonal basis
                
        # Bob's random measurement bases
        bob_bases = np.random.randint(0, 2, self.key_length)
        
        for i in range(self.key_length):
            if bob_bases[i] == 1:
                qc.h(i)  # Measure in diagonal basis
            qc.measure(i, i)
        
        # Execute quantum circuit
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        bob_bits = list(counts.keys())[0]
        
        # Sift key (keep bits where bases match)
        sifted_key = []
        for i in range(self.key_length):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])
        
        # Convert to bytes (pad if necessary)
        if len(sifted_key) < 128:  # Minimum 128 bits for security
            sifted_key.extend([0] * (128 - len(sifted_key)))
            
        key_bytes = bytes([sum(sifted_key[i:i+8]) for i in range(0, len(sifted_key), 8)])
        
        # Calculate fidelity (simulated - would be measured in real system)
        fidelity = 0.95 + 0.05 * np.random.random()  # 95-100% typical range
        
        generation_time = (time.time() - start_time) * 1000
        
        logging.info(f"QKD key generated: {len(key_bytes)} bytes, fidelity: {fidelity:.3f}, time: {generation_time:.2f}ms")
        
        return key_bytes, fidelity


class PQCHandshake:
    """
    Main Post-Quantum Cryptography handshake implementation
    Handles key exchange, authentication, and session establishment
    """
    
    def __init__(self, config: PQCConfiguration, node_id: str):
        self.config = config
        self.node_id = node_id
        self.state = HandshakeState.INIT
        
        # Initialize PQC algorithms
        try:
            self.kem = oqs.KeyEncapsulation(config.kem_algorithm)
            self.signature = oqs.Signature(config.sig_algorithm)
        except Exception as e:
            logging.error(f"Failed to initialize PQC algorithms: {e}")
            raise
            
        # Generate long-term identity keys
        self.identity_public_key = self.signature.generate_keypair()
        
        # QKD integration (for future hardware)
        self.qkd = QuantumKeyDistribution()
        
        # Performance metrics
        self.metrics_counter = Counter('pqc_handshakes_total', 'Total handshakes attempted', ['status'])
        self.latency_histogram = Histogram('pqc_handshake_latency_seconds', 'Handshake latency')
        self.key_generation_gauge = Gauge('pqc_key_generation_time_ms', 'Key generation time')
        
        # Session storage
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logging.info(f"PQC Handshake initialized for node {node_id} with {config.kem_algorithm}/{config.sig_algorithm}")
        
    def initiate_handshake(self, peer_id: str, peer_public_key: bytes) -> Tuple[bytes, str]:
        """
        Initiate handshake with remote peer (ground station or other drone)
        
        Args:
            peer_id: Identifier of the peer node
            peer_public_key: Peer's long-term public key
            
        Returns:
            Tuple of (handshake_message, session_id)
        """
        start_time = time.time()
        session_id = secrets.token_hex(16)
        
        try:
            self.state = HandshakeState.KEY_EXCHANGE
            
            # Generate ephemeral key pair for this session
            kem_public_key = self.kem.generate_keypair()
            
            # Create handshake initiation message
            message_data = {
                'version': '1.0',
                'session_id': session_id,
                'node_id': self.node_id,
                'timestamp': int(time.time()),
                'kem_algorithm': self.config.kem_algorithm,
                'sig_algorithm': self.config.sig_algorithm,
                'kem_public_key': kem_public_key.hex(),
                'identity_public_key': self.identity_public_key.hex()
            }
            
            # Sign the message with identity key
            message_json = json.dumps(message_data, sort_keys=True)
            signature = self.signature.sign(message_json.encode())
            
            handshake_message = {
                'data': message_data,
                'signature': signature.hex()
            }
            
            # Store session state
            self.active_sessions[session_id] = {
                'peer_id': peer_id,
                'peer_public_key': peer_public_key,
                'state': HandshakeState.KEY_EXCHANGE,
                'start_time': start_time,
                'kem_private_key': self.kem.export_secret_key(),
                'kem_public_key': kem_public_key
            }
            
            key_gen_time = (time.time() - start_time) * 1000
            self.key_generation_gauge.set(key_gen_time)
            
            logging.info(f"Handshake initiated with {peer_id}, session: {session_id}")
            
            return json.dumps(handshake_message).encode(), session_id
            
        except Exception as e:
            self.state = HandshakeState.FAILED
            self.metrics_counter.labels(status='failed').inc()
            logging.error(f"Handshake initiation failed: {e}")
            raise
    
    def process_handshake_response(self, response_data: bytes, session_id: str) -> Tuple[bytes, bytes]:
        """
        Process handshake response and complete key agreement
        
        Args:
            response_data: Response message from peer
            session_id: Session identifier
            
        Returns:
            Tuple of (shared_secret, final_handshake_message)
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Unknown session ID: {session_id}")
                
            session = self.active_sessions[session_id]
            response_message = json.loads(response_data.decode())
            
            # Verify peer signature
            peer_signature = oqs.Signature(self.config.sig_algorithm)
            peer_signature.import_public_key(session['peer_public_key'])
            
            message_json = json.dumps(response_message['data'], sort_keys=True)
            if not peer_signature.verify(message_json.encode(), bytes.fromhex(response_message['signature'])):
                raise ValueError("Signature verification failed")
            
            # Extract encapsulated key from response
            encapsulated_key = bytes.fromhex(response_message['data']['encapsulated_key'])
            
            # Decapsulate to get shared secret
            self.kem.import_secret_key(session['kem_private_key'])
            shared_secret = self.kem.decaps_secret(encapsulated_key)
            
            # Generate session authentication token
            auth_token = hashlib.sha256(shared_secret + session_id.encode()).digest()
            
            # Create final handshake message
            final_message = {
                'session_id': session_id,
                'status': 'established',
                'auth_token': auth_token.hex()
            }
            
            # Sign final message
            final_json = json.dumps(final_message, sort_keys=True)
            final_signature = self.signature.sign(final_json.encode())
            
            final_handshake = {
                'data': final_message,
                'signature': final_signature.hex()
            }
            
            # Update session state
            session['state'] = HandshakeState.ESTABLISHED
            session['shared_secret'] = shared_secret
            session['auth_token'] = auth_token
            
            self.state = HandshakeState.ESTABLISHED
            
            # Record metrics
            total_time = time.time() - session['start_time']
            self.latency_histogram.observe(total_time)
            self.metrics_counter.labels(status='success').inc()
            
            logging.info(f"Handshake completed successfully for session {session_id}")
            
            return shared_secret, json.dumps(final_handshake).encode()
            
        except Exception as e:
            self.state = HandshakeState.FAILED
            self.metrics_counter.labels(status='failed').inc()
            logging.error(f"Handshake response processing failed: {e}")
            raise
    
    def handle_incoming_handshake(self, handshake_data: bytes) -> bytes:
        """
        Handle incoming handshake initiation from peer
        
        Args:
            handshake_data: Initial handshake message
            
        Returns:
            Response message bytes
        """
        try:
            handshake_message = json.loads(handshake_data.decode())
            data = handshake_message['data']
            
            # Verify peer signature
            peer_public_key = bytes.fromhex(data['identity_public_key'])
            peer_signature = oqs.Signature(self.config.sig_algorithm)
            peer_signature.import_public_key(peer_public_key)
            
            message_json = json.dumps(data, sort_keys=True)
            signature = bytes.fromhex(handshake_message['signature'])
            
            if not peer_signature.verify(message_json.encode(), signature):
                raise ValueError("Peer signature verification failed")
            
            # Extract peer's KEM public key
            peer_kem_public_key = bytes.fromhex(data['kem_public_key'])
            
            # Encapsulate secret using peer's public key
            kem_temp = oqs.KeyEncapsulation(data['kem_algorithm'])
            kem_temp.import_public_key(peer_kem_public_key)
            encapsulated_key, shared_secret = kem_temp.encaps_secret()
            
            # Create response message
            response_data = {
                'session_id': data['session_id'],
                'node_id': self.node_id,
                'timestamp': int(time.time()),
                'encapsulated_key': encapsulated_key.hex(),
                'status': 'key_exchanged'
            }
            
            # Sign response
            response_json = json.dumps(response_data, sort_keys=True)
            response_signature = self.signature.sign(response_json.encode())
            
            response_message = {
                'data': response_data,
                'signature': response_signature.hex()
            }
            
            # Store session
            session_id = data['session_id']
            self.active_sessions[session_id] = {
                'peer_id': data['node_id'],
                'peer_public_key': peer_public_key,
                'shared_secret': shared_secret,
                'state': HandshakeState.AUTHENTICATION,
                'start_time': time.time()
            }
            
            logging.info(f"Processed incoming handshake from {data['node_id']}")
            
            return json.dumps(response_message).encode()
            
        except Exception as e:
            logging.error(f"Incoming handshake processing failed: {e}")
            raise
    
    def get_session_key(self, session_id: str) -> Optional[bytes]:
        """Get the shared secret for an established session"""
        session = self.active_sessions.get(session_id)
        if session and session['state'] == HandshakeState.ESTABLISHED:
            return session['shared_secret']
        return None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions based on timeout"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session['start_time'] > self.config.session_timeout_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logging.info(f"Cleaned up expired session: {session_id}")
    
    def get_handshake_metrics(self, session_id: str) -> Optional[HandshakeMetrics]:
        """Get performance metrics for a completed handshake"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
            
        return HandshakeMetrics(
            start_time=session['start_time'],
            end_time=time.time(),
            total_latency_ms=(time.time() - session['start_time']) * 1000,
            key_generation_time_ms=0,  # Would be populated during actual handshake
            signature_time_ms=0,
            verification_time_ms=0,
            bytes_transmitted=0,  # Would be calculated during handshake
            security_level=self.config.security_level,
            success=session['state'] == HandshakeState.ESTABLISHED
        )


# Example usage and testing framework
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def test_pqc_handshake():
        """Test the PQC handshake implementation"""
        
        # Create two nodes (drone and ground station)
        config = PQCConfiguration(SecurityLevel.CATEGORY_3)
        
        drone_node = PQCHandshake(config, "DRONE_001")
        ground_node = PQCHandshake(config, "GROUND_001")
        
        # Exchange identity public keys (would happen through certificate exchange)
        drone_identity_key = drone_node.identity_public_key
        ground_identity_key = ground_node.identity_public_key
        
        try:
            # Drone initiates handshake with ground station
            logging.info("=== Starting PQC Handshake Test ===")
            
            handshake_init, session_id = drone_node.initiate_handshake("GROUND_001", ground_identity_key)
            logging.info(f"Drone initiated handshake, session: {session_id}")
            
            # Ground station processes handshake
            handshake_response = ground_node.handle_incoming_handshake(handshake_init)
            logging.info("Ground station processed handshake initiation")
            
            # Drone completes handshake
            shared_secret, final_message = drone_node.process_handshake_response(handshake_response, session_id)
            logging.info("Drone completed handshake")
            
            # Verify shared secrets match
            ground_secret = ground_node.get_session_key(session_id)
            
            if shared_secret == ground_secret:
                logging.info("Handshake successful! Shared secrets match.")
                logging.info(f"Shared secret length: {len(shared_secret)} bytes")
                logging.info(f"Session ID: {session_id}")
                
                # Get performance metrics
                metrics = drone_node.get_handshake_metrics(session_id)
                if metrics:
                    logging.info(f"Handshake latency: {metrics.total_latency_ms:.2f}ms")
                    logging.info(f"Security level: {metrics.security_level}")
                
                # Test QKD simulation
                logging.info("=== Testing QKD Simulation ===")
                qkd = QuantumKeyDistribution(256)
                quantum_key, fidelity = qkd.generate_quantum_key()
                logging.info(f"QKD key generated: {len(quantum_key)} bytes, fidelity: {fidelity:.3f}")
                
            else:
                logging.error("Handshake failed! Shared secrets don't match.")
                
        except Exception as e:
            logging.error(f"Handshake test failed: {e}")
            raise
        
        finally:
            # Cleanup
            drone_node.cleanup_expired_sessions()
            ground_node.cleanup_expired_sessions()
    
    # Run the test
    asyncio.run(test_pqc_handshake())


class HALEDroneCommunications:
    """
    High-level communications module for HALE drone
    Integrates PQC handshake with mission operations
    """
    
    def __init__(self, drone_id: str, security_level: SecurityLevel = SecurityLevel.CATEGORY_3):
        self.drone_id = drone_id
        self.config = PQCConfiguration(security_level)
        self.pqc_handshake = PQCHandshake(self.config, drone_id)
        
        # Communication channels
        self.active_channels: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.message_counter = Counter('drone_messages_total', 'Total messages sent', ['type', 'status'])
        self.encryption_time = Histogram('encryption_time_seconds', 'Message encryption time')
        
        logging.info(f"HALE drone communications initialized for {drone_id}")
    
    def establish_ground_link(self, ground_station_id: str, ground_public_key: bytes) -> str:
        """
        Establish secure link with ground control station
        
        Args:
            ground_station_id: Ground station identifier
            ground_public_key: Ground station's public key
            
        Returns:
            Session ID for the established link
        """
        try:
            handshake_init, session_id = self.pqc_handshake.initiate_handshake(
                ground_station_id, ground_public_key
            )
            
            # In real implementation, this would be sent via radio
            # For simulation, we'll store it for the test framework
            self.active_channels[ground_station_id] = {
                'session_id': session_id,
                'type': 'ground_control',
                'status': 'handshake_initiated',
                'handshake_data': handshake_init
            }
            
            self.message_counter.labels(type='handshake', status='initiated').inc()
            
            logging.info(f"Ground link handshake initiated with {ground_station_id}")
            return session_id
            
        except Exception as e:
            self.message_counter.labels(type='handshake', status='failed').inc()
            logging.error(f"Failed to establish ground link: {e}")
            raise
    
    def establish_mesh_link(self, peer_drone_id: str, peer_public_key: bytes) -> str:
        """
        Establish secure mesh link with another drone
        
        Args:
            peer_drone_id: Peer drone identifier
            peer_public_key: Peer drone's public key
            
        Returns:
            Session ID for the established mesh link
        """
        try:
            handshake_init, session_id = self.pqc_handshake.initiate_handshake(
                peer_drone_id, peer_public_key
            )
            
            self.active_channels[peer_drone_id] = {
                'session_id': session_id,
                'type': 'mesh_peer',
                'status': 'handshake_initiated',
                'handshake_data': handshake_init
            }
            
            self.message_counter.labels(type='mesh_handshake', status='initiated').inc()
            
            logging.info(f"Mesh link handshake initiated with {peer_drone_id}")
            return session_id
            
        except Exception as e:
            self.message_counter.labels(type='mesh_handshake', status='failed').inc()
            logging.error(f"Failed to establish mesh link: {e}")
            raise
    
    def send_telemetry(self, destination_id: str, telemetry_data: Dict[str, Any]) -> bool:
        """
        Send encrypted telemetry data to destination
        
        Args:
            destination_id: Destination node ID
            telemetry_data: Telemetry payload
            
        Returns:
            Success status
        """
        start_time = time.time()
        
        try:
            if destination_id not in self.active_channels:
                raise ValueError(f"No active channel to {destination_id}")
            
            channel = self.active_channels[destination_id]
            session_id = channel['session_id']
            
            # Get session key for encryption
            session_key = self.pqc_handshake.get_session_key(session_id)
            if not session_key:
                raise ValueError(f"No session key available for {destination_id}")
            
            # Create telemetry message
            message = {
                'timestamp': time.time(),
                'drone_id': self.drone_id,
                'session_id': session_id,
                'message_type': 'telemetry',
                'data': telemetry_data
            }
            
            # Encrypt with session key (AES-GCM)
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Derive AES key from shared secret
            aes_key = hashlib.sha256(session_key).digest()[:32]  # 256-bit key
            aesgcm = AESGCM(aes_key)
            
            # Generate nonce
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            
            # Encrypt message
            message_json = json.dumps(message)
            ciphertext = aesgcm.encrypt(nonce, message_json.encode(), None)
            
            encrypted_message = {
                'nonce': nonce.hex(),
                'ciphertext': ciphertext.hex(),
                'sender': self.drone_id
            }
            
            # Record metrics
            encryption_time = (time.time() - start_time)
            self.encryption_time.observe(encryption_time)
            self.message_counter.labels(type='telemetry', status='sent').inc()
            
            # In real implementation, this would be transmitted via radio
            logging.info(f"Telemetry sent to {destination_id}, size: {len(ciphertext)} bytes")
            
            return True
            
        except Exception as e:
            self.message_counter.labels(type='telemetry', status='failed').inc()
            logging.error(f"Failed to send telemetry to {destination_id}: {e}")
            return False
    
    def send_isr_data(self, destination_id: str, image_data: bytes, metadata: Dict[str, Any]) -> bool:
        """
        Send encrypted ISR (Intelligence, Surveillance, Reconnaissance) data
        
        Args:
            destination_id: Destination node ID
            image_data: Raw image/sensor data
            metadata: ISR metadata (coordinates, timestamp, etc.)
            
        Returns:
            Success status
        """
        start_time = time.time()
        
        try:
            if destination_id not in self.active_channels:
                raise ValueError(f"No active channel to {destination_id}")
            
            channel = self.active_channels[destination_id]
            session_id = channel['session_id']
            
            # Get session key for encryption
            session_key = self.pqc_handshake.get_session_key(session_id)
            if not session_key:
                raise ValueError(f"No session key available for {destination_id}")
            
            # Create ISR message
            message = {
                'timestamp': time.time(),
                'drone_id': self.drone_id,
                'session_id': session_id,
                'message_type': 'isr_data',
                'metadata': metadata,
                'data_hash': hashlib.sha256(image_data).hexdigest()
            }
            
            # Encrypt with session key
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = hashlib.sha256(session_key).digest()[:32]
            aesgcm = AESGCM(aes_key)
            
            # Encrypt metadata and image data separately for efficiency
            metadata_nonce = secrets.token_bytes(12)
            data_nonce = secrets.token_bytes(12)
            
            encrypted_metadata = aesgcm.encrypt(metadata_nonce, json.dumps(message).encode(), None)
            encrypted_data = aesgcm.encrypt(data_nonce, image_data, None)
            
            encrypted_message = {
                'metadata_nonce': metadata_nonce.hex(),
                'data_nonce': data_nonce.hex(),
                'encrypted_metadata': encrypted_metadata.hex(),
                'encrypted_data': encrypted_data.hex(),
                'sender': self.drone_id
            }
            
            # Record metrics
            encryption_time = (time.time() - start_time)
            self.encryption_time.observe(encryption_time)
            self.message_counter.labels(type='isr_data', status='sent').inc()
            
            logging.info(f"ISR data sent to {destination_id}, payload: {len(image_data)} bytes")
            
            return True
            
        except Exception as e:
            self.message_counter.labels(type='isr_data', status='failed').inc()
            logging.error(f"Failed to send ISR data to {destination_id}: {e}")
            return False
    
    def handle_incoming_message(self, sender_id: str, encrypted_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decrypt and process incoming message
        
        Args:
            sender_id: Sender node ID
            encrypted_message: Encrypted message payload
            
        Returns:
            Decrypted message or None if decryption fails
        """
        try:
            if sender_id not in self.active_channels:
                # Try to handle as new handshake
                if 'handshake_data' in encrypted_message:
                    response = self.pqc_handshake.handle_incoming_handshake(
                        encrypted_message['handshake_data']
                    )
                    return {'type': 'handshake_response', 'data': response}
                else:
                    raise ValueError(f"No active channel with {sender_id}")
            
            channel = self.active_channels[sender_id]
            session_id = channel['session_id']
            
            # Get session key for decryption
            session_key = self.pqc_handshake.get_session_key(session_id)
            if not session_key:
                raise ValueError(f"No session key available for {sender_id}")
            
            # Decrypt message
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aes_key = hashlib.sha256(session_key).digest()[:32]
            aesgcm = AESGCM(aes_key)
            
            nonce = bytes.fromhex(encrypted_message['nonce'])
            ciphertext = bytes.fromhex(encrypted_message['ciphertext'])
            
            # Decrypt and parse
            decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
            message = json.loads(decrypted_data.decode())
            
            self.message_counter.labels(type='received', status='success').inc()
            
            logging.info(f"Message received from {sender_id}, type: {message.get('message_type', 'unknown')}")
            
            return message
            
        except Exception as e:
            self.message_counter.labels(type='received', status='failed').inc()
            logging.error(f"Failed to decrypt message from {sender_id}: {e}")
            return None
    
    def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active communication channels"""
        status = {}
        for node_id, channel in self.active_channels.items():
            session_key = self.pqc_handshake.get_session_key(channel['session_id'])
            status[node_id] = {
                'type': channel['type'],
                'status': channel['status'],
                'session_established': session_key is not None,
                'session_id': channel['session_id']
            }
        return status


class DroneNetworkManager:
    """
    Network manager for coordinating multiple drone communications
    Handles mesh networking and relay operations
    """
    
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.nodes: Dict[str, HALEDroneCommunications] = {}
        self.network_topology: Dict[str, set] = {}
        
        # Network metrics
        self.network_messages = Counter('network_messages_total', 'Network messages', ['type'])
        self.route_discovery_time = Histogram('route_discovery_seconds', 'Route discovery time')
        
        logging.info(f"Drone network manager initialized for network {network_id}")
    
    def add_drone(self, drone_id: str, security_level: SecurityLevel = SecurityLevel.CATEGORY_3):
        """Add a drone to the network"""
        drone_comms = HALEDroneCommunications(drone_id, security_level)
        self.nodes[drone_id] = drone_comms
        self.network_topology[drone_id] = set()
        
        logging.info(f"Added drone {drone_id} to network {self.network_id}")
    
    def establish_mesh_connections(self, connections: list):
        """
        Establish mesh connections between drones
        
        Args:
            connections: List of (drone1_id, drone2_id) tuples
        """
        for drone1_id, drone2_id in connections:
            if drone1_id in self.nodes and drone2_id in self.nodes:
                # Exchange public keys and establish links
                drone1 = self.nodes[drone1_id]
                drone2 = self.nodes[drone2_id]
                
                # Simulate key exchange (in real system, this would be via certificate authority)
                drone1_key = drone1.pqc_handshake.identity_public_key
                drone2_key = drone2.pqc_handshake.identity_public_key
                
                # Establish bidirectional links
                session1 = drone1.establish_mesh_link(drone2_id, drone2_key)
                session2 = drone2.establish_mesh_link(drone1_id, drone1_key)
                
                # Update topology
                self.network_topology[drone1_id].add(drone2_id)
                self.network_topology[drone2_id].add(drone1_id)
                
                logging.info(f"Mesh connection established between {drone1_id} and {drone2_id}")
    
    def find_route(self, source_id: str, destination_id: str) -> Optional[list]:
        """
        Find routing path through mesh network using Dijkstra's algorithm
        
        Args:
            source_id: Source drone ID
            destination_id: Destination drone ID
            
        Returns:
            List of node IDs representing the route, or None if no route exists
        """
        start_time = time.time()
        
        try:
            if source_id not in self.network_topology or destination_id not in self.network_topology:
                return None
            
            # Simple breadth-first search for shortest path
            from collections import deque
            
            queue = deque([(source_id, [source_id])])
            visited = {source_id}
            
            while queue:
                current_node, path = queue.popleft()
                
                if current_node == destination_id:
                    route_time = time.time() - start_time
                    self.route_discovery_time.observe(route_time)
                    return path
                
                for neighbor in self.network_topology[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return None  # No route found
            
        except Exception as e:
            logging.error(f"Route discovery failed: {e}")
            return None
    
    def relay_message(self, route: list, message_data: Dict[str, Any]) -> bool:
        """
        Relay message through multiple hops in mesh network
        
        Args:
            route: List of node IDs representing the routing path
            message_data: Message to relay
            
        Returns:
            Success status
        """
        try:
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                
                if current_node in self.nodes:
                    drone = self.nodes[current_node]
                    success = drone.send_telemetry(next_node, {
                        'relay_message': message_data,
                        'final_destination': route[-1],
                        'hop_count': i + 1
                    })
                    
                    if not success:
                        logging.error(f"Relay failed at hop {i}: {current_node} -> {next_node}")
                        return False
            
            self.network_messages.labels(type='relay').inc()
            logging.info(f"Message relayed successfully through {len(route)} hops")
            return True
            
        except Exception as e:
            logging.error(f"Message relay failed: {e}")
            return False
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get overall network status and metrics"""
        total_nodes = len(self.nodes)
        total_connections = sum(len(neighbors) for neighbors in self.network_topology.values()) // 2
        
        # Calculate network connectivity
        connected_components = self._find_connected_components()
        
        return {
            'network_id': self.network_id,
            'total_nodes': total_nodes,
            'total_connections': total_connections,
            'connected_components': len(connected_components),
            'largest_component_size': max(len(component) for component in connected_components) if connected_components else 0,
            'network_density': total_connections / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        }
    
    def _find_connected_components(self) -> list:
        """Find connected components in the network topology"""
        visited = set()
        components = []
        
        for node in self.network_topology:
            if node not in visited:
                component = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        stack.extend(self.network_topology[current] - visited)
                
                components.append(component)
        
        return components


# Integration test for complete system
async def test_drone_network():
    """Test complete drone network with mesh communications"""
    
    logging.info("=== Testing Drone Network Communications ===")
    
    # Create network manager
    network = DroneNetworkManager("HALE_NETWORK_001")
    
    # Add drones to network
    drone_ids = ["DRONE_001", "DRONE_002", "DRONE_003", "GROUND_001"]
    for drone_id in drone_ids:
        network.add_drone(drone_id)
    
    # Establish mesh connections (star topology with DRONE_001 as hub)
    connections = [
        ("DRONE_001", "DRONE_002"),
        ("DRONE_001", "DRONE_003"), 
        ("DRONE_001", "GROUND_001"),
        ("DRONE_002", "DRONE_003")  # Direct connection for redundancy
    ]
    
    network.establish_mesh_connections(connections)
    
    # Test routing
    route = network.find_route("DRONE_002", "GROUND_001")
    if route:
        logging.info(f"Route found from DRONE_002 to GROUND_001: {' -> '.join(route)}")
        
        # Test message relay
        test_message = {
            'message_type': 'status_report',
            'timestamp': time.time(),
            'battery_level': 85,
            'altitude': 65000,
            'coordinates': {'lat': 40.7128, 'lon': -74.0060}
        }
        
        success = network.relay_message(route, test_message)
        if success:
            logging.info("Message relay test successful")
        else:
            logging.error("Message relay test failed")
    else:
        logging.error("No route found between DRONE_002 and GROUND_001")
    
    # Display network status
    status = network.get_network_status()
    logging.info(f"Network status: {status}")
    
    logging.info("=== Drone Network Test Complete ===")


if __name__ == "__main__":
    # Run both tests
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "network":
        asyncio.run(test_drone_network())
    else:
        # Run PQC handshake test by default
        asyncio.run(test_pqc_handshake())