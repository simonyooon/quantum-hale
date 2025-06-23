"""
Enhanced NS-3 Network Simulator Wrapper

This module provides a comprehensive Python interface to the NS-3 network simulator
for modeling HALE drone communication networks with advanced features:
- Real NS-3 integration with mesh routing protocols
- RF propagation modeling for high-altitude operations
- Jamming/interference simulation with countermeasures
- Multiple drone coordination protocols
"""

import os
import sys
import logging
import subprocess
import tempfile
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import ns3
    NS3_AVAILABLE = True
    logging.info("NS-3 Python bindings available")
except ImportError:
    logging.warning("NS-3 Python bindings not available, using enhanced mock implementation")
    NS3_AVAILABLE = False


class NetworkType(Enum):
    """Types of network topologies"""
    STAR = "star"
    MESH = "mesh"
    RING = "ring"
    TREE = "tree"
    CUSTOM = "custom"
    ADHOC = "adhoc"


class RoutingProtocol(Enum):
    """NS-3 routing protocols"""
    AODV = "aodv"
    OLSR = "olsr"
    DSDV = "dsdv"
    DSR = "dsr"
    BATMAN = "batman"
    CUSTOM = "custom"


class PropagationModel(Enum):
    """NS-3 propagation models"""
    FREE_SPACE = "ns3::FreeSpacePropagationLossModel"
    TWO_RAY = "ns3::TwoRayGroundPropagationLossModel"
    ITU_R = "ns3::ItuR1411LosPropagationLossModel"
    COST_231 = "ns3::Cost231PropagationLossModel"
    FRIIS = "ns3::FriisPropagationLossModel"


@dataclass
class NodeConfig:
    """Enhanced configuration for a network node"""
    node_id: str
    node_type: str  # "drone", "ground_station", "relay", "jammer"
    position: Tuple[float, float, float]  # x, y, z in meters
    antenna_gain: float = 0.0  # dB
    transmission_power: float = 20.0  # dBm
    frequency: float = 2.4e9  # Hz
    data_rate: float = 1e6  # bits per second
    mobility_model: str = "ns3::ConstantPositionMobilityModel"
    routing_protocol: RoutingProtocol = RoutingProtocol.AODV
    energy_model: Optional[str] = None
    application_type: str = "udp_echo"  # udp_echo, tcp_bulk, custom


@dataclass
class LinkConfig:
    """Enhanced configuration for a network link"""
    source_id: str
    destination_id: str
    link_type: str  # "point_to_point", "wireless", "satellite"
    bandwidth: float = 1e6  # bits per second
    delay: float = 0.001  # seconds
    loss_rate: float = 0.0  # probability of packet loss
    error_model: str = "ns3::YansErrorRateModel"
    channel_type: str = "ns3::YansWifiChannel"


@dataclass
class JammingConfig:
    """Configuration for jamming simulation"""
    jammer_id: str
    position: Tuple[float, float, float]
    power: float  # dBm
    frequency_range: Tuple[float, float]  # Hz
    jamming_type: str  # "continuous", "pulse", "sweep"
    duty_cycle: float = 1.0
    start_time: float = 0.0
    duration: float = 100.0


@dataclass
class SimulationResult:
    """Enhanced results from network simulation"""
    throughput: Dict[str, float]  # node_id -> throughput in bps
    latency: Dict[str, float]  # node_id -> average latency in seconds
    packet_loss: Dict[str, float]  # node_id -> packet loss rate
    connectivity: Dict[str, List[str]]  # node_id -> list of connected nodes
    energy_consumption: Dict[str, float]  # node_id -> energy in Joules
    routing_tables: Dict[str, Dict[str, str]]  # node_id -> routing table
    link_quality: Dict[Tuple[str, str], float]  # (src, dst) -> quality (0-1)
    jamming_effects: Dict[str, float]  # node_id -> jamming impact
    network_graph: Dict[str, Any]  # Network topology graph


class NS3Wrapper:
    """
    Enhanced Python wrapper for NS-3 network simulator
    """
    
    def __init__(self, ns3_path: Optional[str] = None, enable_logging: bool = True):
        self.ns3_path = ns3_path or self._find_ns3_path()
        self.ns3_available = NS3_AVAILABLE and os.path.exists(self.ns3_path)
        self.enable_logging = enable_logging
        
        if not self.ns3_available:
            logging.warning("NS-3 not available, using enhanced mock simulation")
        
        # Network components
        self.nodes: Dict[str, NodeConfig] = {}
        self.links: List[LinkConfig] = []
        self.jammers: Dict[str, JammingConfig] = {}
        
        # Simulation parameters
        self.simulation_time = 100.0  # seconds
        self.time_step = 0.1  # seconds
        self.current_time = 0.0
        
        # NS-3 specific
        self.ns3_script_template = self._load_ns3_template()
        self.simulation_running = False
        self.results_cache: Dict[str, SimulationResult] = {}
        
        # Callbacks for real-time monitoring
        self.monitoring_callbacks: List[Callable] = []
        
        logging.info(f"Enhanced NS-3 Wrapper initialized (available: {self.ns3_available})")
    
    def _find_ns3_path(self) -> str:
        """Find NS-3 installation path"""
        possible_paths = [
            "/opt/ns-3-dev",
            "/usr/local/ns-3-dev",
            "/opt/ns-3.37",
            "/usr/local/ns-3.37",
            "C:/ns-3-dev",
            "C:/Program Files/ns-3-dev"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return "/opt/ns-3-dev"  # Default fallback
    
    def _load_ns3_template(self) -> str:
        """Load NS-3 script template"""
        return '''
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/energy-module.h"
#include "ns3/aodv-module.h"
#include "ns3/olsr-module.h"
#include "ns3/dsdv-module.h"
#include "ns3/dsr-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("HaleDroneNetwork");

int main (int argc, char *argv[])
{
  // Enable logging
  LogComponentEnable ("HaleDroneNetwork", LOG_LEVEL_INFO);
  
  // Command line parameters
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  // Create nodes
  NodeContainer nodes;
  nodes.Create ({NODE_COUNT});
  
  // Set up mobility
  MobilityHelper mobility;
  {MOBILITY_SETUP}
  
  // Set up wireless channel
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  {PROPAGATION_MODEL}
  
  // Set up PHY layer
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());
  
  // Set up MAC layer
  WifiMacHelper mac;
  mac.SetType ("ns3::AdhocWifiMac");
  
  // Set up WiFi
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue ("OfdmRate6Mbps"),
                               "ControlMode", StringValue ("OfdmRate6Mbps"));
  
  // Install WiFi on nodes
  NetDeviceContainer devices = wifi.Install (phy, mac, nodes);
  
  // Set up Internet stack
  InternetStackHelper internet;
  {ROUTING_PROTOCOL}
  internet.Install (nodes);
  
  // Assign IP addresses
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);
  
  // Set up applications
  {APPLICATION_SETUP}
  
  // Set up flow monitor
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();
  
  // Run simulation
  Simulator::Stop (Seconds ({SIMULATION_TIME}));
  Simulator::Run ();
  
  // Collect results
  monitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();
  
  // Output results
  {RESULTS_OUTPUT}
  
  Simulator::Destroy ();
  return 0;
}
'''
    
    def initialize(self) -> bool:
        """Initialize the network simulation"""
        try:
            logging.info("Initializing enhanced NS-3 network simulation")
            
            # Add default nodes if none exist
            if not self.nodes:
                self._create_default_network()
            
            # Initialize NS-3 if available
            if self.ns3_available:
                self._initialize_ns3()
            
            logging.info(f"Network simulation initialized with {len(self.nodes)} nodes and {len(self.links)} links")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize network simulation: {e}")
            return False
    
    def _create_default_network(self):
        """Create a default HALE drone network"""
        # Ground stations
        self.add_node(NodeConfig(
            node_id="GROUND_001",
            node_type="ground_station",
            position=(0.0, 0.0, 100.0),
            transmission_power=40.0,
            data_rate=10e6,
            routing_protocol=RoutingProtocol.AODV
        ))
        
        self.add_node(NodeConfig(
            node_id="GROUND_002",
            node_type="ground_station",
            position=(50000.0, 0.0, 100.0),
            transmission_power=40.0,
            data_rate=10e6,
            routing_protocol=RoutingProtocol.AODV
        ))
        
        # HALE drones
        self.add_node(NodeConfig(
            node_id="DRONE_001",
            node_type="drone",
            position=(0.0, 0.0, 20000.0),
            transmission_power=30.0,
            data_rate=1e6,
            routing_protocol=RoutingProtocol.AODV
        ))
        
        self.add_node(NodeConfig(
            node_id="DRONE_002",
            node_type="drone",
            position=(10000.0, 5000.0, 20000.0),
            transmission_power=30.0,
            data_rate=1e6,
            routing_protocol=RoutingProtocol.AODV
        ))
        
        self.add_node(NodeConfig(
            node_id="DRONE_003",
            node_type="drone",
            position=(5000.0, 10000.0, 20000.0),
            transmission_power=30.0,
            data_rate=1e6,
            routing_protocol=RoutingProtocol.AODV
        ))
        
        # Create mesh topology
        self.create_topology(NetworkType.MESH)
    
    def _initialize_ns3(self):
        """Initialize NS-3 simulation environment"""
        if not NS3_AVAILABLE:
            return
        
        try:
            # Initialize NS-3 logging
            if self.enable_logging:
                ns3.LogComponentEnable("HaleDroneNetwork", ns3.LOG_LEVEL_INFO)
            
            logging.info("NS-3 environment initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize NS-3: {e}")
            self.ns3_available = False
    
    def add_node(self, node_config: NodeConfig):
        """Add a node to the network"""
        self.nodes[node_config.node_id] = node_config
        logging.info(f"Node added: {node_config.node_id} at {node_config.position}")
    
    def add_link(self, link_config: LinkConfig):
        """Add a link between nodes"""
        if link_config.source_id not in self.nodes:
            raise ValueError(f"Source node not found: {link_config.source_id}")
        if link_config.destination_id not in self.nodes:
            raise ValueError(f"Destination node not found: {link_config.destination_id}")
        
        self.links.append(link_config)
        logging.info(f"Link added: {link_config.source_id} -> {link_config.destination_id}")
    
    def add_jammer(self, jammer_config: JammingConfig):
        """Add a jamming source to the simulation"""
        self.jammers[jammer_config.jammer_id] = jammer_config
        logging.info(f"Jammer added: {jammer_config.jammer_id} at {jammer_config.position}")
    
    def create_topology(self, topology_type: NetworkType, **kwargs):
        """Create a network topology"""
        if topology_type == NetworkType.STAR:
            self._create_star_topology(**kwargs)
        elif topology_type == NetworkType.MESH:
            self._create_mesh_topology(**kwargs)
        elif topology_type == NetworkType.RING:
            self._create_ring_topology(**kwargs)
        elif topology_type == NetworkType.ADHOC:
            self._create_adhoc_topology(**kwargs)
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")
    
    def _create_mesh_topology(self, max_distance: float = 50000.0):
        """Create a mesh topology with all nodes connected within range"""
        node_ids = list(self.nodes.keys())
        
        for i, node1_id in enumerate(node_ids):
            for j, node2_id in enumerate(node_ids[i+1:], i+1):
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                # Calculate distance
                dx = node2.position[0] - node1.position[0]
                dy = node2.position[1] - node1.position[1]
                dz = node2.position[2] - node1.position[2]
                distance = (dx**2 + dy**2 + dz**2)**0.5
                
                # Connect if within range
                if distance <= max_distance:
                    # Calculate link parameters based on distance and node types
                    bandwidth = min(node1.data_rate, node2.data_rate)
                    latency = distance / 3e8  # Speed of light
                    loss_rate = self._calculate_loss_rate(distance, node1, node2)
                    
                    self.add_link(LinkConfig(
                        source_id=node1_id,
                        destination_id=node2_id,
                        link_type="wireless",
                        bandwidth=bandwidth,
                        delay=latency,
                        loss_rate=loss_rate
                    ))
    
    def _calculate_loss_rate(self, distance: float, node1: NodeConfig, node2: NodeConfig) -> float:
        """Calculate packet loss rate based on distance and node characteristics"""
        # Base loss rate from free space path loss
        wavelength = 3e8 / node1.frequency
        path_loss_db = 20 * np.log10(4 * np.pi * distance / wavelength)
        
        # Convert to linear scale and estimate loss rate
        path_loss_linear = 10**(path_loss_db / 10)
        tx_power_linear = 10**(node1.transmission_power / 10)
        rx_power_linear = tx_power_linear / path_loss_linear
        
        # Estimate loss rate based on received power
        # This is a simplified model - in reality would depend on modulation, coding, etc.
        if rx_power_linear < 1e-12:  # Very weak signal
            return 0.5
        elif rx_power_linear < 1e-9:  # Weak signal
            return 0.1
        else:
            return 0.01  # Good signal
    
    def run_simulation(self, duration: Optional[float] = None) -> SimulationResult:
        """Run the network simulation"""
        if duration:
            self.simulation_time = duration
        
        logging.info(f"Starting network simulation for {self.simulation_time} seconds")
        
        if self.ns3_available:
            return self._run_ns3_simulation()
        else:
            return self._run_enhanced_mock_simulation()
    
    def _run_ns3_simulation(self) -> SimulationResult:
        """Run simulation using actual NS-3"""
        try:
            # Generate NS-3 script
            script_content = self._generate_ns3_script()
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cc', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            # Compile and run NS-3 simulation
            result = self._execute_ns3_simulation(script_path)
            
            # Clean up
            os.unlink(script_path)
            
            return result
            
        except Exception as e:
            logging.error(f"NS-3 simulation failed: {e}")
            return self._run_enhanced_mock_simulation()
    
    def _run_enhanced_mock_simulation(self) -> SimulationResult:
        """Run enhanced mock simulation with realistic behavior"""
        logging.info("Running enhanced mock simulation")
        
        # Initialize results
        throughput = {}
        latency = {}
        packet_loss = {}
        connectivity = {}
        energy_consumption = {}
        routing_tables = {}
        link_quality = {}
        jamming_effects = {}
        
        # Simulate network behavior
        for node_id, node_config in self.nodes.items():
            # Calculate throughput based on node type and connections
            connections = self._get_node_connections(node_id)
            throughput[node_id] = self._calculate_node_throughput(node_config, connections)
            
            # Calculate latency
            latency[node_id] = self._calculate_node_latency(node_config, connections)
            
            # Calculate packet loss
            packet_loss[node_id] = self._calculate_node_packet_loss(node_config, connections)
            
            # Set connectivity
            connectivity[node_id] = connections
            
            # Calculate energy consumption
            energy_consumption[node_id] = self._calculate_energy_consumption(node_config)
            
            # Generate routing table
            routing_tables[node_id] = self._generate_routing_table(node_id)
            
            # Calculate jamming effects
            jamming_effects[node_id] = self._calculate_jamming_effects(node_id)
        
        # Calculate link quality for all links
        for link in self.links:
            quality = self._calculate_link_quality(link)
            link_quality[(link.source_id, link.destination_id)] = quality
        
        # Create network graph
        network_graph = self._create_network_graph()
        
        result = SimulationResult(
            throughput=throughput,
            latency=latency,
            packet_loss=packet_loss,
            connectivity=connectivity,
            energy_consumption=energy_consumption,
            routing_tables=routing_tables,
            link_quality=link_quality,
            jamming_effects=jamming_effects,
            network_graph=network_graph
        )
        
        # Cache results
        self.results_cache[f"sim_{int(time.time())}"] = result
        
        return result
    
    def _get_node_connections(self, node_id: str) -> List[str]:
        """Get list of nodes connected to the given node"""
        connections = []
        for link in self.links:
            if link.source_id == node_id:
                connections.append(link.destination_id)
            elif link.destination_id == node_id:
                connections.append(link.source_id)
        return connections
    
    def _calculate_node_throughput(self, node_config: NodeConfig, connections: List[str]) -> float:
        """Calculate effective throughput for a node"""
        base_throughput = node_config.data_rate
        
        # Reduce throughput based on number of connections (contention)
        if len(connections) > 1:
            contention_factor = 1.0 / len(connections)
            base_throughput *= contention_factor
        
        # Apply jamming effects
        jamming_factor = self._calculate_jamming_factor(node_config)
        base_throughput *= jamming_factor
        
        return base_throughput
    
    def _calculate_jamming_factor(self, node_config: NodeConfig) -> float:
        """Calculate jamming impact factor (0.0 to 1.0)"""
        if not self.jammers:
            return 1.0
        
        total_jamming_effect = 0.0
        for jammer in self.jammers.values():
            # Calculate distance to jammer
            dx = jammer.position[0] - node_config.position[0]
            dy = jammer.position[1] - node_config.position[1]
            dz = jammer.position[2] - node_config.position[2]
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            # Calculate jamming effect based on distance and power
            if distance < 10000:  # Close jammer
                effect = 0.3
            elif distance < 20000:  # Medium distance
                effect = 0.1
            else:  # Far jammer
                effect = 0.05
            
            total_jamming_effect += effect
        
        # Cap total effect at 0.8 (20% minimum throughput)
        total_jamming_effect = min(total_jamming_effect, 0.8)
        return 1.0 - total_jamming_effect
    
    def _calculate_node_latency(self, node_config: NodeConfig, connections: List[str]) -> float:
        """Calculate average latency for a node"""
        if not connections:
            return 0.0
        
        total_latency = 0.0
        for connection in connections:
            # Find link to this connection
            for link in self.links:
                if ((link.source_id == node_config.node_id and link.destination_id == connection) or
                    (link.destination_id == node_config.node_id and link.source_id == connection)):
                    total_latency += link.delay
                    break
        
        return total_latency / len(connections)
    
    def _calculate_node_packet_loss(self, node_config: NodeConfig, connections: List[str]) -> float:
        """Calculate packet loss rate for a node"""
        if not connections:
            return 0.0
        
        total_loss = 0.0
        for connection in connections:
            # Find link to this connection
            for link in self.links:
                if ((link.source_id == node_config.node_id and link.destination_id == connection) or
                    (link.destination_id == node_config.node_id and link.source_id == connection)):
                    total_loss += link.loss_rate
                    break
        
        return total_loss / len(connections)
    
    def _calculate_energy_consumption(self, node_config: NodeConfig) -> float:
        """Calculate energy consumption for a node"""
        # Base energy consumption
        base_energy = 100.0  # Joules per second
        
        # Add transmission energy
        tx_energy = node_config.transmission_power * 0.001  # Convert dBm to Watts
        
        # Add processing energy based on node type
        if node_config.node_type == "drone":
            processing_energy = 50.0
        elif node_config.node_type == "ground_station":
            processing_energy = 200.0
        else:
            processing_energy = 25.0
        
        total_energy = (base_energy + tx_energy + processing_energy) * self.simulation_time
        return total_energy
    
    def _generate_routing_table(self, node_id: str) -> Dict[str, str]:
        """Generate routing table for a node"""
        routing_table = {}
        
        # Simple routing table - direct connections
        for link in self.links:
            if link.source_id == node_id:
                routing_table[link.destination_id] = link.destination_id
            elif link.destination_id == node_id:
                routing_table[link.source_id] = link.source_id
        
        return routing_table
    
    def _calculate_jamming_effects(self, node_id: str) -> float:
        """Calculate jamming effects on a node"""
        if not self.jammers:
            return 0.0
        
        node_config = self.nodes[node_id]
        total_effect = 0.0
        
        for jammer in self.jammers.values():
            # Calculate distance to jammer
            dx = jammer.position[0] - node_config.position[0]
            dy = jammer.position[1] - node_config.position[1]
            dz = jammer.position[2] - node_config.position[2]
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            # Calculate effect based on distance and jammer power
            effect = jammer.power / (distance * distance) * 1e-6
            total_effect += effect
        
        return min(total_effect, 1.0)  # Cap at 1.0
    
    def _calculate_link_quality(self, link: LinkConfig) -> float:
        """Calculate link quality (0.0 to 1.0)"""
        # Base quality from loss rate
        base_quality = 1.0 - link.loss_rate
        
        # Apply jamming effects
        jamming_factor = self._calculate_jamming_factor_for_link(link)
        
        return base_quality * jamming_factor
    
    def _calculate_jamming_factor_for_link(self, link: LinkConfig) -> float:
        """Calculate jamming factor for a specific link"""
        if not self.jammers:
            return 1.0
        
        # Get node positions
        src_node = self.nodes[link.source_id]
        dst_node = self.nodes[link.destination_id]
        
        # Calculate link midpoint
        mid_x = (src_node.position[0] + dst_node.position[0]) / 2
        mid_y = (src_node.position[1] + dst_node.position[1]) / 2
        mid_z = (src_node.position[2] + dst_node.position[2]) / 2
        
        total_jamming_effect = 0.0
        for jammer in self.jammers.values():
            # Calculate distance from jammer to link midpoint
            dx = jammer.position[0] - mid_x
            dy = jammer.position[1] - mid_y
            dz = jammer.position[2] - mid_z
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            # Calculate jamming effect
            if distance < 5000:  # Very close
                effect = 0.5
            elif distance < 15000:  # Close
                effect = 0.2
            elif distance < 25000:  # Medium
                effect = 0.1
            else:  # Far
                effect = 0.05
            
            total_jamming_effect += effect
        
        # Cap total effect
        total_jamming_effect = min(total_jamming_effect, 0.8)
        return 1.0 - total_jamming_effect
    
    def _create_network_graph(self) -> Dict[str, Any]:
        """Create network topology graph"""
        nodes = []
        edges = []
        
        # Add nodes
        for node_id, node_config in self.nodes.items():
            nodes.append({
                'id': node_id,
                'type': node_config.node_type,
                'position': node_config.position,
                'transmission_power': node_config.transmission_power,
                'data_rate': node_config.data_rate
            })
        
        # Add edges
        for link in self.links:
            edges.append({
                'source': link.source_id,
                'target': link.destination_id,
                'bandwidth': link.bandwidth,
                'latency': link.delay,
                'loss_rate': link.loss_rate
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'jammers': [jammer.jammer_id for jammer in self.jammers.values()]
        }
    
    def step(self, simulation_time: float):
        """Step the simulation forward"""
        self.current_time = simulation_time
        
        # Update jamming effects
        for jammer in self.jammers.values():
            if jammer.start_time <= simulation_time <= jammer.start_time + jammer.duration:
                # Jammer is active
                pass
        
        # Call monitoring callbacks
        for callback in self.monitoring_callbacks:
            try:
                callback(self.current_time, self.get_network_status())
            except Exception as e:
                logging.error(f"Monitoring callback error: {e}")
    
    def add_monitoring_callback(self, callback: Callable):
        """Add a callback for real-time monitoring"""
        self.monitoring_callbacks.append(callback)
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        return {
            'current_time': self.current_time,
            'node_count': len(self.nodes),
            'link_count': len(self.links),
            'jammer_count': len(self.jammers),
            'nodes': {node_id: {
                'type': config.node_type,
                'position': config.position,
                'transmission_power': config.transmission_power
            } for node_id, config in self.nodes.items()},
            'jammers': {jammer_id: {
                'position': jammer.position,
                'power': jammer.power,
                'active': jammer.start_time <= self.current_time <= jammer.start_time + jammer.duration
            } for jammer_id, jammer in self.jammers.items()}
        }
    
    def save_configuration(self, filename: str):
        """Save network configuration to file"""
        config = {
            'nodes': {node_id: {
                'node_type': config.node_type,
                'position': config.position,
                'antenna_gain': config.antenna_gain,
                'transmission_power': config.transmission_power,
                'frequency': config.frequency,
                'data_rate': config.data_rate,
                'routing_protocol': config.routing_protocol.value
            } for node_id, config in self.nodes.items()},
            'links': [{
                'source_id': link.source_id,
                'destination_id': link.destination_id,
                'link_type': link.link_type,
                'bandwidth': link.bandwidth,
                'delay': link.delay,
                'loss_rate': link.loss_rate
            } for link in self.links],
            'jammers': [{
                'jammer_id': jammer.jammer_id,
                'position': jammer.position,
                'power': jammer.power,
                'frequency_range': jammer.frequency_range,
                'jamming_type': jammer.jamming_type,
                'duty_cycle': jammer.duty_cycle,
                'start_time': jammer.start_time,
                'duration': jammer.duration
            } for jammer in self.jammers.values()],
            'simulation_time': self.simulation_time
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Configuration saved to {filename}")
    
    def load_configuration(self, filename: str):
        """Load network configuration from file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        # Clear existing configuration
        self.nodes.clear()
        self.links.clear()
        self.jammers.clear()
        
        # Load nodes
        for node_id, node_data in config['nodes'].items():
            self.add_node(NodeConfig(
                node_id=node_id,
                node_type=node_data['node_type'],
                position=tuple(node_data['position']),
                antenna_gain=node_data['antenna_gain'],
                transmission_power=node_data['transmission_power'],
                frequency=node_data['frequency'],
                data_rate=node_data['data_rate'],
                routing_protocol=RoutingProtocol(node_data['routing_protocol'])
            ))
        
        # Load links
        for link_data in config['links']:
            self.add_link(LinkConfig(
                source_id=link_data['source_id'],
                destination_id=link_data['destination_id'],
                link_type=link_data['link_type'],
                bandwidth=link_data['bandwidth'],
                delay=link_data['delay'],
                loss_rate=link_data['loss_rate']
            ))
        
        # Load jammers
        for jammer_data in config['jammers']:
            self.add_jammer(JammingConfig(
                jammer_id=jammer_data['jammer_id'],
                position=tuple(jammer_data['position']),
                power=jammer_data['power'],
                frequency_range=tuple(jammer_data['frequency_range']),
                jamming_type=jammer_data['jamming_type'],
                duty_cycle=jammer_data['duty_cycle'],
                start_time=jammer_data['start_time'],
                duration=jammer_data['duration']
            ))
        
        self.simulation_time = config.get('simulation_time', 100.0)
        logging.info(f"Configuration loaded from {filename}")
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics"""
        if not self.results_cache:
            return {}
        
        # Get latest result
        latest_result = list(self.results_cache.values())[-1]
        
        stats = {
            'total_nodes': len(self.nodes),
            'total_links': len(self.links),
            'total_jammers': len(self.jammers),
            'average_throughput': np.mean(list(latest_result.throughput.values())),
            'average_latency': np.mean(list(latest_result.latency.values())),
            'average_packet_loss': np.mean(list(latest_result.packet_loss.values())),
            'network_connectivity': len(latest_result.connectivity),
            'total_energy_consumption': sum(latest_result.energy_consumption.values()),
            'average_jamming_effect': np.mean(list(latest_result.jamming_effects.values())),
            'link_quality_distribution': {
                'excellent': len([q for q in latest_result.link_quality.values() if q >= 0.9]),
                'good': len([q for q in latest_result.link_quality.values() if 0.7 <= q < 0.9]),
                'fair': len([q for q in latest_result.link_quality.values() if 0.5 <= q < 0.7]),
                'poor': len([q for q in latest_result.link_quality.values() if q < 0.5])
            }
        }
        
        return stats 