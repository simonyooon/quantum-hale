"""
NS-3 Network Simulator Wrapper

This module provides a Python interface to the NS-3 network simulator
for modeling HALE drone communication networks.
"""

import os
import sys
import logging
import subprocess
import tempfile
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import ns3
    NS3_AVAILABLE = True
except ImportError:
    logging.warning("NS-3 Python bindings not available, using mock implementation")
    NS3_AVAILABLE = False


class NetworkType(Enum):
    """Types of network topologies"""
    STAR = "star"
    MESH = "mesh"
    RING = "ring"
    TREE = "tree"
    CUSTOM = "custom"


@dataclass
class NodeConfig:
    """Configuration for a network node"""
    node_id: str
    node_type: str  # "drone", "ground_station", "relay"
    position: Tuple[float, float, float]  # x, y, z in meters
    antenna_gain: float = 0.0  # dB
    transmission_power: float = 20.0  # dBm
    frequency: float = 2.4e9  # Hz
    data_rate: float = 1e6  # bits per second


@dataclass
class LinkConfig:
    """Configuration for a network link"""
    source_id: str
    destination_id: str
    link_type: str  # "point_to_point", "wireless", "satellite"
    bandwidth: float = 1e6  # bits per second
    delay: float = 0.001  # seconds
    loss_rate: float = 0.0  # probability of packet loss


@dataclass
class SimulationResult:
    """Results from network simulation"""
    throughput: Dict[str, float]  # node_id -> throughput in bps
    latency: Dict[str, float]  # node_id -> average latency in seconds
    packet_loss: Dict[str, float]  # node_id -> packet loss rate
    connectivity: Dict[str, List[str]]  # node_id -> list of connected nodes
    energy_consumption: Dict[str, float]  # node_id -> energy in Joules


class NS3Wrapper:
    """
    Python wrapper for NS-3 network simulator
    """
    
    def __init__(self, ns3_path: Optional[str] = None):
        self.ns3_path = ns3_path or "/opt/ns-3-dev"
        self.ns3_available = NS3_AVAILABLE and os.path.exists(self.ns3_path)
        
        if not self.ns3_available:
            logging.warning("NS-3 not available, using mock simulation")
        
        self.nodes: Dict[str, NodeConfig] = {}
        self.links: List[LinkConfig] = []
        self.simulation_time = 100.0  # seconds
        
        logging.info(f"NS-3 Wrapper initialized (available: {self.ns3_available})")
    
    def initialize(self) -> bool:
        """Initialize the network simulation"""
        try:
            logging.info("Initializing NS-3 network simulation")
            # Add default nodes if none exist
            if not self.nodes:
                # Add a default ground station
                self.add_node(NodeConfig(
                    node_id="GROUND_001",
                    node_type="ground_station",
                    position=(0.0, 0.0, 100.0),
                    transmission_power=30.0,
                    data_rate=10e6
                ))
                
                # Add a default drone
                self.add_node(NodeConfig(
                    node_id="DRONE_001", 
                    node_type="drone",
                    position=(0.0, 0.0, 20000.0),
                    transmission_power=20.0,
                    data_rate=1e6
                ))
                
                # Create default mesh topology
                self.create_topology(NetworkType.MESH)
            
            logging.info(f"Network simulation initialized with {len(self.nodes)} nodes and {len(self.links)} links")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize network simulation: {e}")
            return False
    
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
    
    def create_topology(self, topology_type: NetworkType, **kwargs):
        """Create a network topology"""
        if topology_type == NetworkType.STAR:
            self._create_star_topology(**kwargs)
        elif topology_type == NetworkType.MESH:
            self._create_mesh_topology(**kwargs)
        elif topology_type == NetworkType.RING:
            self._create_ring_topology(**kwargs)
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")
    
    def _create_star_topology(self, center_node_id: str, radius: float = 1000.0):
        """Create a star topology with center node and surrounding nodes"""
        if center_node_id not in self.nodes:
            raise ValueError(f"Center node not found: {center_node_id}")
        
        center_node = self.nodes[center_node_id]
        center_pos = center_node.position
        
        # Create links from center to all other nodes
        for node_id, node_config in self.nodes.items():
            if node_id != center_node_id:
                # Calculate distance
                dx = node_config.position[0] - center_pos[0]
                dy = node_config.position[1] - center_pos[1]
                dz = node_config.position[2] - center_pos[2]
                distance = (dx**2 + dy**2 + dz**2)**0.5
                
                # Add bidirectional links
                self.add_link(LinkConfig(
                    source_id=center_node_id,
                    destination_id=node_id,
                    link_type="wireless",
                    bandwidth=min(center_node.data_rate, node_config.data_rate),
                    delay=distance / 3e8,  # Speed of light
                    loss_rate=0.01 if distance < radius else 0.1
                ))
                
                self.add_link(LinkConfig(
                    source_id=node_id,
                    destination_id=center_node_id,
                    link_type="wireless",
                    bandwidth=min(center_node.data_rate, node_config.data_rate),
                    delay=distance / 3e8,
                    loss_rate=0.01 if distance < radius else 0.1
                ))
    
    def _create_mesh_topology(self, max_distance: float = 2000.0):
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
                    self.add_link(LinkConfig(
                        source_id=node1_id,
                        destination_id=node2_id,
                        link_type="wireless",
                        bandwidth=min(node1.data_rate, node2.data_rate),
                        delay=distance / 3e8,
                        loss_rate=0.01 if distance < max_distance/2 else 0.05
                    ))
    
    def _create_ring_topology(self):
        """Create a ring topology connecting nodes in sequence"""
        node_ids = list(self.nodes.keys())
        
        if len(node_ids) < 3:
            raise ValueError("Ring topology requires at least 3 nodes")
        
        for i in range(len(node_ids)):
            node1_id = node_ids[i]
            node2_id = node_ids[(i + 1) % len(node_ids)]
            
            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]
            
            # Calculate distance
            dx = node2.position[0] - node1.position[0]
            dy = node2.position[1] - node1.position[1]
            dz = node2.position[2] - node1.position[2]
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            self.add_link(LinkConfig(
                source_id=node1_id,
                destination_id=node2_id,
                link_type="wireless",
                bandwidth=min(node1.data_rate, node2.data_rate),
                delay=distance / 3e8,
                loss_rate=0.01
            ))
    
    def run_simulation(self, duration: Optional[float] = None) -> SimulationResult:
        """
        Run the network simulation
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            Simulation results
        """
        if duration is not None:
            self.simulation_time = duration
        
        if self.ns3_available:
            return self._run_ns3_simulation()
        else:
            return self._run_mock_simulation()
    
    def _run_ns3_simulation(self) -> SimulationResult:
        """Run simulation using actual NS-3"""
        try:
            # Create NS-3 script
            script_content = self._generate_ns3_script()
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cc', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            # Run NS-3 simulation
            cmd = [os.path.join(self.ns3_path, "ns3"), "run", script_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up
            os.unlink(script_path)
            
            if result.returncode != 0:
                logging.error(f"NS-3 simulation failed: {result.stderr}")
                return self._run_mock_simulation()
            
            # Parse results (simplified)
            return self._parse_ns3_results(result.stdout)
            
        except Exception as e:
            logging.error(f"NS-3 simulation error: {e}")
            return self._run_mock_simulation()
    
    def _run_mock_simulation(self) -> SimulationResult:
        """Run mock simulation when NS-3 is not available"""
        import random
        import math
        
        # Initialize results
        throughput = {}
        latency = {}
        packet_loss = {}
        connectivity = {}
        energy_consumption = {}
        
        # Calculate results for each node
        for node_id, node_config in self.nodes.items():
            # Find connected nodes
            connected = []
            total_bandwidth = 0
            total_delay = 0
            total_loss = 0
            link_count = 0
            
            for link in self.links:
                if link.source_id == node_id:
                    connected.append(link.destination_id)
                    total_bandwidth += link.bandwidth
                    total_delay += link.delay
                    total_loss += link.loss_rate
                    link_count += 1
            
            connectivity[node_id] = connected
            
            # Calculate metrics
            if link_count > 0:
                throughput[node_id] = total_bandwidth / link_count
                latency[node_id] = total_delay / link_count
                packet_loss[node_id] = total_loss / link_count
            else:
                throughput[node_id] = 0.0
                latency[node_id] = float('inf')
                packet_loss[node_id] = 1.0
            
            # Calculate energy consumption (simplified)
            power_consumption = node_config.transmission_power / 1000  # Convert to Watts
            energy_consumption[node_id] = power_consumption * self.simulation_time
        
        return SimulationResult(
            throughput=throughput,
            latency=latency,
            packet_loss=packet_loss,
            connectivity=connectivity,
            energy_consumption=energy_consumption
        )
    
    def _generate_ns3_script(self) -> str:
        """Generate NS-3 C++ script"""
        script = """
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("HaleDroneNetwork");

int main (int argc, char *argv[])
{
  LogComponentEnable ("HaleDroneNetwork", LOG_LEVEL_INFO);
  
  // Create nodes
  NodeContainer nodes;
"""
        
        # Add nodes
        for node_id, node_config in self.nodes.items():
            script += f"  nodes.Create (1); // {node_id}\n"
        
        script += """
  // Create WiFi
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());
  
  WifiMacHelper mac;
  mac.SetType ("ns3::AdhocWifiMac");
  
  WifiHelper wifi;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue ("OfdmRate6Mbps"),
                               "ControlMode", StringValue ("OfdmRate6Mbps"));
  
  NetDeviceContainer devices = wifi.Install (phy, mac, nodes);
  
  // Set up mobility
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
"""
        
        # Add positions
        for node_config in self.nodes.values():
            x, y, z = node_config.position
            script += f"  positionAlloc->Add (Vector ({x}, {y}, {z}));\n"
        
        script += """
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);
  
  // Install Internet stack
  InternetStackHelper internet;
  internet.Install (nodes);
  
  // Assign IP addresses
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (devices);
  
  // Create applications
  uint16_t port = 9;
  OnOffHelper onoff ("ns3::UdpSocketFactory", Address ());
  onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
  
  ApplicationContainer apps;
"""
        
        # Add applications
        for link in self.links:
            script += f"""
  // {link.source_id} -> {link.destination_id}
  AddressValue remoteAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  onoff.SetAttribute ("Remote", remoteAddress);
  apps = onoff.Install (nodes.Get (0)); // Simplified - use first node
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds ({self.simulation_time}));
"""
        
        script += f"""
  // Enable flow monitor
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();
  
  Simulator::Stop (Seconds ({self.simulation_time}));
  Simulator::Run ();
  
  // Print results
  monitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();
  
  for (auto iter = stats.begin (); iter != stats.end (); ++iter)
  {{
    Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (iter->first);
    std::cout << "Flow " << iter->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\\n";
    std::cout << "  Tx Packets: " << iter->second.txPackets << "\\n";
    std::cout << "  Rx Packets: " << iter->second.rxPackets << "\\n";
    std::cout << "  Throughput: " << iter->second.rxBytes * 8.0 / {self.simulation_time} / 1000 / 1000 << " Mbps\\n";
    std::cout << "  Delay: " << iter->second.delaySum.GetSeconds () / iter->second.rxPackets << " s\\n";
  }}
  
  Simulator::Destroy ();
  return 0;
}}
"""
        return script
    
    def _parse_ns3_results(self, output: str) -> SimulationResult:
        """Parse NS-3 simulation output"""
        # Simplified parsing - in practice, this would be more sophisticated
        throughput = {}
        latency = {}
        packet_loss = {}
        connectivity = {}
        energy_consumption = {}
        
        # Extract results from output
        lines = output.split('\n')
        for line in lines:
            if "Throughput:" in line:
                # Parse throughput information
                pass
            elif "Delay:" in line:
                # Parse delay information
                pass
        
        # Fallback to mock results if parsing fails
        return self._run_mock_simulation()
    
    def get_network_graph(self) -> Dict[str, Any]:
        """Get network topology as a graph representation"""
        return {
            "nodes": [{"id": node_id, **node_config.__dict__} 
                     for node_id, node_config in self.nodes.items()],
            "links": [link_config.__dict__ for link_config in self.links],
            "simulation_time": self.simulation_time
        }
    
    def save_configuration(self, filename: str):
        """Save network configuration to file"""
        config = {
            "nodes": {node_id: node_config.__dict__ 
                     for node_id, node_config in self.nodes.items()},
            "links": [link_config.__dict__ for link_config in self.links],
            "simulation_time": self.simulation_time
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Configuration saved to {filename}")
    
    def load_configuration(self, filename: str):
        """Load network configuration from file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Load nodes
            for node_data in config.get('nodes', []):
                node = NodeConfig(**node_data)
                self.add_node(node)
            
            # Load links
            for link_data in config.get('links', []):
                link = LinkConfig(**link_data)
                self.add_link(link)
                
            logging.info(f"Configuration loaded from {filename}")
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
    
    def step(self, simulation_time: float):
        """Update network simulation for current time step"""
        # Update network state based on time
        for node_id, node in self.nodes.items():
            # For now, just log that nodes are active
            if simulation_time % 10 < 1:  # Log every 10 seconds
                logging.debug(f"Node {node_id} active at time {simulation_time}")
        
        # Update link states
        for link in self.links:
            # Simulate link performance changes over time
            if simulation_time % 5 < 1:  # Log every 5 seconds
                logging.debug(f"Link {link.source_id} -> {link.destination_id} active")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status and statistics"""
        return {
            'num_nodes': len(self.nodes),
            'num_links': len(self.links),
            'nodes': list(self.nodes.keys()),
            'links': [f"{link.source_id} -> {link.destination_id}" for link in self.links],
            'topology_type': 'mesh',  # Default for now
            'connectivity': self._calculate_connectivity(),
            'performance': {
                'average_latency': 0.001,  # Mock values
                'packet_loss_rate': 0.01,
                'throughput': 1e6
            }
        }
    
    def _calculate_connectivity(self) -> Dict[str, List[str]]:
        """Calculate network connectivity matrix"""
        connectivity = {}
        for node_id in self.nodes.keys():
            connected_nodes = []
            for link in self.links:
                if link.source_id == node_id:
                    connected_nodes.append(link.destination_id)
                elif link.destination_id == node_id:
                    connected_nodes.append(link.source_id)
            connectivity[node_id] = connected_nodes 