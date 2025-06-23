"""
Enhanced Mesh Network Routing for HALE Drone Communications

This module provides advanced mesh network routing algorithms and protocols
for HALE drone communication networks with features:
- Advanced routing protocols (AODV, OLSR, DSDV, DSR, BATMAN)
- Multipath routing for reliability
- Load balancing and traffic engineering
- Multiple drone coordination protocols
- Adaptive routing based on link quality
"""

import math
import heapq
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np


class RoutingAlgorithm(Enum):
    """Types of routing algorithms"""
    DIJKSTRA = "dijkstra"
    AODV = "aodv"
    OLSR = "olsr"
    DSDV = "dsdv"
    DSR = "dsr"
    BATMAN = "batman"
    MULTIPATH = "multipath"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"


class CoordinationProtocol(Enum):
    """Types of drone coordination protocols"""
    LEADER_FOLLOWER = "leader_follower"
    CONSENSUS = "consensus"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    SWARM = "swarm"


@dataclass
class Route:
    """Enhanced route information"""
    source: str
    destination: str
    path: List[str]
    cost: float
    hops: int
    quality: float  # 0.0 to 1.0
    timestamp: float
    bandwidth: float  # Available bandwidth
    latency: float  # End-to-end latency
    reliability: float  # Route reliability
    energy_cost: float  # Energy consumption


@dataclass
class LinkState:
    """Enhanced state of a network link"""
    source: str
    destination: str
    bandwidth: float  # bits per second
    latency: float  # seconds
    loss_rate: float  # 0.0 to 1.0
    quality: float  # 0.0 to 1.0
    last_updated: float
    utilization: float  # Current utilization (0.0 to 1.0)
    interference: float  # Interference level
    energy_cost: float  # Energy cost per bit


@dataclass
class NodeState:
    """State of a network node"""
    node_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    energy_level: float = 100.0  # Percentage
    load: float = 0.0  # Current load (0.0 to 1.0)
    neighbors: List[str] = field(default_factory=list)
    routing_table: Dict[str, str] = field(default_factory=dict)
    last_updated: float = 0.0


@dataclass
class CoordinationMessage:
    """Message for drone coordination"""
    message_id: str
    source: str
    destination: str
    message_type: str  # "position", "status", "command", "route_update"
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 0  # 0=low, 1=normal, 2=high, 3=urgent


class EnhancedMeshRouting:
    """
    Enhanced mesh network routing for HALE drone communications
    """
    
    def __init__(self, algorithm: RoutingAlgorithm = RoutingAlgorithm.AODV,
                 coordination_protocol: CoordinationProtocol = CoordinationProtocol.LEADER_FOLLOWER):
        self.algorithm = algorithm
        self.coordination_protocol = coordination_protocol
        
        # Network state
        self.network_graph = nx.Graph()
        self.link_states: Dict[Tuple[str, str], LinkState] = {}
        self.node_states: Dict[str, NodeState] = {}
        self.routes: Dict[Tuple[str, str], Route] = {}
        
        # Coordination
        self.coordination_messages: List[CoordinationMessage] = []
        self.leader_node: Optional[str] = None
        self.coordination_callbacks: List[Callable] = []
        
        # Routing parameters
        self.route_timeout = 30.0  # seconds
        self.hello_interval = 2.0  # seconds
        self.max_hops = 10
        self.quality_threshold = 0.7
        
        # Load balancing
        self.load_balancing_enabled = True
        self.traffic_distribution: Dict[Tuple[str, str], float] = {}
        
        # Threading
        self.update_thread = None
        self.running = False
        
        logging.info(f"Enhanced Mesh Routing initialized with {algorithm.value} algorithm and {coordination_protocol.value} coordination")
    
    def start(self):
        """Start the routing system"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logging.info("Enhanced Mesh Routing started")
    
    def stop(self):
        """Stop the routing system"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logging.info("Enhanced Mesh Routing stopped")
    
    def _update_loop(self):
        """Main update loop for routing and coordination"""
        while self.running:
            try:
                current_time = time.time()
                
                # Update link states
                self._update_link_states(current_time)
                
                # Update routing tables
                self._update_routing_tables(current_time)
                
                # Process coordination messages
                self._process_coordination_messages(current_time)
                
                # Update load balancing
                if self.load_balancing_enabled:
                    self._update_load_balancing(current_time)
                
                # Call coordination callbacks
                for callback in self.coordination_callbacks:
                    try:
                        callback(current_time, self.get_network_state())
                    except Exception as e:
                        logging.error(f"Coordination callback error: {e}")
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logging.error(f"Error in routing update loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def add_node(self, node_id: str, position: Tuple[float, float, float], 
                node_type: str = "drone"):
        """Add a node to the mesh network"""
        self.network_graph.add_node(node_id)
        
        node_state = NodeState(
            node_id=node_id,
            position=position
        )
        self.node_states[node_id] = node_state
        
        logging.info(f"Node added to mesh: {node_id} at {position}")
    
    def add_link(self, source: str, destination: str, bandwidth: float,
                latency: float, loss_rate: float = 0.0, energy_cost: float = 0.0):
        """Add a link between nodes"""
        # Add to network graph
        self.network_graph.add_edge(source, destination, 
                                   weight=latency, bandwidth=bandwidth, 
                                   loss_rate=loss_rate, energy_cost=energy_cost)
        
        # Create link state
        link_state = LinkState(
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            latency=latency,
            loss_rate=loss_rate,
            quality=1.0 - loss_rate,
            last_updated=time.time(),
            utilization=0.0,
            interference=0.0,
            energy_cost=energy_cost
        )
        
        self.link_states[(source, destination)] = link_state
        self.link_states[(destination, source)] = link_state  # Bidirectional
        
        # Update node neighbors
        if source in self.node_states:
            if destination not in self.node_states[source].neighbors:
                self.node_states[source].neighbors.append(destination)
        
        if destination in self.node_states:
            if source not in self.node_states[destination].neighbors:
                self.node_states[destination].neighbors.append(source)
        
        logging.info(f"Link added: {source} <-> {destination} ({bandwidth/1e6:.1f} Mbps)")
    
    def update_link_quality(self, source: str, destination: str, 
                          quality: float, timestamp: float, utilization: float = 0.0):
        """Update link quality and utilization"""
        link_key = (source, destination)
        if link_key in self.link_states:
            self.link_states[link_key].quality = quality
            self.link_states[link_key].loss_rate = 1.0 - quality
            self.link_states[link_key].last_updated = timestamp
            self.link_states[link_key].utilization = utilization
            
            # Update reverse link
            reverse_key = (destination, source)
            if reverse_key in self.link_states:
                self.link_states[reverse_key].quality = quality
                self.link_states[reverse_key].loss_rate = 1.0 - loss_rate
                self.link_states[reverse_key].last_updated = timestamp
                self.link_states[reverse_key].utilization = utilization
            
            # Update graph edge weight
            if self.network_graph.has_edge(source, destination):
                # Weight based on quality and utilization
                weight = (1.0 / quality) * (1.0 + utilization)
                self.network_graph[source][destination]['weight'] = weight
                self.network_graph[destination][source]['weight'] = weight
    
    def find_route(self, source: str, destination: str, 
                  timestamp: float = 0.0, requirements: Dict[str, Any] = None) -> Optional[Route]:
        """
        Find optimal route between two nodes with requirements
        
        Args:
            source: Source node ID
            destination: Destination node ID
            timestamp: Current timestamp
            requirements: Route requirements (bandwidth, latency, reliability, etc.)
            
        Returns:
            Route information or None if no route exists
        """
        if source == destination:
            return Route(source, destination, [source], 0.0, 0, 1.0, timestamp, 
                        float('inf'), 0.0, 1.0, 0.0)
        
        if not self.network_graph.has_node(source) or not self.network_graph.has_node(destination):
            logging.warning(f"Node not found: {source} or {destination}")
            return None
        
        requirements = requirements or {}
        
        if self.algorithm == RoutingAlgorithm.DIJKSTRA:
            return self._dijkstra_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.AODV:
            return self._aodv_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.OLSR:
            return self._olsr_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.DSDV:
            return self._dsdv_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.DSR:
            return self._dsr_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.BATMAN:
            return self._batman_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.MULTIPATH:
            return self._multipath_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.LOAD_BALANCED:
            return self._load_balanced_route(source, destination, timestamp, requirements)
        elif self.algorithm == RoutingAlgorithm.ADAPTIVE:
            return self._adaptive_route(source, destination, timestamp, requirements)
        else:
            return self._dijkstra_route(source, destination, timestamp, requirements)
    
    def _dijkstra_route(self, source: str, destination: str, 
                       timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using Dijkstra's algorithm with requirements"""
        try:
            # Use NetworkX implementation with custom weight function
            path = nx.shortest_path(self.network_graph, source, destination, weight='weight')
            
            # Calculate route metrics
            total_cost = 0.0
            total_quality = 1.0
            total_latency = 0.0
            min_bandwidth = float('inf')
            total_energy = 0.0
            
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.network_graph[node1][node2]
                total_cost += edge_data['weight']
                total_quality *= edge_data.get('quality', 1.0)
                total_latency += edge_data.get('latency', 0.0)
                min_bandwidth = min(min_bandwidth, edge_data.get('bandwidth', float('inf')))
                total_energy += edge_data.get('energy_cost', 0.0)
            
            # Check requirements
            if requirements:
                if 'min_bandwidth' in requirements and min_bandwidth < requirements['min_bandwidth']:
                    return None
                if 'max_latency' in requirements and total_latency > requirements['max_latency']:
                    return None
                if 'min_quality' in requirements and total_quality < requirements['min_quality']:
                    return None
            
            route = Route(
                source=source,
                destination=destination,
                path=path,
                cost=total_cost,
                hops=len(path) - 1,
                quality=total_quality,
                timestamp=timestamp,
                bandwidth=min_bandwidth,
                latency=total_latency,
                reliability=total_quality,
                energy_cost=total_energy
            )
            
            # Cache route
            self.routes[(source, destination)] = route
            
            return route
            
        except nx.NetworkXNoPath:
            logging.warning(f"No path found from {source} to {destination}")
            return None
    
    def _aodv_route(self, source: str, destination: str, 
                   timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using AODV-like algorithm"""
        # Check if we have a cached route
        route_key = (source, destination)
        if route_key in self.routes:
            cached_route = self.routes[route_key]
            # Check if route is still valid
            if timestamp - cached_route.timestamp < self.route_timeout:
                # Verify route is still valid
                if self._verify_route(cached_route):
                    return cached_route
        
        # Perform route discovery (simplified AODV)
        # In a real implementation, this would involve RREQ/RREP messages
        return self._dijkstra_route(source, destination, timestamp, requirements)
    
    def _olsr_route(self, source: str, destination: str, 
                   timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using OLSR-like algorithm"""
        # OLSR uses link state routing with MPR selection
        # Simplified implementation using Dijkstra with link state updates
        return self._dijkstra_route(source, destination, timestamp, requirements)
    
    def _dsdv_route(self, source: str, destination: str, 
                   timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using DSDV-like algorithm"""
        # DSDV maintains routing tables with sequence numbers
        # Simplified implementation
        return self._dijkstra_route(source, destination, timestamp, requirements)
    
    def _dsr_route(self, source: str, destination: str, 
                  timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using DSR-like algorithm"""
        # DSR uses source routing with route discovery
        # Simplified implementation
        return self._dijkstra_route(source, destination, timestamp, requirements)
    
    def _batman_route(self, source: str, destination: str, 
                     timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using BATMAN-like algorithm"""
        # BATMAN uses originator messages for route selection
        # Simplified implementation
        return self._dijkstra_route(source, destination, timestamp, requirements)
    
    def _multipath_route(self, source: str, destination: str, 
                        timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find multiple paths and select the best one"""
        try:
            # Find k-shortest paths
            k = requirements.get('max_paths', 3)
            paths = list(nx.shortest_simple_paths(self.network_graph, source, destination, weight='weight'))
            
            if not paths:
                return None
            
            # Select best path based on requirements
            best_route = None
            best_score = float('-inf')
            
            for path in paths[:k]:
                route = self._calculate_route_metrics(path, timestamp)
                if route:
                    score = self._calculate_route_score(route, requirements)
                    if score > best_score:
                        best_score = score
                        best_route = route
            
            return best_route
            
        except nx.NetworkXNoPath:
            return None
    
    def _load_balanced_route(self, source: str, destination: str, 
                           timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route with load balancing considerations"""
        # Consider current link utilization in route selection
        try:
            # Create weighted graph with utilization
            weighted_graph = self.network_graph.copy()
            
            for edge in weighted_graph.edges(data=True):
                node1, node2, data = edge
                link_key = (node1, node2)
                if link_key in self.link_states:
                    utilization = self.link_states[link_key].utilization
                    # Increase weight based on utilization
                    data['weight'] *= (1.0 + utilization * 2.0)
            
            path = nx.shortest_path(weighted_graph, source, destination, weight='weight')
            return self._calculate_route_metrics(path, timestamp)
            
        except nx.NetworkXNoPath:
            return None
    
    def _adaptive_route(self, source: str, destination: str, 
                       timestamp: float, requirements: Dict[str, Any]) -> Optional[Route]:
        """Find route using adaptive routing based on current conditions"""
        # Adaptive routing considers multiple factors
        # For now, use a combination of quality and load balancing
        return self._load_balanced_route(source, destination, timestamp, requirements)
    
    def _calculate_route_metrics(self, path: List[str], timestamp: float) -> Optional[Route]:
        """Calculate metrics for a given path"""
        if len(path) < 2:
            return None
        
        total_cost = 0.0
        total_quality = 1.0
        total_latency = 0.0
        min_bandwidth = float('inf')
        total_energy = 0.0
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            edge_data = self.network_graph[node1][node2]
            total_cost += edge_data['weight']
            total_quality *= edge_data.get('quality', 1.0)
            total_latency += edge_data.get('latency', 0.0)
            min_bandwidth = min(min_bandwidth, edge_data.get('bandwidth', float('inf')))
            total_energy += edge_data.get('energy_cost', 0.0)
        
        return Route(
            source=path[0],
            destination=path[-1],
            path=path,
            cost=total_cost,
            hops=len(path) - 1,
            quality=total_quality,
            timestamp=timestamp,
            bandwidth=min_bandwidth,
            latency=total_latency,
            reliability=total_quality,
            energy_cost=total_energy
        )
    
    def _calculate_route_score(self, route: Route, requirements: Dict[str, Any]) -> float:
        """Calculate a score for a route based on requirements"""
        score = 0.0
        
        # Quality score
        score += route.quality * 10.0
        
        # Bandwidth score
        if 'min_bandwidth' in requirements:
            bandwidth_ratio = min(route.bandwidth / requirements['min_bandwidth'], 2.0)
            score += bandwidth_ratio * 5.0
        
        # Latency score
        if 'max_latency' in requirements:
            latency_ratio = max(0, 1.0 - route.latency / requirements['max_latency'])
            score += latency_ratio * 5.0
        
        # Energy efficiency score
        score += (1.0 / (1.0 + route.energy_cost)) * 3.0
        
        # Hop count penalty
        score -= route.hops * 0.5
        
        return score
    
    def _verify_route(self, route: Route) -> bool:
        """Verify if a cached route is still valid"""
        for i in range(len(route.path) - 1):
            node1, node2 = route.path[i], route.path[i + 1]
            if not self.network_graph.has_edge(node1, node2):
                return False
            
            # Check link quality
            link_key = (node1, node2)
            if link_key in self.link_states:
                if self.link_states[link_key].quality < self.quality_threshold:
                    return False
        
        return True
    
    def find_multipath_routes(self, source: str, destination: str, 
                            max_paths: int = 3, requirements: Dict[str, Any] = None) -> List[Route]:
        """Find multiple paths between source and destination"""
        try:
            paths = list(nx.shortest_simple_paths(self.network_graph, source, destination, weight='weight'))
            
            routes = []
            for path in paths[:max_paths]:
                route = self._calculate_route_metrics(path, time.time())
                if route and self._meets_requirements(route, requirements):
                    routes.append(route)
            
            return routes
            
        except nx.NetworkXNoPath:
            return []
    
    def _meets_requirements(self, route: Route, requirements: Dict[str, Any]) -> bool:
        """Check if route meets specified requirements"""
        if not requirements:
            return True
        
        if 'min_bandwidth' in requirements and route.bandwidth < requirements['min_bandwidth']:
            return False
        
        if 'max_latency' in requirements and route.latency > requirements['max_latency']:
            return False
        
        if 'min_quality' in requirements and route.quality < requirements['min_quality']:
            return False
        
        if 'max_hops' in requirements and route.hops > requirements['max_hops']:
            return False
        
        return True
    
    def send_coordination_message(self, message: CoordinationMessage):
        """Send a coordination message"""
        self.coordination_messages.append(message)
        logging.info(f"Coordination message sent: {message.message_type} from {message.source} to {message.destination}")
    
    def _process_coordination_messages(self, current_time: float):
        """Process coordination messages"""
        # Remove old messages
        self.coordination_messages = [msg for msg in self.coordination_messages 
                                    if current_time - msg.timestamp < 60.0]
        
        # Process messages based on coordination protocol
        if self.coordination_protocol == CoordinationProtocol.LEADER_FOLLOWER:
            self._process_leader_follower_messages(current_time)
        elif self.coordination_protocol == CoordinationProtocol.CONSENSUS:
            self._process_consensus_messages(current_time)
        elif self.coordination_protocol == CoordinationProtocol.HIERARCHICAL:
            self._process_hierarchical_messages(current_time)
        elif self.coordination_protocol == CoordinationProtocol.PEER_TO_PEER:
            self._process_peer_to_peer_messages(current_time)
        elif self.coordination_protocol == CoordinationProtocol.SWARM:
            self._process_swarm_messages(current_time)
    
    def _process_leader_follower_messages(self, current_time: float):
        """Process leader-follower coordination messages"""
        # Select leader if none exists
        if not self.leader_node and self.node_states:
            # Select leader based on energy level and position
            best_leader = max(self.node_states.values(), 
                            key=lambda n: n.energy_level + (1.0 / (1.0 + len(n.neighbors))))
            self.leader_node = best_leader.node_id
            logging.info(f"Leader selected: {self.leader_node}")
        
        # Process messages
        for message in self.coordination_messages:
            if message.message_type == "position":
                # Update node position
                if message.source in self.node_states:
                    self.node_states[message.source].position = message.payload.get('position', (0, 0, 0))
                    self.node_states[message.source].last_updated = current_time
            
            elif message.message_type == "status":
                # Update node status
                if message.source in self.node_states:
                    self.node_states[message.source].energy_level = message.payload.get('energy_level', 100.0)
                    self.node_states[message.source].load = message.payload.get('load', 0.0)
    
    def _process_consensus_messages(self, current_time: float):
        """Process consensus-based coordination messages"""
        # Simplified consensus implementation
        pass
    
    def _process_hierarchical_messages(self, current_time: float):
        """Process hierarchical coordination messages"""
        # Simplified hierarchical implementation
        pass
    
    def _process_peer_to_peer_messages(self, current_time: float):
        """Process peer-to-peer coordination messages"""
        # Simplified P2P implementation
        pass
    
    def _process_swarm_messages(self, current_time: float):
        """Process swarm coordination messages"""
        # Simplified swarm implementation
        pass
    
    def _update_link_states(self, current_time: float):
        """Update link states based on current conditions"""
        for link_key, link_state in self.link_states.items():
            # Update interference based on nearby transmissions
            link_state.interference = self._calculate_interference(link_key)
            
            # Update quality based on interference
            if link_state.interference > 0.0:
                link_state.quality = max(0.1, link_state.quality - link_state.interference * 0.1)
    
    def _calculate_interference(self, link_key: Tuple[str, str]) -> float:
        """Calculate interference level for a link"""
        # Simplified interference calculation
        # In reality, this would consider frequency overlap, distance, power, etc.
        interference = 0.0
        
        link_state = self.link_states[link_key]
        source_pos = self.node_states[link_key[0]].position
        dest_pos = self.node_states[link_key[1]].position
        
        # Check interference from other links
        for other_key, other_state in self.link_states.items():
            if other_key == link_key:
                continue
            
            other_source_pos = self.node_states[other_key[0]].position
            other_dest_pos = self.node_states[other_key[1]].position
            
            # Calculate distance between link midpoints
            mid1 = ((source_pos[0] + dest_pos[0]) / 2, 
                   (source_pos[1] + dest_pos[1]) / 2, 
                   (source_pos[2] + dest_pos[2]) / 2)
            mid2 = ((other_source_pos[0] + other_dest_pos[0]) / 2,
                   (other_source_pos[1] + other_dest_pos[1]) / 2,
                   (other_source_pos[2] + other_dest_pos[2]) / 2)
            
            distance = math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2 + (mid1[2] - mid2[2])**2)
            
            # Interference decreases with distance
            if distance < 1000:  # Close links
                interference += 0.3
            elif distance < 5000:  # Medium distance
                interference += 0.1
            elif distance < 10000:  # Far links
                interference += 0.05
        
        return min(interference, 1.0)
    
    def _update_routing_tables(self, current_time: float):
        """Update routing tables for all nodes"""
        for node_id in self.node_states:
            routing_table = {}
            
            for dest_id in self.node_states:
                if dest_id != node_id:
                    route = self.find_route(node_id, dest_id, current_time)
                    if route:
                        routing_table[dest_id] = route.path[1] if len(route.path) > 1 else dest_id
            
            self.node_states[node_id].routing_table = routing_table
            self.node_states[node_id].last_updated = current_time
    
    def _update_load_balancing(self, current_time: float):
        """Update load balancing across the network"""
        # Reset traffic distribution
        self.traffic_distribution.clear()
        
        # Calculate traffic distribution based on link quality and utilization
        for link_key, link_state in self.link_states.items():
            # Higher quality and lower utilization links get more traffic
            quality_factor = link_state.quality
            utilization_factor = 1.0 - link_state.utilization
            self.traffic_distribution[link_key] = quality_factor * utilization_factor
    
    def add_coordination_callback(self, callback: Callable):
        """Add a callback for coordination events"""
        self.coordination_callbacks.append(callback)
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get current network state"""
        return {
            'nodes': {node_id: {
                'position': state.position,
                'energy_level': state.energy_level,
                'load': state.load,
                'neighbors': state.neighbors,
                'routing_table': state.routing_table
            } for node_id, state in self.node_states.items()},
            'links': {link_key: {
                'quality': link_state.quality,
                'utilization': link_state.utilization,
                'interference': link_state.interference,
                'bandwidth': link_state.bandwidth
            } for link_key, link_state in self.link_states.items()},
            'routes': len(self.routes),
            'leader_node': self.leader_node,
            'coordination_protocol': self.coordination_protocol.value
        }
    
    def calculate_network_connectivity(self) -> Dict[str, Any]:
        """Calculate comprehensive network connectivity metrics"""
        if not self.node_states:
            return {}
        
        # Calculate connectivity metrics
        total_nodes = len(self.node_states)
        total_links = len(self.link_states) // 2  # Divide by 2 for bidirectional links
        
        # Calculate average node degree
        total_degree = sum(len(state.neighbors) for state in self.node_states.values())
        avg_degree = total_degree / total_nodes if total_nodes > 0 else 0
        
        # Calculate network diameter
        try:
            diameter = nx.diameter(self.network_graph)
        except nx.NetworkXError:
            diameter = float('inf')
        
        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(self.network_graph)
        except nx.NetworkXError:
            avg_path_length = float('inf')
        
        # Calculate clustering coefficient
        try:
            clustering_coeff = nx.average_clustering(self.network_graph)
        except nx.NetworkXError:
            clustering_coeff = 0.0
        
        # Calculate link quality distribution
        qualities = [link_state.quality for link_state in self.link_states.values()]
        quality_distribution = {
            'excellent': len([q for q in qualities if q >= 0.9]),
            'good': len([q for q in qualities if 0.7 <= q < 0.9]),
            'fair': len([q for q in qualities if 0.5 <= q < 0.7]),
            'poor': len([q for q in qualities if q < 0.5])
        }
        
        return {
            'total_nodes': total_nodes,
            'total_links': total_links,
            'average_degree': avg_degree,
            'network_diameter': diameter,
            'average_path_length': avg_path_length,
            'clustering_coefficient': clustering_coeff,
            'quality_distribution': quality_distribution,
            'connectivity_ratio': total_links / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        }
    
    def get_node_neighbors(self, node_id: str) -> List[str]:
        """Get list of neighbors for a node"""
        if node_id in self.node_states:
            return self.node_states[node_id].neighbors
        return []
    
    def get_link_state(self, source: str, destination: str) -> Optional[LinkState]:
        """Get state of a specific link"""
        link_key = (source, destination)
        return self.link_states.get(link_key)
    
    def remove_node(self, node_id: str):
        """Remove a node from the network"""
        if node_id in self.node_states:
            # Remove from network graph
            self.network_graph.remove_node(node_id)
            
            # Remove from node states
            del self.node_states[node_id]
            
            # Remove associated links
            links_to_remove = []
            for link_key in self.link_states:
                if link_key[0] == node_id or link_key[1] == node_id:
                    links_to_remove.append(link_key)
            
            for link_key in links_to_remove:
                del self.link_states[link_key]
            
            # Remove from neighbor lists
            for state in self.node_states.values():
                if node_id in state.neighbors:
                    state.neighbors.remove(node_id)
            
            # Update leader if necessary
            if self.leader_node == node_id:
                self.leader_node = None
            
            logging.info(f"Node removed: {node_id}")
    
    def remove_link(self, source: str, destination: str):
        """Remove a link between nodes"""
        # Remove from network graph
        if self.network_graph.has_edge(source, destination):
            self.network_graph.remove_edge(source, destination)
        
        # Remove from link states
        link_key = (source, destination)
        reverse_key = (destination, source)
        
        if link_key in self.link_states:
            del self.link_states[link_key]
        if reverse_key in self.link_states:
            del self.link_states[reverse_key]
        
        # Remove from neighbor lists
        if source in self.node_states and destination in self.node_states[source].neighbors:
            self.node_states[source].neighbors.remove(destination)
        if destination in self.node_states and source in self.node_states[destination].neighbors:
            self.node_states[destination].neighbors.remove(source)
        
        logging.info(f"Link removed: {source} <-> {destination}")
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get complete network topology"""
        return {
            'nodes': {node_id: {
                'position': state.position,
                'type': 'drone' if 'DRONE' in node_id else 'ground_station',
                'energy_level': state.energy_level,
                'neighbors': state.neighbors
            } for node_id, state in self.node_states.items()},
            'links': [{
                'source': link_key[0],
                'destination': link_key[1],
                'quality': link_state.quality,
                'bandwidth': link_state.bandwidth,
                'latency': link_state.latency
            } for link_key, link_state in self.link_states.items()],
            'algorithm': self.algorithm.value,
            'coordination_protocol': self.coordination_protocol.value
        }
    
    def clear_network(self):
        """Clear all network data"""
        self.network_graph.clear()
        self.link_states.clear()
        self.node_states.clear()
        self.routes.clear()
        self.coordination_messages.clear()
        self.leader_node = None
        self.traffic_distribution.clear()
        logging.info("Network cleared") 