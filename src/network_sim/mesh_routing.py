"""
Mesh Network Routing for HALE Drone Communications

This module provides mesh network routing algorithms and protocols
for HALE drone communication networks.
"""

import math
import heapq
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx


class RoutingAlgorithm(Enum):
    """Types of routing algorithms"""
    DIJKSTRA = "dijkstra"
    AODV = "aodv"
    OLSR = "olsr"
    BATMAN = "batman"
    CUSTOM = "custom"


@dataclass
class Route:
    """Route information"""
    source: str
    destination: str
    path: List[str]
    cost: float
    hops: int
    quality: float  # 0.0 to 1.0
    timestamp: float


@dataclass
class LinkState:
    """State of a network link"""
    source: str
    destination: str
    bandwidth: float  # bits per second
    latency: float  # seconds
    loss_rate: float  # 0.0 to 1.0
    quality: float  # 0.0 to 1.0
    last_updated: float


class MeshRouting:
    """
    Mesh network routing for HALE drone communications
    """
    
    def __init__(self, algorithm: RoutingAlgorithm = RoutingAlgorithm.DIJKSTRA):
        self.algorithm = algorithm
        self.network_graph = nx.Graph()
        self.link_states: Dict[Tuple[str, str], LinkState] = {}
        self.routes: Dict[Tuple[str, str], Route] = {}
        self.node_positions: Dict[str, Tuple[float, float, float]] = {}
        
        logging.info(f"Mesh Routing initialized with {algorithm.value} algorithm")
    
    def add_node(self, node_id: str, position: Tuple[float, float, float]):
        """Add a node to the mesh network"""
        self.network_graph.add_node(node_id)
        self.node_positions[node_id] = position
        logging.info(f"Node added to mesh: {node_id} at {position}")
    
    def add_link(self, source: str, destination: str, bandwidth: float,
                latency: float, loss_rate: float = 0.0):
        """Add a link between nodes"""
        # Add to network graph
        self.network_graph.add_edge(source, destination, 
                                   weight=latency, bandwidth=bandwidth, loss_rate=loss_rate)
        
        # Create link state
        link_state = LinkState(
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            latency=latency,
            loss_rate=loss_rate,
            quality=1.0 - loss_rate,
            last_updated=0.0
        )
        
        self.link_states[(source, destination)] = link_state
        self.link_states[(destination, source)] = link_state  # Bidirectional
        
        logging.info(f"Link added: {source} <-> {destination} ({bandwidth/1e6:.1f} Mbps)")
    
    def update_link_quality(self, source: str, destination: str, 
                          quality: float, timestamp: float):
        """Update link quality"""
        link_key = (source, destination)
        if link_key in self.link_states:
            self.link_states[link_key].quality = quality
            self.link_states[link_key].loss_rate = 1.0 - quality
            self.link_states[link_key].last_updated = timestamp
            
            # Update reverse link
            reverse_key = (destination, source)
            if reverse_key in self.link_states:
                self.link_states[reverse_key].quality = quality
                self.link_states[reverse_key].loss_rate = 1.0 - quality
                self.link_states[reverse_key].last_updated = timestamp
            
            # Update graph edge weight
            if self.network_graph.has_edge(source, destination):
                self.network_graph[source][destination]['weight'] = 1.0 / quality
                self.network_graph[destination][source]['weight'] = 1.0 / quality
    
    def find_route(self, source: str, destination: str, 
                  timestamp: float = 0.0) -> Optional[Route]:
        """
        Find optimal route between two nodes
        
        Args:
            source: Source node ID
            destination: Destination node ID
            timestamp: Current timestamp
            
        Returns:
            Route information or None if no route exists
        """
        if source == destination:
            return Route(source, destination, [source], 0.0, 0, 1.0, timestamp)
        
        if not self.network_graph.has_node(source) or not self.network_graph.has_node(destination):
            logging.warning(f"Node not found: {source} or {destination}")
            return None
        
        if self.algorithm == RoutingAlgorithm.DIJKSTRA:
            return self._dijkstra_route(source, destination, timestamp)
        elif self.algorithm == RoutingAlgorithm.AODV:
            return self._aodv_route(source, destination, timestamp)
        elif self.algorithm == RoutingAlgorithm.OLSR:
            return self._olsr_route(source, destination, timestamp)
        else:
            return self._dijkstra_route(source, destination, timestamp)
    
    def _dijkstra_route(self, source: str, destination: str, 
                       timestamp: float) -> Optional[Route]:
        """Find route using Dijkstra's algorithm"""
        try:
            # Use NetworkX implementation
            path = nx.shortest_path(self.network_graph, source, destination, weight='weight')
            
            # Calculate route metrics
            total_cost = 0.0
            total_quality = 1.0
            
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.network_graph[node1][node2]
                total_cost += edge_data['weight']
                total_quality *= edge_data.get('quality', 1.0)
            
            route = Route(
                source=source,
                destination=destination,
                path=path,
                cost=total_cost,
                hops=len(path) - 1,
                quality=total_quality,
                timestamp=timestamp
            )
            
            # Cache route
            self.routes[(source, destination)] = route
            
            return route
            
        except nx.NetworkXNoPath:
            logging.warning(f"No path found from {source} to {destination}")
            return None
    
    def _aodv_route(self, source: str, destination: str, 
                   timestamp: float) -> Optional[Route]:
        """Find route using AODV-like algorithm"""
        # Simplified AODV implementation
        # In practice, this would involve route discovery and maintenance
        
        # Check if we have a cached route
        route_key = (source, destination)
        if route_key in self.routes:
            cached_route = self.routes[route_key]
            # Check if route is still valid (simplified)
            if timestamp - cached_route.timestamp < 30.0:  # 30 second timeout
                return cached_route
        
        # Perform route discovery (simplified)
        return self._dijkstra_route(source, destination, timestamp)
    
    def _olsr_route(self, source: str, destination: str, 
                   timestamp: float) -> Optional[Route]:
        """Find route using OLSR-like algorithm"""
        # Simplified OLSR implementation
        # In practice, this would use link state routing with MPR selection
        
        # For now, use Dijkstra's algorithm
        return self._dijkstra_route(source, destination, timestamp)
    
    def find_multipath_routes(self, source: str, destination: str, 
                            max_paths: int = 3) -> List[Route]:
        """Find multiple paths between source and destination"""
        routes = []
        
        try:
            # Find k-shortest paths
            paths = list(nx.shortest_simple_paths(self.network_graph, source, destination, 
                                                weight='weight', k=max_paths))
            
            for path in paths:
                # Calculate route metrics
                total_cost = 0.0
                total_quality = 1.0
                
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i + 1]
                    edge_data = self.network_graph[node1][node2]
                    total_cost += edge_data['weight']
                    total_quality *= edge_data.get('quality', 1.0)
                
                route = Route(
                    source=source,
                    destination=destination,
                    path=path,
                    cost=total_cost,
                    hops=len(path) - 1,
                    quality=total_quality,
                    timestamp=0.0
                )
                
                routes.append(route)
            
        except nx.NetworkXNoPath:
            logging.warning(f"No paths found from {source} to {destination}")
        
        return routes
    
    def calculate_network_connectivity(self) -> Dict[str, Any]:
        """Calculate network connectivity metrics"""
        if not self.network_graph.nodes():
            return {'connected_components': 0, 'largest_component_size': 0}
        
        # Find connected components
        components = list(nx.connected_components(self.network_graph))
        
        # Calculate metrics
        largest_component = max(components, key=len) if components else set()
        
        # Calculate average node degree
        avg_degree = sum(dict(self.network_graph.degree()).values()) / len(self.network_graph.nodes())
        
        # Calculate network diameter (longest shortest path)
        try:
            diameter = nx.diameter(self.network_graph)
        except nx.NetworkXError:
            diameter = float('inf')
        
        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(self.network_graph)
        except nx.NetworkXError:
            avg_path_length = float('inf')
        
        return {
            'connected_components': len(components),
            'largest_component_size': len(largest_component),
            'total_nodes': len(self.network_graph.nodes()),
            'total_links': len(self.network_graph.edges()),
            'average_degree': avg_degree,
            'diameter': diameter,
            'average_path_length': avg_path_length
        }
    
    def get_node_neighbors(self, node_id: str) -> List[str]:
        """Get list of neighboring nodes"""
        if node_id in self.network_graph:
            return list(self.network_graph.neighbors(node_id))
        return []
    
    def get_link_state(self, source: str, destination: str) -> Optional[LinkState]:
        """Get current state of a link"""
        return self.link_states.get((source, destination))
    
    def remove_node(self, node_id: str):
        """Remove a node and all its links"""
        if node_id in self.network_graph:
            self.network_graph.remove_node(node_id)
            
            # Remove from positions
            if node_id in self.node_positions:
                del self.node_positions[node_id]
            
            # Remove related link states
            keys_to_remove = []
            for key in self.link_states.keys():
                if node_id in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.link_states[key]
            
            # Remove related routes
            routes_to_remove = []
            for key in self.routes.keys():
                if node_id in key:
                    routes_to_remove.append(key)
            
            for key in routes_to_remove:
                del self.routes[key]
            
            logging.info(f"Node removed from mesh: {node_id}")
    
    def remove_link(self, source: str, destination: str):
        """Remove a link between nodes"""
        if self.network_graph.has_edge(source, destination):
            self.network_graph.remove_edge(source, destination)
            
            # Remove link states
            keys_to_remove = [(source, destination), (destination, source)]
            for key in keys_to_remove:
                if key in self.link_states:
                    del self.link_states[key]
            
            # Remove related routes
            routes_to_remove = []
            for key in self.routes.keys():
                route = self.routes[key]
                if source in route.path and destination in route.path:
                    routes_to_remove.append(key)
            
            for key in routes_to_remove:
                del self.routes[key]
            
            logging.info(f"Link removed: {source} <-> {destination}")
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get current network topology"""
        return {
            'nodes': list(self.network_graph.nodes()),
            'edges': list(self.network_graph.edges()),
            'node_positions': self.node_positions,
            'link_states': {str(k): v.__dict__ for k, v in self.link_states.items()},
            'connectivity': self.calculate_network_connectivity()
        }
    
    def clear_network(self):
        """Clear all network data"""
        self.network_graph.clear()
        self.link_states.clear()
        self.routes.clear()
        self.node_positions.clear()
        logging.info("Mesh network cleared") 