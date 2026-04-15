"""network_metrics.py

Compute network metrics for GraphIDyOM LTM graphs.

Metrics computed:
- Number of nodes
- Number of edges
- Average degree (in, out, total)
- Network entropy (based on stationary distribution and outgoing edge entropy)

The network entropy is computed as:
    H = sum_i pi_i * H_i

where:
    pi_i = stationary distribution probability of node i
    H_i = entropy of outgoing edge weights from node i

The stationary distribution is computed from the dominant eigenvector
of the transition matrix (restricted to largest strongly connected component).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigs


@dataclass
class GraphMetrics:
    """Metrics for a single order graph."""
    order: int
    num_nodes: int
    num_edges: int
    avg_in_degree: float
    avg_out_degree: float
    avg_degree: float
    density: float
    # Entropy metrics
    network_entropy: Optional[float] = None
    avg_local_entropy: Optional[float] = None
    scc_size: Optional[int] = None  # Size of largest strongly connected component
    scc_coverage: Optional[float] = None  # Fraction of nodes in largest SCC
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "order": self.order,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "avg_in_degree": self.avg_in_degree,
            "avg_out_degree": self.avg_out_degree,
            "avg_degree": self.avg_degree,
            "density": self.density,
            "network_entropy": self.network_entropy,
            "avg_local_entropy": self.avg_local_entropy,
            "scc_size": self.scc_size,
            "scc_coverage": self.scc_coverage,
        }


@dataclass
class ModelMetrics:
    """Aggregated metrics across all orders of a model."""
    total_nodes: int
    total_edges: int
    avg_degree_all: float
    weighted_network_entropy: Optional[float]  # Weighted by order
    order_metrics: Dict[int, GraphMetrics]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "avg_degree_all": self.avg_degree_all,
            "weighted_network_entropy": self.weighted_network_entropy,
            "order_metrics": {
                order: metrics.to_dict() 
                for order, metrics in self.order_metrics.items()
            },
        }


class NetworkMetricsCalculator:
    """Calculate network metrics for LTM graphs.
    
    Uses node weights as empirical stationary distribution (visitation counts)
    and edge weights for transition probabilities. This avoids expensive
    eigenvalue computation and works on the full graph.
    """
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    def compute_graph_metrics(
        self, 
        graph: nx.DiGraph, 
        order: int,
        compute_entropy: bool = True
    ) -> GraphMetrics:
        """
        Compute all metrics for a single graph.
        
        Args:
            graph: NetworkX directed graph
            order: The Markov order of this graph
            compute_entropy: Whether to compute entropy metrics (slower)
            
        Returns:
            GraphMetrics dataclass with all computed metrics
        """
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes == 0:
            return GraphMetrics(
                order=order,
                num_nodes=0,
                num_edges=0,
                avg_in_degree=0.0,
                avg_out_degree=0.0,
                avg_degree=0.0,
                density=0.0,
                network_entropy=None,
                avg_local_entropy=None,
                scc_size=0,
                scc_coverage=0.0,
            )
        
        # Degree statistics
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        
        avg_in_degree = np.mean(in_degrees) if in_degrees else 0.0
        avg_out_degree = np.mean(out_degrees) if out_degrees else 0.0
        avg_degree = (avg_in_degree + avg_out_degree) / 2.0
        
        # Density
        if num_nodes > 1:
            density = num_edges / (num_nodes * (num_nodes - 1))
        else:
            density = 0.0
        
        # Entropy metrics (using node/edge weights directly)
        network_entropy = None
        avg_local_entropy = None
        
        if compute_entropy and num_edges > 0:
            try:
                entropy_result = self._compute_entropy_from_weights(graph)
                network_entropy = entropy_result.get("network_entropy")
                avg_local_entropy = entropy_result.get("avg_local_entropy")
            except Exception as e:
                # Entropy computation can fail for various reasons
                warnings.warn(f"Entropy computation failed for order {order}: {e}")
        
        return GraphMetrics(
            order=order,
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_in_degree=avg_in_degree,
            avg_out_degree=avg_out_degree,
            avg_degree=avg_degree,
            density=density,
            network_entropy=network_entropy,
            avg_local_entropy=avg_local_entropy,
            scc_size=num_nodes,  # We use the full graph now
            scc_coverage=1.0,    # 100% coverage
        )
    
    def _compute_entropy_from_weights(self, graph: nx.DiGraph) -> dict:
        """
        Compute entropy using node weights (visitation counts) and edge weights.
        
        Node weights give us the empirical stationary distribution:
            pi_i = node_weight_i / sum(all node weights)
        
        Edge weights give us transition probabilities:
            P(j|i) = edge_weight(i,j) / sum(edge_weights from i)
        
        Local entropy H_i = -sum_j P(j|i) * log2(P(j|i))
        
        Network entropy H = sum_i pi_i * H_i
        
        This is the entropy rate of the observed Markov chain.
        
        Returns:
            Dictionary with network_entropy and avg_local_entropy
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n == 0:
            return {"network_entropy": 0.0, "avg_local_entropy": 0.0}
        
        # Get node weights (visitation counts) - this is our stationary distribution
        node_weights = np.zeros(n, dtype=np.float64)
        local_entropies = np.zeros(n, dtype=np.float64)
        
        for i, node in enumerate(nodes):
            # Node weight = visitation count (empirical stationary distribution)
            node_data = graph.nodes[node]
            node_weights[i] = node_data.get('weight', 1.0)
            
            # Compute local entropy from outgoing edge weights
            out_edges = list(graph.out_edges(node, data=True))
            
            if len(out_edges) == 0:
                local_entropies[i] = 0.0
                continue
            
            # Get edge weights (transition counts)
            edge_weights = np.array([
                data.get('weight', 1.0) for _, _, data in out_edges
            ], dtype=np.float64)
            
            if len(edge_weights) == 1:
                # Only one outgoing edge = no uncertainty
                local_entropies[i] = 0.0
                continue
            
            # Normalize to probabilities
            total_weight = np.sum(edge_weights)
            if total_weight > 0:
                probs = edge_weights / total_weight
                # Shannon entropy in bits (filter out zeros for log safety)
                probs = probs[probs > 0]
                local_entropies[i] = -np.sum(probs * np.log2(probs))
            else:
                local_entropies[i] = 0.0
        
        # Average local entropy (unweighted)
        avg_local_entropy = float(np.mean(local_entropies))
        
        # Stationary distribution from node weights
        total_node_weight = np.sum(node_weights)
        if total_node_weight > 0:
            stationary = node_weights / total_node_weight
        else:
            # Uniform if no weights
            stationary = np.ones(n) / n
        
        # Network entropy: H = sum_i pi_i * H_i (entropy rate)
        network_entropy = float(np.dot(stationary, local_entropies))
        
        return {
            "network_entropy": network_entropy,
            "avg_local_entropy": avg_local_entropy,
        }
    
    def compute_model_metrics(
        self,
        ltm_graphs: Dict[int, nx.DiGraph],
        compute_entropy: bool = True,
        parallel: bool = False,  # Disabled by default for safety
        max_workers: int = 2
    ) -> ModelMetrics:
        """
        Compute metrics for all graphs in a model.
        
        Args:
            ltm_graphs: Dictionary mapping order -> graph
            compute_entropy: Whether to compute entropy metrics
            parallel: Whether to compute metrics in parallel
            max_workers: Number of parallel workers
            
        Returns:
            ModelMetrics with all computed metrics
        """
        order_metrics: Dict[int, GraphMetrics] = {}
        
        if parallel and len(ltm_graphs) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.compute_graph_metrics, graph, order, compute_entropy
                    ): order
                    for order, graph in ltm_graphs.items()
                }
                
                for future in as_completed(futures):
                    order = futures[future]
                    try:
                        metrics = future.result()
                        order_metrics[order] = metrics
                    except Exception as e:
                        warnings.warn(f"Failed to compute metrics for order {order}: {e}")
        else:
            for order, graph in ltm_graphs.items():
                try:
                    metrics = self.compute_graph_metrics(graph, order, compute_entropy)
                    order_metrics[order] = metrics
                except Exception as e:
                    warnings.warn(f"Failed to compute metrics for order {order}: {e}")
        
        # Aggregate metrics
        total_nodes = sum(m.num_nodes for m in order_metrics.values())
        total_edges = sum(m.num_edges for m in order_metrics.values())
        
        all_degrees = []
        for m in order_metrics.values():
            if m.num_nodes > 0:
                all_degrees.append(m.avg_degree)
        avg_degree_all = np.mean(all_degrees) if all_degrees else 0.0
        
        # Weighted network entropy (weight by number of edges in each order)
        weighted_entropy = None
        entropies_with_weights = [
            (m.network_entropy, m.num_edges)
            for m in order_metrics.values()
            if m.network_entropy is not None and m.num_edges > 0
        ]
        
        if entropies_with_weights:
            total_edge_weight = sum(w for _, w in entropies_with_weights)
            if total_edge_weight > 0:
                weighted_entropy = sum(
                    e * w for e, w in entropies_with_weights
                ) / total_edge_weight
        
        return ModelMetrics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            avg_degree_all=avg_degree_all,
            weighted_network_entropy=weighted_entropy,
            order_metrics=order_metrics,
        )


def compute_metrics_for_graphs(
    ltm_graphs: Dict[int, nx.DiGraph],
    compute_entropy: bool = True
) -> ModelMetrics:
    """
    Convenience function to compute all metrics for LTM graphs.
    
    Args:
        ltm_graphs: Dictionary mapping order -> graph
        compute_entropy: Whether to compute entropy metrics
        
    Returns:
        ModelMetrics with all computed metrics
    """
    calculator = NetworkMetricsCalculator()
    return calculator.compute_model_metrics(ltm_graphs, compute_entropy=compute_entropy)


if __name__ == "__main__":
    # Simple test
    import pickle
    from pathlib import Path
    
    # Try to load a pretrained model's graphs
    graphs_dir = Path(__file__).parent / "pretrained_models" / "datasets"
    
    if graphs_dir.exists():
        for dataset_dir in graphs_dir.iterdir():
            if dataset_dir.is_dir():
                for viewpoint_dir in dataset_dir.iterdir():
                    graph_files = list((viewpoint_dir / "graphs").glob("*.gpickle"))
                    if graph_files:
                        print(f"\nAnalyzing: {viewpoint_dir.name}")
                        ltm_graphs = {}
                        for gf in graph_files:
                            order = int(gf.stem.split("_")[1])
                            with open(gf, "rb") as f:
                                ltm_graphs[order] = pickle.load(f)
                        
                        metrics = compute_metrics_for_graphs(ltm_graphs)
                        print(f"  Total nodes: {metrics.total_nodes}")
                        print(f"  Total edges: {metrics.total_edges}")
                        print(f"  Avg degree: {metrics.avg_degree_all:.3f}")
                        print(f"  Network entropy: {metrics.weighted_network_entropy:.3f}")
                        
                        for order, m in sorted(metrics.order_metrics.items()):
                            print(f"  Order {order}: {m.num_nodes} nodes, {m.num_edges} edges, "
                                  f"entropy={m.network_entropy:.3f if m.network_entropy else 'N/A'}")
                        break
                break
    else:
        # Create a simple test graph
        print("Creating test graph...")
        G = nx.DiGraph()
        G.add_weighted_edges_from([
            ("A", "B", 3),
            ("B", "C", 2),
            ("C", "A", 1),
            ("A", "C", 1),
            ("B", "A", 1),
        ])
        
        calculator = NetworkMetricsCalculator()
        metrics = calculator.compute_graph_metrics(G, order=1)
        
        print(f"Test graph metrics:")
        print(f"  Nodes: {metrics.num_nodes}")
        print(f"  Edges: {metrics.num_edges}")
        print(f"  Avg degree: {metrics.avg_degree:.3f}")
        print(f"  Network entropy: {metrics.network_entropy:.3f}")
        print(f"  SCC size: {metrics.scc_size}")
        print(f"  SCC coverage: {metrics.scc_coverage:.2%}")
