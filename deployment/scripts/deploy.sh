#!/bin/bash

# Quantum HALE Drone System Deployment Script
# This script deploys the complete Quantum HALE Drone System to Kubernetes

set -e

# Configuration
NAMESPACE="quantum-hale"
REGISTRY="your-registry.com"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if Docker is installed (for building images)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed. Image building will be skipped."
        SKIP_BUILD=true
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    if [ "$SKIP_BUILD" = true ]; then
        log_warning "Skipping image build (Docker not available)"
        return
    fi
    
    log_info "Building Docker images..."
    
    # Build main simulation image
    log_info "Building quantum-hale simulation image..."
    docker build -f Dockerfile.simulation -t ${REGISTRY}/quantum-hale:${VERSION} .
    
    # Build network simulator image
    log_info "Building network simulator image..."
    docker build -f Dockerfile.network-sim -t ${REGISTRY}/quantum-hale-network:${VERSION} .
    
    # Build quantum simulator image
    log_info "Building quantum simulator image..."
    docker build -f Dockerfile.quantum-sim -t ${REGISTRY}/quantum-hale-quantum:${VERSION} .
    
    log_success "Docker images built successfully"
}

# Push Docker images
push_images() {
    if [ "$SKIP_BUILD" = true ]; then
        log_warning "Skipping image push (Docker not available)"
        return
    fi
    
    log_info "Pushing Docker images to registry..."
    
    docker push ${REGISTRY}/quantum-hale:${VERSION}
    docker push ${REGISTRY}/quantum-hale-network:${VERSION}
    docker push ${REGISTRY}/quantum-hale-quantum:${VERSION}
    
    log_success "Docker images pushed successfully"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: ${NAMESPACE}"
    
    kubectl apply -f deployment/kubernetes/namespace.yaml
    
    log_success "Namespace created successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    kubectl apply -f deployment/kubernetes/monitoring-stack.yaml
    
    # Wait for monitoring services to be ready
    log_info "Waiting for monitoring services to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/influxdb -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n ${NAMESPACE}
    
    log_success "Monitoring stack deployed successfully"
}

# Deploy simulation services
deploy_simulation() {
    log_info "Deploying simulation services..."
    
    kubectl apply -f deployment/kubernetes/simulation-deployment.yaml
    
    # Wait for simulation services to be ready
    log_info "Waiting for simulation services to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/quantum-hale-simulation -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/network-simulator -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/quantum-simulator -n ${NAMESPACE}
    
    log_success "Simulation services deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if all pods are running
    PODS=$(kubectl get pods -n ${NAMESPACE} -o jsonpath='{.items[*].status.phase}')
    RUNNING_PODS=$(echo $PODS | tr ' ' '\n' | grep -c "Running" || true)
    TOTAL_PODS=$(echo $PODS | tr ' ' '\n' | wc -l)
    
    if [ "$RUNNING_PODS" -eq "$TOTAL_PODS" ]; then
        log_success "All pods are running successfully"
    else
        log_error "Some pods are not running. Check with: kubectl get pods -n ${NAMESPACE}"
        exit 1
    fi
    
    # Check services
    SERVICES=$(kubectl get services -n ${NAMESPACE} -o jsonpath='{.items[*].metadata.name}')
    log_info "Available services: $SERVICES"
    
    log_success "Deployment verification completed"
}

# Display access information
display_access_info() {
    log_info "Deployment completed successfully!"
    echo
    echo "=== Access Information ==="
    echo "Namespace: ${NAMESPACE}"
    echo
    echo "Services:"
    echo "- Simulation Dashboard: http://localhost:30080"
    echo "- Grafana Dashboard: http://localhost:30300 (admin/quantum-hale-2024)"
    echo "- Prometheus: http://localhost:30990"
    echo "- InfluxDB: http://localhost:8086 (admin/quantum-hale-2024)"
    echo
    echo "Useful commands:"
    echo "- View pods: kubectl get pods -n ${NAMESPACE}"
    echo "- View logs: kubectl logs -f deployment/quantum-hale-simulation -n ${NAMESPACE}"
    echo "- Port forward: kubectl port-forward service/quantum-hale-simulation-service 8080:8080 -n ${NAMESPACE}"
    echo
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment..."
    
    kubectl delete -f deployment/kubernetes/simulation-deployment.yaml --ignore-not-found=true
    kubectl delete -f deployment/kubernetes/monitoring-stack.yaml --ignore-not-found=true
    kubectl delete -f deployment/kubernetes/namespace.yaml --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting Quantum HALE Drone System deployment..."
    
    check_prerequisites
    build_images
    push_images
    create_namespace
    deploy_monitoring
    deploy_simulation
    verify_deployment
    display_access_info
    
    log_success "Quantum HALE Drone System deployed successfully!"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "cleanup")
        cleanup
        ;;
    "verify")
        verify_deployment
        ;;
    "monitoring")
        create_namespace
        deploy_monitoring
        ;;
    "simulation")
        create_namespace
        deploy_simulation
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|verify|monitoring|simulation}"
        echo
        echo "Commands:"
        echo "  deploy     - Deploy the complete system (default)"
        echo "  cleanup    - Remove all deployments"
        echo "  verify     - Verify deployment status"
        echo "  monitoring - Deploy only monitoring stack"
        echo "  simulation - Deploy only simulation services"
        exit 1
        ;;
esac 