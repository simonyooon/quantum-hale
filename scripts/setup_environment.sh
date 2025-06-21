#!/bin/bash

set -e

echo "Setting up Quantum HALE Drone development environment..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root"
   exit 1
fi

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get install -y \
    build-essential cmake ninja-build git curl wget \
    python3.10 python3.10-dev python3-pip python3-venv \
    software-properties-common

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Install liboqs (post-quantum cryptography library)
echo "Installing liboqs..."
if [ ! -d "/usr/local/include/oqs" ]; then
    git clone https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs
    cd /tmp/liboqs
    mkdir build && cd build
    cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local ..
    ninja
    sudo ninja install
    sudo ldconfig
    cd /
    rm -rf /tmp/liboqs
fi

# Create project directory structure
echo "Creating project structure..."
mkdir -p quantum-hale-drone/{src,tests,configs,data,models,scripts,docs,deployment}
cd quantum-hale-drone

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
echo "Setting up environment variables..."
cp env.example .env

# Initialize Git repository
echo "Initializing Git repository..."
git init
git add .
git commit -m "Initial commit: Quantum HALE Drone project setup"

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pip install pre-commit
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.10
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=120]
        
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-r", "src/"]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

pre-commit install

# Build Docker images
echo "Building Docker images..."
docker-compose build

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the Python environment: source venv/bin/activate"
echo "2. Start the simulation environment: docker-compose up -d"
echo "3. Run the example simulation: python scripts/run_simulations.py"
echo "4. Access Grafana dashboard: http://localhost:3000 (admin/quantum-hale-2024)"
echo "5. Access simulation logs: docker-compose logs -f quantum-hale-simulation"
echo ""
echo "For more information, see docs/installation.md" 