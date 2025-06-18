# Project Structure for Quantum-Compatible HALE Drone System
# ============================================================

# Directory Structure:
# quantum-hale-drone/
# ├── README.md
# ├── LICENSE
# ├── .gitignore
# ├── requirements.txt
# ├── setup.py
# ├── docker-compose.yml
# ├── Dockerfile.simulation
# ├── Dockerfile.development
# ├── .env.example
# ├── .github/
# │   └── workflows/
# │       ├── ci.yml
# │       └── security-scan.yml
# ├── docs/
# │   ├── architecture.md
# │   ├── api-reference.md
# │   ├── installation.md
# │   └── user-guide.md
# ├── src/
# │   ├── __init__.py
# │   ├── quantum_comms/
# │   │   ├── __init__.py
# │   │   ├── pqc_handshake.py
# │   │   ├── qkd_simulation.py
# │   │   └── crypto_utils.py
# │   ├── network_sim/
# │   │   ├── __init__.py
# │   │   ├── ns3_wrapper.py
# │   │   ├── rf_propagation.py
# │   │   ├── jamming_models.py
# │   │   └── mesh_routing.py
# │   ├── flight_sim/
# │   │   ├── __init__.py
# │   │   ├── hale_dynamics.py
# │   │   ├── autonomy_engine.py
# │   │   ├── sensor_fusion.py
# │   │   └── gazebo_interface.py
# │   ├── integration/
# │   │   ├── __init__.py
# │   │   ├── simulation_orchestrator.py
# │   │   ├── data_collector.py
# │   │   └── metrics_analyzer.py
# │   └── utils/
# │       ├── __init__.py
# │       ├── config.py
# │       ├── logging_setup.py
# │       └── performance_monitor.py
# ├── tests/
# │   ├── __init__.py
# │   ├── unit/
# │   │   ├── test_pqc_handshake.py
# │   │   ├── test_qkd_simulation.py
# │   │   └── test_network_simulation.py
# │   ├── integration/
# │   │   ├── test_end_to_end.py
# │   │   └── test_performance.py
# │   └── fixtures/
# │       ├── test_data.json
# │       └── simulation_configs.yaml
# ├── scripts/
# │   ├── setup_environment.sh
# │   ├── run_simulations.py
# │   ├── generate_reports.py
# │   └── deploy_containers.sh
# ├── configs/
# │   ├── simulation_params.yaml
# │   ├── pqc_settings.yaml
# │   ├── network_topology.yaml
# │   └── flight_missions.yaml
# ├── data/
# │   ├── simulation_results/
# │   ├── performance_logs/
# │   └── test_vectors/
# ├── models/
# │   ├── gazebo/
# │   │   ├── hale_drone.sdf
# │   │   └── environments/
# │   └── simulation/
# │       ├── rf_models/
# │       └── flight_dynamics/
# └── deployment/
#     ├── kubernetes/
#     │   ├── namespace.yaml
#     │   ├── simulation-deployment.yaml
#     │   └── monitoring-stack.yaml
#     ├── docker/
#     │   ├── production.dockerfile
#     │   └── testing.dockerfile
#     └── scripts/
#         ├── deploy.sh
#         └── monitoring_setup.sh

---

    # Docker Compose Configuration
    version: '3.8'
    
    services:
      # Main simulation environment
      quantum-hale-sim:
        build:
          context: .
          dockerfile: Dockerfile.simulation
        container_name: quantum-hale-simulation
        volumes:
          - ./src:/workspace/src:rw
          - ./configs:/workspace/configs:ro
          - ./data:/workspace/data:rw
          - ./models:/workspace/models:ro
          - /tmp/.X11-unix:/tmp/.X11-unix:rw
        environment:
          - DISPLAY=${DISPLAY}
          - PYTHONPATH=/workspace/src
          - GAZEBO_MODEL_PATH=/workspace/models/gazebo
          - ROS_DOMAIN_ID=42
        networks:
          - quantum-hale-net
        ports:
          - "8080:8080"  # Web dashboard
          - "9090:9090"  # Prometheus metrics
          - "3000:3000"  # Grafana
        privileged: true  # Required for Gazebo simulation
        
      # Network simulation service
      network-simulator:
        build:
          context: .
          dockerfile: Dockerfile.network-sim
        container_name: ns3-network-sim
        volumes:
          - ./src/network_sim:/workspace/network_sim:rw
          - ./data:/workspace/data:rw
        environment:
          - PYTHONPATH=/workspace
        networks:
          - quantum-hale-net
        depends_on:
          - influxdb
          
      # Quantum simulation service
      quantum-simulator:
        build:
          context: .
          dockerfile: Dockerfile.quantum-sim
        container_name: qiskit-quantum-sim
        volumes:
          - ./src/quantum_comms:/workspace/quantum_comms:rw
          - ./data:/workspace/data:rw
        environment:
          - PYTHONPATH=/workspace
          - QISKIT_SETTINGS=/workspace/configs/qiskit_settings.json
        networks:
          - quantum-hale-net
        depends_on:
          - redis
          
      # Database services
      influxdb:
        image: influxdb:2.7
        container_name: quantum-hale-influxdb
        environment:
          - DOCKER_INFLUXDB_INIT_MODE=setup
          - DOCKER_INFLUXDB_INIT_USERNAME=admin
          - DOCKER_INFLUXDB_INIT_PASSWORD=quantum-hale-2024
          - DOCKER_INFLUXDB_INIT_ORG=quantum-hale
          - DOCKER_INFLUXDB_INIT_BUCKET=simulation-data
        volumes:
          - influxdb-storage:/var/lib/influxdb2
          - ./configs/influxdb.conf:/etc/influxdb2/influxdb.conf:ro
        ports:
          - "8086:8086"
        networks:
          - quantum-hale-net
          
      redis:
        image: redis:7-alpine
        container_name: quantum-hale-redis
        volumes:
          - redis-storage:/data
        ports:
          - "6379:6379"
        networks:
          - quantum-hale-net
          
      # Monitoring and visualization
      grafana:
        image: grafana/grafana:10.2.0
        container_name: quantum-hale-grafana
        environment:
          - GF_SECURITY_ADMIN_PASSWORD=quantum-hale-2024
          - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
        volumes:
          - grafana-storage:/var/lib/grafana
          - ./configs/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
          - ./configs/grafana/datasources:/etc/grafana/provisioning/datasources:ro
        ports:
          - "3000:3000"
        networks:
          - quantum-hale-net
        depends_on:
          - influxdb
          
      prometheus:
        image: prom/prometheus:v2.47.0
        container_name: quantum-hale-prometheus
        volumes:
          - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
          - prometheus-storage:/prometheus
        command:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--web.console.libraries=/etc/prometheus/console_libraries'
          - '--web.console.templates=/etc/prometheus/consoles'
          - '--storage.tsdb.retention.time=200h'
          - '--web.enable-lifecycle'
        ports:
          - "9090:9090"
        networks:
          - quantum-hale-net
    
    networks:
      quantum-hale-net:
        driver: bridge
        ipam:
          config:
            - subnet: 172.20.0.0/16
    
    volumes:
      influxdb-storage:
      redis-storage:
      grafana-storage:
      prometheus-storage:
    
    ---
    
    # Dockerfile.simulation
    FROM ubuntu:22.04
    
    # Avoid interactive prompts during package installation
    ENV DEBIAN_FRONTEND=noninteractive
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        # Build tools
        build-essential cmake ninja-build git curl wget \
        # Python development
        python3.10 python3.10-dev python3-pip python3-venv \
        # ROS2 Humble
        software-properties-common \
        # Gazebo simulation
        gazebo11 gazebo11-plugin-base libgazebo11-dev \
        # Network tools
        net-tools tcpdump wireshark-common \
        # Graphics (for Gazebo GUI)
        xvfb x11-apps mesa-utils \
        # Additional utilities
        vim nano htop tmux \
        && rm -rf /var/lib/apt/lists/*
    
    # Install ROS2 Humble
    RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
        && apt-get update && apt-get install -y \
        ros-humble-desktop-full \
        ros-humble-gazebo-* \
        python3-colcon-common-extensions \
        && rm -rf /var/lib/apt/lists/*
    
    # Install PX4 dependencies
    RUN git clone https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
        && cd /opt/PX4-Autopilot \
        && bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
    
    # Install ns-3 network simulator
    RUN git clone https://gitlab.com/nsnam/ns-3-dev.git /opt/ns-3-dev \
        && cd /opt/ns-3-dev \
        && ./ns3 configure --enable-python-bindings \
        && ./ns3 build
    
    # Set up Python environment
    COPY requirements.txt /tmp/requirements.txt
    RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
    
    # Install additional quantum libraries
    RUN pip3 install --no-cache-dir \
        qiskit[all]==0.45.0 \
        qiskit-aer==0.13.0 \
        cirq==1.2.0 \
        pennylane==0.32.0 \
        liboqs-python==0.8.0
    
    # Create workspace
    WORKDIR /workspace
    RUN mkdir -p /workspace/{src,configs,data,models,logs}
    
    # Set environment variables
    ENV PYTHONPATH="/workspace/src:/opt/ns-3-dev/src/python"
    ENV GAZEBO_MODEL_PATH="/workspace/models/gazebo:/usr/share/gazebo-11/models"
    ENV ROS_DOMAIN_ID=42
    ENV GAZEBO_MASTER_URI=http://localhost:11345
    
    # Source ROS2 setup
    RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    RUN echo "source /opt/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash /opt/PX4-Autopilot /opt/PX4-Autopilot/build/px4_sitl_default" >> ~/.bashrc
    
    # Copy entrypoint script
    COPY scripts/docker-entrypoint.sh /usr/local/bin/
    RUN chmod +x /usr/local/bin/docker-entrypoint.sh
    
    EXPOSE 8080 9090 3000
    
    ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
    CMD ["bash"]
    
    ---
    
    # requirements.txt
    # Core dependencies for Quantum HALE Drone System
    
    # Quantum Computing & Cryptography
    qiskit[all]==0.45.0
    qiskit-aer==0.13.0
    qiskit-ignis==0.7.1
    cirq==1.2.0
    pennylane==0.32.0
    liboqs-python==0.8.0
    pycryptodome==3.19.0
    cryptography==41.0.7
    
    # Network Simulation & Analysis
    networkx==3.2.1
    numpy==1.24.4
    scipy==1.11.4
    matplotlib==3.8.2
    seaborn==0.13.0
    plotly==5.17.0
    dash==2.14.1
    simpy==4.0.1
    
    # Robotics & Flight Simulation
    pymavlink==2.4.37
    mavsdk==1.4.14
    dronekit==2.9.2
    pyproj==3.6.1
    geopy==2.4.0
    astropy==5.3.4
    
    # Data Processing & Storage
    pandas==2.1.4
    influxdb-client==1.38.0
    redis==5.0.1
    sqlalchemy==2.0.23
    
    # Configuration & Utilities
    pyyaml==6.0.1
    toml==0.10.2
    configparser==6.0.0
    python-dotenv==1.0.0
    
    # Logging & Monitoring
    loguru==0.7.2
    structlog==23.2.0
    prometheus-client==0.19.0
    
    # Parallel Processing
    multiprocessing-utils==0.5.0
    joblib==1.3.2
    ray[default]==2.8.1
    
    # Development & Testing
    pytest==7.4.3
    pytest-asyncio==0.21.1
    pytest-cov==4.1.0
    black==23.11.0
    flake8==6.1.0
    mypy==1.7.1
    bandit==1.7.5
    safety==2.3.5
    
    # Web Interface
    fastapi==0.104.1
    uvicorn==0.24.0
    websockets==12.0
    jinja2==3.1.2
    
    # Scientific Computing
    sympy==1.12
    control==0.9.4
    scikit-learn==1.3.2
    
    ---
    
    # .env.example
    # Environment configuration for Quantum HALE Drone System
    
    # Project Configuration
    PROJECT_NAME=quantum-hale-drone
    PROJECT_VERSION=1.0.0
    ENVIRONMENT=development
    
    # Simulation Parameters
    SIMULATION_DURATION=3600  # seconds
    SIMULATION_TIMESTEP=0.1   # seconds
    RANDOM_SEED=42
    
    # Network Configuration
    NETWORK_ID=HALE_NETWORK_001
    MESH_TOPOLOGY=star
    RF_FREQUENCY_BAND=2.4  # GHz
    JAMMING_POWER_DBM=50
    
    # Quantum Configuration
    SECURITY_LEVEL=3  # NIST Category 3
    QKD_KEY_LENGTH=256
    QUANTUM_FIDELITY_THRESHOLD=0.95
    
    # Database Configuration
    INFLUXDB_URL=http://influxdb:8086
    INFLUXDB_TOKEN=quantum-hale-token
    INFLUXDB_ORG=quantum-hale
    INFLUXDB_BUCKET=simulation-data
    
    REDIS_URL=redis://redis:6379/0
    
    # Monitoring Configuration
    GRAFANA_URL=http://grafana:3000
    PROMETHEUS_URL=http://prometheus:9090
    
    # Logging Configuration
    LOG_LEVEL=INFO
    LOG_FORMAT=json
    LOG_FILE=/workspace/logs/quantum-hale.log
    
    # Development Configuration
    DEBUG=true
    TESTING=false
    PROFILING=false
    
    ---
    
    # scripts/docker-entrypoint.sh
    #!/bin/bash
    set -e
    
    # Source ROS2 environment
    source /opt/ros/humble/setup.bash
    
    # Source PX4 environment
    source /opt/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash /opt/PX4-Autopilot /opt/PX4-Autopilot/build/px4_sitl_default
    
    # Set up X11 forwarding for Gazebo GUI
    export DISPLAY=${DISPLAY:-:0}
    
    # Create necessary directories
    mkdir -p /workspace/logs
    mkdir -p /workspace/data/simulation_results
    mkdir -p /workspace/data/performance_logs
    
    # Start Xvfb if no display is available
    if ! xset q &>/dev/null; then
        echo "Starting Xvfb for headless operation..."
        Xvfb :99 -screen 0 1024x768x24 &
        export DISPLAY=:99
    fi
    
    # Initialize logging
    echo "Quantum HALE Drone Simulation Environment"
    echo "========================================="
    echo "Timestamp: $(date)"
    echo "Workspace: /workspace"
    echo "Python Path: $PYTHONPATH"
    echo "ROS Domain ID: $ROS_DOMAIN_ID"
    echo "Gazebo Model Path: $GAZEBO_MODEL_PATH"
    echo ""
    
    # Execute the provided command or start bash
    exec "$@"
    
    ---
    
    # configs/simulation_params.yaml
    # Simulation configuration parameters
    
    simulation:
      name: "quantum_hale_baseline"
      duration: 3600  # seconds
      timestep: 0.1   # seconds
      random_seed: 42
      output_directory: "/workspace/data/simulation_results"
    
    environment:
      atmosphere:
        density_model: "exponential"
        wind_model: "turbulence"
        weather_effects: true
      
      terrain:
        type: "flat"
        altitude: 0  # meters MSL
        obstacles: []
    
    drones:
      - id: "DRONE_001"
        type: "hale_platform"
        initial_position: [0, 0, 20000]  # x, y, z in meters
        initial_velocity: [50, 0, 0]     # vx, vy, vz in m/s
        battery_capacity: 100000         # Wh
        payload_weight: 75               # kg
        
      - id: "DRONE_002"  
        type: "hale_platform"
        initial_position: [10000, 5000, 20000]
        initial_velocity: [50, 0, 0]
        battery_capacity: 100000
        payload_weight: 50
    
    ground_stations:
      - id: "GROUND_001"
        position: [0, 0, 100]  # x, y, z in meters
        communication_range: 200000  # meters
        
    missions:
      - drone_id: "DRONE_001"
        type: "isr_patrol"
        waypoints:
          - [0, 0, 20000]
          - [50000, 0, 20000]
          - [50000, 50000, 20000]
          - [0, 50000, 20000]
        patrol_duration: 7200  # seconds
        
    threats:
      jamming_sources:
        - position: [25000, 25000, 0]
          frequency_range: [2.4e9, 2.5e9]  # Hz
          power: 50  # dBm
          active_time: [1800, 3600]  # start, end in seconds
    
    ---
    
    # configs/pqc_settings.yaml
    # Post-Quantum Cryptography configuration
    
    algorithms:
      key_encapsulation:
        primary: "Kyber768"
        fallback: "Kyber512"
        
      digital_signature:
        primary: "Dilithium3"
        fallback: "Dilithium2"
        
      hash_function: "SHA3-256"
    
    security:
      category: 3  # NIST security category
      session_timeout: 3600  # seconds
      key_rotation_interval: 1800  # seconds
      max_handshake_retries: 3
      handshake_timeout: 5000  # milliseconds
    
    quantum_simulation:
      qkd_protocol: "BB84"
      key_length: 256  # bits
      fidelity_threshold: 0.95
      error_correction: true
      privacy_amplification: true
    
    performance:
      target_handshake_latency: 500  # milliseconds
      target_throughput: 1000000     # bits per second
      max_cpu_usage: 80              # percentage
      max_memory_usage: 512          # MB
    
    ---
    
    # .github/workflows/ci.yml
    name: Quantum HALE CI/CD
    
    on:
      push:
        branches: [ main, develop ]
      pull_request:
        branches: [ main ]
    
    jobs:
      test:
        runs-on: ubuntu-22.04
        
        services:
          redis:
            image: redis:7-alpine
            options: >-
              --health-cmd "redis-cli ping"
              --health-interval 10s
              --health-timeout 5s
              --health-retries 5
            ports:
              - 6379:6379
              
        steps:
        - uses: actions/checkout@v4
        
        - name: Set up Python 3.10
          uses: actions/setup-python@v4
          with:
            python-version: '3.10'
            
        - name: Cache pip dependencies
          uses: actions/cache@v3
          with:
            path: ~/.cache/pip
            key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
            restore-keys: |
              ${{ runner.os }}-pip-
              
        - name: Install system dependencies
          run: |
            sudo apt-get update
            sudo apt-get install -y build-essential cmake ninja-build
            
        - name: Install Python dependencies
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements.txt
            pip install pytest pytest-cov pytest-asyncio
            
        - name: Install liboqs for PQC
          run: |
            git clone https://github.com/open-quantum-safe/liboqs.git
            cd liboqs
            mkdir build && cd build
            cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local ..
            ninja install
            
        - name: Run unit tests
          run: |
            pytest tests/unit/ -v --cov=src --cov-report=xml
            
        - name: Run integration tests
          run: |
            pytest tests/integration/ -v
            
        - name: Security scan
          run: |
            bandit -r src/ -f json -o bandit-report.json
            safety check --json --output safety-report.json
            
        - name: Code quality checks
          run: |
            flake8 src/ --max-line-length=120 --statistics
            black --check src/
            mypy src/ --ignore-missing-imports
            
        - name: Upload coverage reports
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml
            flags: unittests
            
      build-docker:
        runs-on: ubuntu-22.04
        needs: test
        
        steps:
        - uses: actions/checkout@v4
        
        - name: Set up Docker Buildx
          uses: docker/setup-buildx-action@v3
          
        - name: Build simulation image
          run: |
            docker build -f Dockerfile.simulation -t quantum-hale:${{ github.sha }} .
            
        - name: Test Docker image
          run: |
            docker run --rm quantum-hale:${{ github.sha }} python3 -c "import src.quantum_comms.pqc_handshake; print('Import successful')"
    
    ---
    
    # Setup script for development environment
    # scripts/setup_environment.sh
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
    cp .env.example .env
    
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
    
    echo "✅ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the Python environment: source venv/bin/activate"
    echo "2. Start the simulation environment: docker-compose up -d"
    echo "3. Run the example simulation: python scripts/run_simulations.py"
    echo "4. Access Grafana dashboard: http://localhost:3000 (admin/quantum-hale-2024)"
    echo "5. Access simulation logs: docker-compose logs -f quantum-hale-simulation"
    echo ""
    echo "For more information, see docs/installation.md"