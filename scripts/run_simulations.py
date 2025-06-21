#!/usr/bin/env python3
"""
Quantum HALE Drone Simulation Runner

This script runs the main simulation scenarios for the Quantum HALE Drone System.
"""

import argparse
import logging
import sys
import time
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from integration.simulation_orchestrator import SimulationOrchestrator, SimulationConfig
from flight_sim.hale_dynamics import AircraftParameters
from flight_sim.autonomy_engine import Waypoint, MissionType
from quantum_comms.pqc_handshake import SecurityLevel


def setup_logging(level: str = "INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log')
        ]
    )


def load_config(config_file: str) -> dict:
    """Load simulation configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def parse_simulation_config(config_data: dict) -> SimulationConfig:
    """Parse the simulation config from a dictionary."""
    sim_params = config_data.get('simulation', {})
    env_params = config_data.get('environment', {})
    drone_params = config_data.get('drones', [])
    gs_params = config_data.get('ground_stations', [])
    mission_params = config_data.get('missions', [])
    threat_params = config_data.get('threats', {})

    # For now, we'll configure for the first drone and mission
    # A more robust implementation would handle multiple drones and missions
    main_drone = drone_params[0] if drone_params else {}
    main_mission = mission_params[0] if mission_params else {}

    aircraft_params = AircraftParameters(
        wingspan=35.0,
        wing_area=45.0,
        length=15.0,
        mass_empty=1200.0,
        mass_max_takeoff=2500.0,
        cl_alpha=5.0,
        cd0=0.02,
        oswald_efficiency=0.85,
        aspect_ratio=27.0,
        thrust_max=5000.0,
        specific_fuel_consumption=0.0001,
        propeller_efficiency=0.8,
        elevator_effectiveness=0.1,
        aileron_effectiveness=0.1,
        rudder_effectiveness=0.1,
        stall_speed=25.0,
        max_speed=120.0,
        service_ceiling=20000.0,
        range_max=500000.0,
        endurance_max=86400.0
    )

    waypoints = [Waypoint(latitude=w[0], longitude=w[1], altitude=w[2], speed=50.0) for w in main_mission.get('waypoints', [])]

    ground_stations = [
        (gs['position'][0], gs['position'][1]) for gs in gs_params
    ]

    sim_config = SimulationConfig(
        duration=sim_params.get('duration', 3600),
        timestep=sim_params.get('timestep', 0.1),
        real_time_factor=sim_params.get('real_time_factor', 1.0),
        aircraft_params=aircraft_params,
        mission_type=MissionType(main_mission.get('type', 'isr_patrol')),
        waypoints=waypoints,
        num_drones=len(drone_params),
        ground_stations=ground_stations,
        security_level=SecurityLevel.CATEGORY_3, # This could be made configurable
        enable_qkd=True, # This could be made configurable
        jamming_sources=threat_params.get('jamming_sources', []),
        output_directory=sim_params.get('output_directory', 'simulation_results')
    )
    return sim_config


def run_full_simulation(config: dict):
    """Run full integrated simulation using the SimulationOrchestrator."""
    print("Running Full Integrated Simulation via Orchestrator")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create SimulationConfig from the loaded dictionary
    try:
        sim_config = parse_simulation_config(config)
        logging.info("Successfully parsed simulation configuration.")
    except Exception as e:
        logging.error(f"Failed to parse simulation config: {e}")
        return

    # Initialize and run the orchestrator
    orchestrator = SimulationOrchestrator(sim_config)
    
    if not orchestrator.initialize():
        logging.error("Failed to initialize simulation orchestrator.")
        return

    if orchestrator.start_simulation():
        logging.info(f"Simulation started. Will run for {sim_config.duration} seconds.")
        try:
            # Wait for the simulation to complete
            while orchestrator.get_simulation_status()['state'] == 'running':
                time.sleep(1)
                # Here you could add some progress reporting
                sim_time = orchestrator.get_simulation_status().get('simulation_time', 0)
                print(f"\rSimulation time: {sim_time:.2f}s / {sim_config.duration}s", end="")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            print("\nStopping simulation...")
            orchestrator.stop_simulation()
            
            # Save simulation data
            output_file = Path(sim_config.output_directory) / f"simulation_run_{int(time.time())}.json"
            orchestrator.save_simulation_data(str(output_file))
            logging.info(f"Simulation data saved to {output_file}")

    else:
        logging.error("Failed to start simulation.")

    end_time = time.time()
    print(f"\n✅ Full simulation orchestration finished in {end_time - start_time:.2f} seconds")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quantum HALE Drone Simulation Runner")
    parser.add_argument("--config", "-c", default="configs/simulation_params.yaml",
                       help="Configuration file path")
    parser.add_argument("--test", "-t", choices=["full"],
                       default="full", help="Test to run (only 'full' is supported with orchestrator)")
    parser.add_argument("--duration", "-d", type=int, default=None,
                       help="Override simulation duration in seconds")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override duration if provided
    if args.duration is not None:
        config['simulation']['duration'] = args.duration
    
    # Run selected test
    if args.test == "full":
        run_full_simulation(config)
    else:
        logging.error(f"Test type '{args.test}' is not supported in this version.")
        print("This script now uses the SimulationOrchestrator. Please use '--test full'.")
        sys.exit(1)


if __name__ == "__main__":
    main() 