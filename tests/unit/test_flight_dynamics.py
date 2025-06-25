"""
Unit Tests for Flight Dynamics and Control Systems
==================================================

This module contains comprehensive unit tests for the HALE drone flight dynamics,
PID controllers, and Gazebo interface components.

Author: Quantum HALE Development Team
License: MIT
"""

import unittest
import numpy as np
import time
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from flight_sim.hale_dynamics import HALEDynamics, AircraftState, AircraftParameters, AtmosphericModel
from flight_sim.pid_controller import PIDController, PIDGains, ControlMode, FlightController
from flight_sim.gazebo_interface import GazeboInterface, GazeboModelConfig


class TestAtmosphericModel(unittest.TestCase):
    """Test atmospheric modeling for high-altitude operations"""
    
    def setUp(self):
        self.atmosphere = AtmosphericModel()
        
    def test_troposphere_conditions(self):
        """Test atmospheric conditions in troposphere"""
        conditions = self.atmosphere.get_atmospheric_conditions(5000)  # 5km altitude
        
        self.assertGreater(conditions['temperature'], 250)  # Should be above 250K
        self.assertLess(conditions['temperature'], 290)     # Should be below 290K
        self.assertGreater(conditions['pressure'], 50000)   # Should be above 50kPa
        self.assertLess(conditions['pressure'], 101325)     # Should be below sea level
        self.assertGreater(conditions['density'], 0.5)      # Should be above 0.5 kg/m³
        self.assertLess(conditions['density'], 1.225)       # Should be below sea level
        
    def test_stratosphere_conditions(self):
        """Test atmospheric conditions in stratosphere"""
        conditions = self.atmosphere.get_atmospheric_conditions(20000)  # 20km altitude
        
        self.assertAlmostEqual(conditions['temperature'], 216.65, places=1)  # Constant temp
        self.assertLess(conditions['pressure'], 10000)     # Should be below 10kPa
        self.assertLess(conditions['density'], 0.1)        # Should be below 0.1 kg/m³
        
    def test_wind_model(self):
        """Test wind model at different altitudes"""
        wind_north, wind_east, wind_up = self.atmosphere.get_wind_model(1000, 0.0, 0.0)
        
        # Wind should be reasonable values
        self.assertGreater(abs(wind_north), 0)
        self.assertGreater(abs(wind_east), 0)
        self.assertEqual(wind_up, 0.0)  # Vertical wind should be small
        
        # High altitude should have stronger winds
        wind_north_high, wind_east_high, _ = self.atmosphere.get_wind_model(15000, 0.0, 0.0)
        self.assertGreater(abs(wind_north_high), abs(wind_north))


class TestPIDController(unittest.TestCase):
    """Test PID controller functionality"""
    
    def setUp(self):
        self.gains = PIDGains(kp=1.0, ki=0.1, kd=0.05, output_limit=1.0)
        self.controller = PIDController(self.gains, ControlMode.POSITION)
        
    def test_basic_pid_response(self):
        """Test basic PID controller response"""
        # Set setpoint
        self.controller.set_setpoint(10.0)
        
        # Simulate step response
        outputs = []
        for i in range(100):
            output = self.controller.update(5.0, 0.01)  # Constant measurement
            outputs.append(output)
            
        # Should show typical PID response
        self.assertGreater(outputs[-1], outputs[0])  # Output should increase
        self.assertLessEqual(outputs[-1], self.gains.output_limit)  # Should respect limits
        
    def test_anti_windup(self):
        """Test integral anti-windup protection"""
        # Set very high setpoint
        self.controller.set_setpoint(1000.0)
        
        # Simulate with constant measurement
        for i in range(1000):
            output = self.controller.update(0.0, 0.01)
            
        # Output should be limited
        self.assertLessEqual(abs(output), self.gains.output_limit)
        
    def test_deadband(self):
        """Test deadband functionality"""
        gains_with_deadband = PIDGains(kp=1.0, ki=0.0, kd=0.0, deadband=0.5)
        controller = PIDController(gains_with_deadband, ControlMode.POSITION)
        controller.set_setpoint(10.0)
        
        # Small error should be ignored
        output = controller.update(9.6, 0.01)  # Error = 0.4 < deadband
        self.assertEqual(output, 0.0)
        
        # Large error should be processed
        output = controller.update(8.0, 0.01)  # Error = 2.0 > deadband
        self.assertNotEqual(output, 0.0)


class TestFlightController(unittest.TestCase):
    """Test multi-loop flight controller"""
    
    def setUp(self):
        self.controller = FlightController()
        
    def test_control_mode_setting(self):
        """Test flight control mode setting"""
        self.controller.set_control_mode("altitude_hold")
        self.assertEqual(self.controller.control_mode, "altitude_hold")
        
        self.controller.set_control_mode("auto")
        self.assertEqual(self.controller.control_mode, "auto")
        
    def test_manual_controls(self):
        """Test manual control inputs"""
        self.controller.set_manual_controls(0.5, 0.1, -0.2, 0.05)
        
        self.assertEqual(self.controller.controls['throttle'], 0.5)
        self.assertEqual(self.controller.controls['elevator'], 0.1)
        self.assertEqual(self.controller.controls['aileron'], -0.2)
        self.assertEqual(self.controller.controls['rudder'], 0.05)
        
    def test_control_limits(self):
        """Test control input limits"""
        # Test values outside limits
        self.controller.set_manual_controls(1.5, 1.0, -1.0, 0.5)
        
        self.assertEqual(self.controller.controls['throttle'], 1.0)  # Clipped to max
        self.assertEqual(self.controller.controls['elevator'], 0.5)  # Clipped to max
        self.assertEqual(self.controller.controls['aileron'], -0.5)  # Clipped to min
        self.assertEqual(self.controller.controls['rudder'], 0.3)    # Clipped to max


class TestHALEDynamics(unittest.TestCase):
    """Test HALE drone flight dynamics"""
    
    def setUp(self):
        # Create realistic aircraft parameters
        self.params = AircraftParameters(
            wingspan=25.0,
            wing_area=31.5,
            length=7.5,
            mass_empty=75.0,
            mass_max_takeoff=140.0,
            cl_alpha=5.73,
            cd0=0.015,
            oswald_efficiency=0.85,
            aspect_ratio=19.8,
            thrust_max=800.0,
            specific_fuel_consumption=0.0,
            propeller_efficiency=0.85,
            elevator_effectiveness=0.12,
            aileron_effectiveness=0.10,
            rudder_effectiveness=0.08,
            stall_speed=12.0,
            max_speed=35.0,
            service_ceiling=70000.0,
            range_max=5000000.0,
            endurance_max=2592000.0
        )
        
        self.dynamics = HALEDynamics(self.params)
        
    def test_initialization(self):
        """Test dynamics initialization"""
        self.assertIsNotNone(self.dynamics)
        self.assertEqual(self.dynamics.params, self.params)
        self.assertIsNone(self.dynamics.state)  # State not initialized yet
        
    def test_state_initialization(self):
        """Test aircraft state initialization"""
        initial_state = AircraftState(
            latitude=np.radians(0.0),
            longitude=np.radians(0.0),
            altitude=20000.0,
            velocity_north=50.0,
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        
        self.dynamics.initialize_state(initial_state)
        self.assertIsNotNone(self.dynamics.state)
        self.assertEqual(self.dynamics.state.altitude, 20000.0)
        
    def test_aerodynamic_forces(self):
        """Test aerodynamic force calculation"""
        # Initialize state first
        initial_state = AircraftState(
            latitude=np.radians(0.0),
            longitude=np.radians(0.0),
            altitude=20000.0,
            velocity_north=50.0,
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        self.dynamics.initialize_state(initial_state)
        
        # Calculate aerodynamic forces
        forces, moments = self.dynamics.calculate_aerodynamic_forces()
        
        # Should return 3D vectors
        self.assertEqual(len(forces), 3)
        self.assertEqual(len(moments), 3)
        
        # Forces should be reasonable values
        self.assertGreater(abs(forces[0]), 0)  # Drag
        self.assertGreater(abs(forces[2]), 0)  # Lift
        
    def test_propulsion_forces(self):
        """Test propulsion force calculation"""
        # Initialize state
        initial_state = AircraftState(
            latitude=np.radians(0.0),
            longitude=np.radians(0.0),
            altitude=20000.0,
            velocity_north=50.0,
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        self.dynamics.initialize_state(initial_state)
        
        # Set throttle
        self.dynamics.set_controls(0.5, 0.0, 0.0, 0.0)
        
        # Calculate propulsion forces
        thrust = self.dynamics.calculate_propulsion_forces()
        
        # Should return 3D vector
        self.assertEqual(len(thrust), 3)
        
        # Thrust should be in forward direction
        self.assertGreater(thrust[0], 0)  # Forward thrust
        self.assertEqual(thrust[1], 0.0)  # No side thrust
        self.assertEqual(thrust[2], 0.0)  # No vertical thrust
        
    def test_flight_envelope_protection(self):
        """Test flight envelope protection"""
        # Initialize state with valid airspeed
        initial_state = AircraftState(
            latitude=np.radians(0.0),
            longitude=np.radians(0.0),
            altitude=20000.0,
            velocity_north=30.0,  # Use 30 m/s instead of 50 m/s
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=30.0, ground_speed=30.0, heading=0.0, flight_path_angle=0.0,  # Update airspeed
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        self.dynamics.initialize_state(initial_state)
        
        # Check envelope
        envelope = self.dynamics.check_flight_envelope()
        self.assertTrue(envelope['within_envelope'])
        
        # Test altitude limit violation
        self.dynamics.state.altitude = 80000.0  # Above service ceiling
        envelope = self.dynamics.check_flight_envelope()
        self.assertFalse(envelope['within_envelope'])
        self.assertIn('Above service ceiling', envelope['warnings'])
        
    def test_telemetry_data(self):
        """Test telemetry data generation"""
        # Initialize state
        initial_state = AircraftState(
            latitude=np.radians(0.0),
            longitude=np.radians(0.0),
            altitude=20000.0,
            velocity_north=50.0,
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        self.dynamics.initialize_state(initial_state)
        
        # Get telemetry
        telemetry = self.dynamics.get_telemetry()
        
        # Should contain expected fields
        self.assertIn('position', telemetry)
        self.assertIn('velocity', telemetry)
        self.assertIn('attitude', telemetry)
        self.assertIn('energy', telemetry)
        self.assertIn('flight_phase', telemetry)
        self.assertIn('controls', telemetry)
        self.assertIn('performance', telemetry)
        self.assertIn('environment', telemetry)


class TestGazeboInterface(unittest.TestCase):
    """Test Gazebo interface functionality"""
    
    def setUp(self):
        # Create model configuration
        self.model_config = GazeboModelConfig(
            model_name="test_drone",
            sdf_file="models/test_drone.sdf",
            initial_pose=(0.0, 0.0, 20000.0, 0.0, 0.0, 0.0),
            aircraft_type="zephyr_s"
        )
        
        self.interface = GazeboInterface(self.model_config)
        
    def test_initialization(self):
        """Test interface initialization"""
        self.assertIsNotNone(self.interface)
        self.assertEqual(self.interface.model_name, "test_drone")
        self.assertIsNotNone(self.interface.flight_dynamics)
        self.assertIsNotNone(self.interface.flight_controller)
        
    def test_aircraft_parameters_loading(self):
        """Test aircraft parameters loading"""
        # Should load parameters from config file
        self.assertIsNotNone(self.interface.aircraft_params)
        self.assertEqual(self.interface.aircraft_params.wingspan, 25.0)
        self.assertEqual(self.interface.aircraft_params.mass_max_takeoff, 140.0)
        
    def test_control_inputs(self):
        """Test control input setting"""
        self.interface.set_control_inputs(0.5, 0.1, -0.2, 0.05)
        
        self.assertEqual(self.interface.control_inputs['throttle'], 0.5)
        self.assertEqual(self.interface.control_inputs['elevator'], 0.1)
        self.assertEqual(self.interface.control_inputs['aileron'], -0.2)
        self.assertEqual(self.interface.control_inputs['rudder'], 0.05)
        
    def test_flight_mode_setting(self):
        """Test flight mode setting"""
        self.interface.set_flight_mode("altitude_hold")
        self.assertEqual(self.interface.flight_mode, "altitude_hold")
        self.assertEqual(self.interface.flight_controller.control_mode, "altitude_hold")
        
    def test_simulation_status(self):
        """Test simulation status reporting"""
        status = self.interface.get_simulation_status()
        
        self.assertIn('simulation_time', status)
        self.assertIn('model_loaded', status)
        self.assertIn('flight_mode', status)
        self.assertIn('aircraft_type', status)
        self.assertIn('model_state', status)
        self.assertIn('sensor_data', status)
        self.assertIn('flight_controller_status', status)
        
    def test_simulation_step(self):
        """Test simulation step execution"""
        # Initialize with a valid state
        initial_state = AircraftState(
            latitude=np.radians(0.0),
            longitude=np.radians(0.0),
            altitude=20000.0,
            velocity_north=50.0,
            velocity_east=0.0,
            velocity_down=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=50.0, ground_speed=50.0, heading=0.0, flight_path_angle=0.0,
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        self.interface.flight_dynamics.initialize_state(initial_state)
        
        # Execute step
        initial_time = self.interface.simulation_time
        self.interface.step()
        
        # Time should advance
        self.assertGreater(self.interface.simulation_time, initial_time)
        
        # Should have sensor data
        sensor_data = self.interface.get_sensor_data()
        self.assertIsNotNone(sensor_data['imu'])
        self.assertIsNotNone(sensor_data['gps'])
        self.assertIsNotNone(sensor_data['air_data'])


class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def test_flight_dynamics_with_controller(self):
        """Test flight dynamics with PID controller integration"""
        # Create dynamics
        params = AircraftParameters(
            wingspan=25.0, wing_area=31.5, length=7.5,
            mass_empty=75.0, mass_max_takeoff=140.0,
            cl_alpha=5.73, cd0=0.015, oswald_efficiency=0.85, aspect_ratio=19.8,
            thrust_max=800.0, specific_fuel_consumption=0.0, propeller_efficiency=0.85,
            elevator_effectiveness=0.12, aileron_effectiveness=0.10, rudder_effectiveness=0.08,
            stall_speed=12.0, max_speed=35.0, service_ceiling=70000.0,
            range_max=5000000.0, endurance_max=2592000.0
        )
        dynamics = HALEDynamics(params)
        
        # Initialize state with valid airspeed
        initial_state = AircraftState(
            latitude=np.radians(0.0), longitude=np.radians(0.0), altitude=20000.0,
            velocity_north=30.0, velocity_east=0.0, velocity_down=0.0,  # Use 30 m/s
            roll=0.0, pitch=0.0, yaw=0.0, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0,
            airspeed=30.0, ground_speed=30.0, heading=0.0, flight_path_angle=0.0,  # Update airspeed
            total_energy=0.0, potential_energy=0.0, kinetic_energy=0.0,
            fuel_remaining=100000.0, fuel_consumption_rate=0.0,
            air_density=0.0889, temperature=216.65, pressure=5474.9,
            wind_north=0.0, wind_east=0.0, wind_up=0.0
        )
        dynamics.initialize_state(initial_state)
        
        # Set controller mode
        dynamics.set_flight_controller_mode("altitude_hold")
        
        # Run simulation for several steps
        for i in range(100):
            dynamics.step()
            
        # Should have advanced in time
        self.assertGreater(dynamics.time, 0.0)
        
        # Should have consumed some fuel
        self.assertLess(dynamics.state.fuel_remaining, 100000.0)
        
        # Should have traveled some distance
        self.assertGreater(dynamics.distance_traveled, 0.0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestAtmosphericModel))
    test_suite.addTest(unittest.makeSuite(TestPIDController))
    test_suite.addTest(unittest.makeSuite(TestFlightController))
    test_suite.addTest(unittest.makeSuite(TestHALEDynamics))
    test_suite.addTest(unittest.makeSuite(TestGazeboInterface))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}") 