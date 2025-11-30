HYDRION — RESEARCH-GRADE DIGITAL TWIN & SAFE RL ENVIRONMENT
==========================================================

Hydrion is a high-fidelity 2D digital twin of the Hydrion Microfiber Extraction System. 
It integrates multilayer filtration physics, electrostatic capture, particle dynamics, 
optical sensor simulation, anomaly injection, safe reinforcement learning, and a 
real-time visualization engine.

This project is engineered for serious research in:
1. Safe RL
2. Digital twin development
3. Microfluidic simulation
4. Optical sensing
5. Robust control
6. Environmental technology
7. Fault-tolerant systems

Hydrion is not a toy model. It is a modular, configurable, research-grade RL 
environment aligned with real hardware principles and future sim-to-real calibration.

--------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------

Hydrion simulates the full operational flow of a real microfiber capture device:
- Water inflow from washing machine
- Polarization of microplastics
- Multi-mesh filtration (500 → 100 → 5 µm)
- Electrostatic post-capture by electronode array
- Storage and upcycling chambers
- Sensor fusion from optical and physical instruments
- Safe-RL decision making (pump, valve, backflush, node voltages)
- 2D animated rendering of flow, particles, meshes, and fields
- Anomaly scenarios
- Full data logging and analytics

Hydrion enables:
- Digital-twin prototyping
- Safe RL controller development
- Simulation-based system validation
- Visualization of capture dynamics
- Research into anomaly handling and robust control

--------------------------------------------------
MAJOR FEATURES
--------------------------------------------------

DIGITAL TWIN PHYSICS
- Hydraulic pressure & flow modeling
- Nonlinear pump and valve characteristics
- Tri-mesh clog accumulation (M1, M2, M3)
- Population-level microfiber dynamics (size bins)
- Electrostatic node capture modeling
- Upcycling chamber logic

SENSOR SIMULATION
- Optical microplastic cameras
- Pressure sensors
- Flow meters
- Sensor noise, drift, occlusion, turbidity effects
- Camera-based concentration estimation
- Sensor fusion into RL-friendly observation vector

SAFE REINFORCEMENT LEARNING
- Action safety shield
- Hard constraint termination
- Pressure, flow, and voltage safety limits
- Smoothness penalties
- Energy-aware actuation
- Compatible with PPO, SAC, TD3, A2C, DDPG

ANOMALY INJECTION
- Shedding bursts
- Foam occlusion events
- Sensor drift
- Pump degradation
- Valve sticking
- Uneven mesh clogging
- Node voltage sag
- Outflow blockage events

HIGH-FIDELITY 2D RENDERER
- Cutaway diagram of full Hydrion system
- Animated particle flow
- Transparent mesh layers
- Electrostatic glow visualization
- Flow arrows & pressure bars
- Storage and upcycling chamber display
- RL action overlay and reward HUD
- Camera feed visualization

--------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------

Hydrion/
  hydrion/ (core package)
    env.py
    config.py
    physics/ (hydraulics, clogging, particles, electrostatics, upcycling)
    sensors/ (optical cameras, pressure/flow sensors, fusion)
    anomalies/ (events, scheduler)
    safety/ (constraints, shield)
    rendering/ (renderer, assets)
    utils/ (math and logging tools)
  agents/ (PPO and baseline controllers)
  scripts/ (training, evaluation, visualization)
  configs/ (YAML configuration files)
  data/ (logs and sensor dumps)
  models/ (RL models)
  tests/ (unit and integration tests)
  notebooks/ (analysis notebooks)
  requirements.txt
  README.txt
  .gitignore

--------------------------------------------------
INSTALLATION
--------------------------------------------------

1. Clone the repository:
   git clone https://github.com/<your-username>/Hydrion.git

2. Create and activate a virtual environment:
   python -m venv .venv
   .venv\Scripts\activate   (Windows)
   source .venv/bin/activate (macOS/Linux)

3. Install dependencies:
   pip install -r requirements.txt

4. (Optional) Install package locally:
   pip install -e .

--------------------------------------------------
USAGE
--------------------------------------------------

Train PPO:
   python scripts/train_ppo.py

Evaluate agent:
   python scripts/evaluate_ppo.py

Visualize simulation:
   python scripts/visualize_rollout.py

Create environment manually in Python:
   from hydrion.env import HydrionEnv
   env = HydrionEnv()
   obs, info = env.reset()
   env.render()

--------------------------------------------------
CONFIGURATION
--------------------------------------------------

Hydrion is fully config-driven. YAML files control:
- Physics parameters
- Mesh sizes and clog dynamics
- Node voltages
- Particle size distributions
- Sensor noise and drift
- Anomaly rates
- Reward shaping
- Rendering details

--------------------------------------------------
ROADMAP
--------------------------------------------------

Current version:
- High-fidelity filtration physics
- Particle clouds
- Safe RL
- 2D rendering
- Cameras and sensors
- Anomalies
- Training scripts
- Test suite

Future extensions:
- Full 3D visualization (Omniverse/Unreal)
- Sim-to-real calibration
- Advanced optical modeling
- Multi-agent hybrid controllers
- Hardware integration

--------------------------------------------------
PROJECT PURPOSE
--------------------------------------------------

Hydrion supports the advancement of environmental microplastic filtration, 
intelligent self-maintenance systems, and circular upcycling pipelines. It is built 
to help engineers, researchers, and data scientists develop robust RL-driven 
control solutions under realistic physical constraints and anomaly behaviors.

This project aims to enable:
- Cleaner water systems
- Sustainable microplastic capture
- Safe and intelligent filtration control
- Digital-twin prototyping for environmental hardware

--------------------------------------------------
END OF README
--------------------------------------------------
