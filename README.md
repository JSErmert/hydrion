# HYDRION — Research-Grade Digital Twin & Safe RL Environment

HYDRION SETUP AND RUN GUIDE

1. SYSTEM REQUIREMENTS
- Python 3.9–3.11
- pip
- VS Code (recommended)
- ffmpeg (required for MP4 video export)

Install ffmpeg:
1. Download "ffmpeg-release-essentials.zip" from:
   https://www.gyan.dev/ffmpeg/builds/
2. Extract to: C:\ffmpeg
3. Ensure C:\ffmpeg\bin contains ffmpeg.exe
4. Add C:\ffmpeg\bin to Windows PATH:
   - Search "Environment Variables"
   - Edit system environment variables
   - Environment Variables...
   - System Variables → Path → Edit → New
   - Add: C:\ffmpeg\bin
5. Restart VS Code
6. Test: ffmpeg -version

------------------------------------------

2. CLONE THE REPOSITORY
git clone <repo-url>
cd hydrion

------------------------------------------

3. CREATE PYTHON ENVIRONMENT
python -m venv .venv

Activate:

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

------------------------------------------

4. INSTALL DEPENDENCIES
pip install -r requirements.txt

requirements.txt should include:
gymnasium>=0.29.1
numpy>=1.24
pyyaml>=6.0
stable-baselines3[extra]>=2.1.0
tensorboard>=2.12
matplotlib>=3.6
pandas>=2.0
opencv-python>=4.7
scikit-image>=0.21
pyqt5   (optional for GUI backends)

------------------------------------------

5. REPOSITORY STRUCTURE
hydrion/
  env.py                  # Main RL environment (12D multi-physics)
  config.py
  physics/
     hydraulics.py
     clogging.py
     electrostatics.py
     particles.py
  sensors/
     optical.py
  rendering/
     viz2d.py
  utils/
     episode_recorder.py
  tests/
     test_env_api.py
     test_hydraulics.py
     test_clogging.py
     test_electrostatics.py
     test_particles.py
     test_sensors.py
  train_ppo.py
  eval_ppo.py
  make_video.py          # Creates vertical reactor MP4 animation
  checkpoints/
  videos/
  runs/

------------------------------------------

6. UNIT TESTS
Run these to ensure everything works:

python -m tests.test_env_api
python -m tests.test_hydraulics
python -m tests.test_clogging
python -m tests.test_electrostatics
python -m tests.test_particles
python -m tests.test_sensors

------------------------------------------

7. TRAIN PPO
python -m hydrion.train_ppo

Outputs:
  ppo_hydrion_final_12d.zip
  ppo_hydrion_vecnormalize_12d.pkl

------------------------------------------

8. EVALUATE PPO
python -m hydrion.eval_ppo

Expected:
  returns ≈ 2981
  std ≈ 0.001

------------------------------------------

9. VISUALIZE PHYSICS
python -m hydrion.visualize_episode

------------------------------------------

10. GENERATE DIGITAL TWIN VIDEO (MP4)
python -m hydrion.make_video

Creates:
videos/hydrion_run.mp4

------------------------------------------

11. TROUBLESHOOTING

FFmpeg not found:
  Ensure C:\ffmpeg\bin is added to PATH.

VecNormalize errors:
  Ensure training produced:
    ppo_hydrion_vecnormalize_12d.pkl

Matplotlib backend issues:
  pip install pyqt5

------------------------------------------

12. SIMPLIFIED TASKS CHECKLIST

Task 1 — Setup  
Task 2 — Run Tests  
Task 3 — Train PPO  
Task 4 — Visualization  
Task 5 — MP4 Generation  
Task 6 — Development  

------------------------------------------
END OF FILE