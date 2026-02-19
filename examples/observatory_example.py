"""
examples/observatory_example.py

Example usage of HydrOS Observatory for research visualization.

This demonstrates how to:
1. Record an episode with Observatory
2. Generate comprehensive dashboard plots
3. Save frames for playback
4. Create video from frames
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from hydrion.env import HydrionEnv
from hydrion.rendering import Observatory
from hydrion.utils.visualization import record_episode_with_observatory


def main():
    """Run example episode with Observatory visualization."""
    
    # Create environment
    print("Creating HydrionEnv...")
    env = HydrionEnv(config_path="configs/default.yaml")
    
    # Create Observatory
    output_dir = Path("outputs/observatory_example")
    observatory = Observatory(
        save_dir=output_dir,
        time_axis="time",  # Use time instead of step
    )
    
    # Run episode and record
    print("Running episode...")
    observations, rewards, terminated, truncated, info = record_episode_with_observatory(
        env=env,
        observatory=observatory,
        policy=None,  # Random policy
        max_steps=1000,
    )
    
    print(f"Episode completed: terminated={terminated}, truncated={truncated}")
    print(f"Total steps: {len(observatory)}")
    print(f"Total reward: {sum(rewards):.2f}")
    
    # Generate dashboard
    print("Generating dashboard plots...")
    figures = observatory.plot_dashboard(save=True, show=False)
    print(f"Generated {len(figures)} plots:")
    for name in figures.keys():
        print(f"  - {name}")
    
    # Get anomaly summary
    anomaly_summary = observatory.get_anomaly_summary()
    print("\nAnomaly Summary:")
    print(f"  Anomalies detected: {anomaly_summary.get('anomalies', False)}")
    if anomaly_summary.get('anomalies'):
        print(f"  Total count: {anomaly_summary.get('count', 0)}")
        print(f"  By type: {anomaly_summary.get('by_type', {})}")
    
    # Save frames for playback
    print("\nSaving frames...")
    observatory.save_frames(
        variables=["flow", "pressure", "clog", "E_norm", "C_out"],
        every_n_steps=5,  # Save every 5th step
    )
    
    # Create video from frames (optional, requires ffmpeg)
    try:
        print("Creating video from frames...")
        video_path = observatory.create_video_from_frames(fps=10)
        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Video creation failed (may require ffmpeg): {e}")
    
    # Example: Custom time series plot
    print("\nGenerating custom time series plot...")
    observatory.plot_custom_time_series(
        variables=["Q_out_Lmin", "P_in", "P_m1", "P_m2", "P_m3"],
        save=True,
        show=False,
        filename="pressure_traces.png",
    )
    
    print("\n✅ Observatory example complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
