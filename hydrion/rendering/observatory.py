"""
hydrion/rendering/observatory.py

Research Observatory Dashboard for HydrOS.
Main entry point for visualization and observability.

Identity: Research Observatory (not consumer UI)
- Makes internal dynamics legible
- Makes anomalies obvious
- Makes RL behavior interpretable
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .episode_history import EpisodeHistory
from .time_series import (
    plot_time_series,
    plot_actions,
    plot_psd_time_series,
    plot_psd_summary,
    plot_shape_variables,
    plot_reward_trace,
)
from .anomaly_detector import AnomalyDetector, plot_anomalies, plot_shield_events


class Observatory:
    """
    Research Observatory Dashboard for HydrOS.
    
    Provides comprehensive visualization of:
    - Time-series plots for key truth variables
    - Action traces synchronized with state
    - PSD observability (if enabled)
    - Shape observability (if enabled)
    - Anomaly visibility
    - Episode playback support
    
    All visualization is side-effect free and does not modify simulation state.
    """
    
    def __init__(
        self,
        save_dir: Optional[str | Path] = None,
        time_axis: str = "time",
        figsize: Tuple[float, float] = (12, 6),
    ):
        """
        Initialize Observatory.
        
        Args:
            save_dir: Optional directory to save plots/frames
            time_axis: "time" or "step" for x-axis
            figsize: Default figure size (width, height)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.time_axis = time_axis
        self.figsize = figsize
        self.history: Optional[EpisodeHistory] = None
        self.detector: Optional[AnomalyDetector] = None
    
    def reset(self):
        """Reset observatory state."""
        self.history = EpisodeHistory()
        self.detector = None
    
    def record_step(
        self,
        step: int,
        truth_state: Dict[str, Any],
        sensor_state: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any],
        observation: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ):
        """
        Record a single step of episode data.
        
        Args:
            step: Step number (0-indexed)
            truth_state: Physics truth state dict
            sensor_state: Sensor state dict
            action: Action array [valve, pump, backflush, node_voltage]
            reward: Reward value
            info: Info dict
            observation: Optional 12D observation vector
            dt: Time step duration
        """
        if self.history is None:
            self.history = EpisodeHistory()
        
        self.history.record_step(
            step=step,
            truth_state=truth_state,
            sensor_state=sensor_state,
            action=action,
            reward=reward,
            info=info,
            observation=observation,
            dt=dt,
        )
    
    def finalize_episode(self, terminated: bool = False, truncated: bool = False):
        """Mark episode as complete and run anomaly detection."""
        if self.history is None:
            return
        
        self.history.finalize(terminated=terminated, truncated=truncated)
        self.detector = AnomalyDetector(self.history)
    
    def plot_dashboard(
        self,
        save: bool = False,
        show: bool = True,
        key_variables: Optional[List[str]] = None,
    ) -> Dict[str, Figure]:
        """
        Generate comprehensive dashboard with all visualizations.
        
        Args:
            save: Whether to save figures to save_dir
            show: Whether to display figures
            key_variables: Optional list of key variables to plot (defaults to core set)
        
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        if self.history is None or len(self.history) == 0:
            raise ValueError("No episode data recorded. Call record_step() first.")
        
        if self.detector is None:
            self.finalize_episode()
        
        figures = {}
        
        # Default key variables
        if key_variables is None:
            key_variables = ["flow", "pressure", "clog", "E_norm", "C_out", "particle_capture_eff"]
        
        # 1. Core time-series plots
        fig, _ = plot_time_series(
            self.history,
            variables=key_variables,
            time_axis=self.time_axis,
            figsize=self.figsize,
        )
        figures["core_time_series"] = fig
        if save and self.save_dir:
            fig.savefig(self.save_dir / "core_time_series.png", dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)
        
        # 2. Action traces
        fig, _ = plot_actions(
            self.history,
            time_axis=self.time_axis,
            figsize=self.figsize,
        )
        figures["actions"] = fig
        if save and self.save_dir:
            fig.savefig(self.save_dir / "actions.png", dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)
        
        # 3. Reward trace
        fig, _ = plot_reward_trace(
            self.history,
            time_axis=self.time_axis,
            cumulative=True,
            figsize=self.figsize,
        )
        figures["rewards"] = fig
        if save and self.save_dir:
            fig.savefig(self.save_dir / "rewards.png", dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)
        
        # 4. PSD observability (if enabled)
        if self.history.has_psd():
            # PSD time series
            result = plot_psd_time_series(
                self.history,
                time_axis=self.time_axis,
                figsize=(self.figsize[0], 8),
            )
            if result:
                fig, _ = result
                figures["psd_time_series"] = fig
                if save and self.save_dir:
                    fig.savefig(self.save_dir / "psd_time_series.png", dpi=150, bbox_inches="tight")
                if not show:
                    plt.close(fig)
            
            # PSD summary (C_L, C_M, C_S)
            result = plot_psd_summary(
                self.history,
                time_axis=self.time_axis,
                figsize=self.figsize,
            )
            if result:
                fig, _ = result
                figures["psd_summary"] = fig
                if save and self.save_dir:
                    fig.savefig(self.save_dir / "psd_summary.png", dpi=150, bbox_inches="tight")
                if not show:
                    plt.close(fig)
        
        # 5. Shape observability (if enabled)
        if self.history.has_shape():
            result = plot_shape_variables(
                self.history,
                time_axis=self.time_axis,
                figsize=self.figsize,
            )
            if result:
                fig, _ = result
                figures["shape"] = fig
                if save and self.save_dir:
                    fig.savefig(self.save_dir / "shape.png", dpi=150, bbox_inches="tight")
                if not show:
                    plt.close(fig)
        
        # 6. Anomaly visualization
        if self.detector and self.detector.has_anomalies():
            result = plot_anomalies(
                self.history,
                detector=self.detector,
                time_axis=self.time_axis,
                figsize=(self.figsize[0], 8),
            )
            if result:
                fig, _ = result
                figures["anomalies"] = fig
                if save and self.save_dir:
                    fig.savefig(self.save_dir / "anomalies.png", dpi=150, bbox_inches="tight")
                if not show:
                    plt.close(fig)
        
        # 7. Shield events (if present)
        result = plot_shield_events(
            self.history,
            time_axis=self.time_axis,
            figsize=self.figsize,
        )
        if result:
            fig, _ = result
            figures["shield_events"] = fig
            if save and self.save_dir:
                fig.savefig(self.save_dir / "shield_events.png", dpi=150, bbox_inches="tight")
            if not show:
                plt.close(fig)
        
        return figures
    
    def plot_custom_time_series(
        self,
        variables: List[str],
        save: bool = False,
        show: bool = True,
        filename: str = "custom_time_series.png",
    ) -> Optional[Figure]:
        """
        Plot custom set of variables.
        
        Args:
            variables: List of variable keys to plot
            save: Whether to save figure
            show: Whether to display figure
            filename: Filename if saving
        
        Returns:
            Figure object or None
        """
        if self.history is None or len(self.history) == 0:
            return None
        
        fig, _ = plot_time_series(
            self.history,
            variables=variables,
            time_axis=self.time_axis,
            figsize=self.figsize,
        )
        
        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)
        
        return fig
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected anomalies.
        
        Returns:
            Dictionary with anomaly counts and details
        """
        if self.detector is None:
            if self.history is None:
                return {"error": "No episode data recorded"}
            self.finalize_episode()
        
        if not self.detector.has_anomalies():
            return {"anomalies": False, "count": 0}
        
        return {
            "anomalies": True,
            "count": len(self.detector.anomalies),
            "by_type": self.detector.summary(),
            "total_steps": len(self.history),
        }
    
    def save_frames(
        self,
        frame_dir: Optional[str | Path] = None,
        variables: Optional[List[str]] = None,
        every_n_steps: int = 1,
    ):
        """
        Save individual frames for episode playback.
        
        Args:
            frame_dir: Directory to save frames (defaults to save_dir/frames)
            variables: Variables to include in frame plots
            every_n_steps: Save every Nth step (default: every step)
        """
        if self.history is None or len(self.history) == 0:
            raise ValueError("No episode data recorded")
        
        if frame_dir is None:
            if self.save_dir:
                frame_dir = self.save_dir / "frames"
            else:
                raise ValueError("Must specify frame_dir or save_dir")
        
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        if variables is None:
            variables = ["flow", "pressure", "clog", "E_norm", "C_out"]
        
        # Create frames
        for step_idx in range(0, len(self.history), every_n_steps):
            # Create a figure showing state up to this step
            fig, axes = plt.subplots(len(variables), 1, figsize=(10, 2 * len(variables)), sharex=True)
            if len(variables) == 1:
                axes = [axes]
            
            # Get time array up to this step
            if self.time_axis == "time":
                x = self.history.get_time_array()[:step_idx + 1]
            else:
                x = self.history.get_step_array()[:step_idx + 1]
            
            # Plot each variable
            for ax_idx, var_key in enumerate(variables):
                ax = axes[ax_idx]
                
                # Extract data up to this step
                if var_key.startswith("sensor_"):
                    y = self.history.get_sensor_variable(var_key)[:step_idx + 1]
                else:
                    y = self.history.get_truth_variable(var_key)[:step_idx + 1]
                
                # Get metadata
                from .time_series import VARIABLE_METADATA
                label, unit, ylim = VARIABLE_METADATA.get(var_key, (var_key, "", None))
                
                ax.plot(x, y, linewidth=1.5)
                ax.set_ylabel(f"{label} ({unit})" if unit else label)
                ax.grid(True, alpha=0.3)
                
                if ylim:
                    ax.set_ylim(ylim)
            
            axes[-1].set_xlabel("Time (s)" if self.time_axis == "time" else "Step")
            fig.suptitle(f"Step {step_idx}", fontsize=12)
            fig.tight_layout()
            
            # Save frame
            fig.savefig(frame_dir / f"frame_{step_idx:05d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
    
    def create_video_from_frames(
        self,
        frame_dir: Optional[str | Path] = None,
        video_path: Optional[str | Path] = None,
        fps: int = 10,
        bitrate: int = 2400,
    ) -> Path:
        """
        Create video from saved frames using matplotlib animation.
        
        Requires ffmpeg to be installed and on PATH.
        
        Args:
            frame_dir: Directory containing frames (defaults to save_dir/frames)
            video_path: Output video path (defaults to save_dir/episode.mp4)
            fps: Frames per second
            bitrate: Video bitrate
        
        Returns:
            Path to created video file
        """
        try:
            from matplotlib import animation
        except ImportError:
            raise ImportError("matplotlib.animation required for video creation")
        
        if frame_dir is None:
            if self.save_dir:
                frame_dir = self.save_dir / "frames"
            else:
                raise ValueError("Must specify frame_dir or save_dir")
        
        frame_dir = Path(frame_dir)
        if not frame_dir.exists():
            raise ValueError(f"Frame directory does not exist: {frame_dir}")
        
        # Find all frame files
        frame_files = sorted(frame_dir.glob("frame_*.png"))
        if not frame_files:
            raise ValueError(f"No frame files found in {frame_dir}")
        
        if video_path is None:
            if self.save_dir:
                video_path = self.save_dir / "episode.mp4"
            else:
                video_path = frame_dir.parent / "episode.mp4"
        
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load first frame to get dimensions
        import matplotlib.image as mpimg
        img = mpimg.imread(str(frame_files[0]))
        height, width = img.shape[:2]
        
        # Create figure
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        
        im = ax.imshow(img, animated=True)
        
        def animate(frame_idx):
            img = mpimg.imread(str(frame_files[frame_idx]))
            im.set_array(img)
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(frame_files),
            interval=1000 / fps,
            blit=True,
            repeat=False,
        )
        
        # Save video
        writer = animation.FFMpegWriter(
            fps=fps,
            metadata={"artist": "HydrOS Observatory"},
            bitrate=bitrate,
        )
        anim.save(str(video_path), writer=writer)
        plt.close(fig)
        
        return video_path
    
    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.history) if self.history else 0
