"""
Checkpoint Management for GRPO Training with Resume and Recovery

This module provides comprehensive checkpoint management including:
- Regular training checkpoints with rotation
- Recovery checkpoints for stability
- Automatic checkpoint discovery for resume
- Checkpoint validation and integrity checks
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch


class CheckpointManager:
    """Manages training checkpoints with resume and recovery capabilities."""
    
    def __init__(
        self, 
        output_dir: str,
        max_keep: int = 3,
        recovery_interval: int = 500
    ):
        """
        Initialize checkpoint manager
        
        Args:
            output_dir: Directory for saving checkpoints
            max_keep: Maximum number of regular checkpoints to keep
            recovery_interval: Steps between recovery checkpoints
        """
        self.output_dir = Path(output_dir)
        self.max_keep = max_keep
        self.recovery_interval = recovery_interval
        
        self.regular_checkpoints: List[Path] = []
        self.stable_checkpoints: List[Path] = []
        self.recovery_checkpoint: Optional[Path] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self, 
        step: int, 
        checkpoint_data: Dict[str, Any], 
        is_stable: bool = False,
        is_recovery: bool = False
    ) -> Path:
        """
        Save a checkpoint with proper categorization
        
        Args:
            step: Training step number
            checkpoint_data: Complete checkpoint dictionary
            is_stable: Whether this is a stable checkpoint (after evaluation)
            is_recovery: Whether this is a recovery checkpoint
            
        Returns:
            Path to saved checkpoint directory
        """
        # Determine checkpoint type and path
        if is_recovery:
            checkpoint_dir = self.output_dir / "checkpoint-recovery"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Add checkpoint metadata
        checkpoint_data.update({
            'checkpoint_type': 'recovery' if is_recovery else 'regular',
            'is_stable': is_stable,
            'save_time': time.time(),
            'step': step
        })
        
        # Save trainer state (main checkpoint)
        trainer_path = checkpoint_dir / "trainer.pt"
        torch.save(checkpoint_data, trainer_path)
        
        # Save human-readable metadata
        metadata_path = checkpoint_dir / "metadata.json"
        metadata = {
            'step': step,
            'checkpoint_type': checkpoint_data['checkpoint_type'],
            'is_stable': is_stable,
            'save_time': checkpoint_data['save_time'],
            'run_id': checkpoint_data.get('run_id'),
            'total_samples_trained': checkpoint_data.get('total_samples_trained', 0),
            'nan_skip_count': checkpoint_data.get('nan_skip_count', 0)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update checkpoint tracking
        if is_recovery:
            self.recovery_checkpoint = checkpoint_dir
        else:
            self.regular_checkpoints.append(checkpoint_dir)
            
            if is_stable:
                self.stable_checkpoints.append(checkpoint_dir)
            
            # Cleanup old checkpoints
            self._cleanup_checkpoints()
        
        self.logger.info(f"Saved {'recovery' if is_recovery else 'regular'} checkpoint: {checkpoint_dir}")
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint data with validation
        
        Args:
            checkpoint_path: Path to checkpoint directory or trainer.pt file
            
        Returns:
            Tuple of (checkpoint_data, metadata)
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint is corrupted
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Handle both directory and file paths
        if checkpoint_path.is_dir():
            trainer_path = checkpoint_path / "trainer.pt"
            metadata_path = checkpoint_path / "metadata.json"
        else:
            trainer_path = checkpoint_path
            metadata_path = checkpoint_path.parent / "metadata.json"
        
        if not trainer_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {trainer_path}")
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(trainer_path, map_location='cpu')
            
            # Load metadata if available
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Validate checkpoint integrity
            required_keys = ['model_state_dict', 'step_count', 'run_id']
            missing_keys = [key for key in required_keys if key not in checkpoint_data]
            if missing_keys:
                raise RuntimeError(f"Corrupted checkpoint, missing keys: {missing_keys}")
            
            self.logger.info(f"Loaded checkpoint from step {checkpoint_data['step_count']}")
            return checkpoint_data, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {trainer_path}: {e}")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint in output directory
        
        Returns:
            Path to latest checkpoint directory, or None if no checkpoints found
        """
        checkpoint_dirs = []
        
        # Find all checkpoint directories
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                # Extract step number for sorting
                try:
                    if item.name == 'checkpoint-recovery':
                        # Recovery checkpoint gets high priority
                        checkpoint_dirs.append((float('inf'), item))
                    else:
                        step = int(item.name.split('-')[1])
                        checkpoint_dirs.append((step, item))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Return latest checkpoint (highest step number)
        checkpoint_dirs.sort(reverse=True)
        latest_path = checkpoint_dirs[0][1]
        
        self.logger.info(f"Found latest checkpoint: {latest_path}")
        return latest_path
    
    def find_last_stable_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent stable checkpoint
        
        Returns:
            Path to last stable checkpoint, or None if none found
        """
        if not self.stable_checkpoints:
            # Try to identify stable checkpoints from existing directories
            self._discover_existing_checkpoints()
        
        if self.stable_checkpoints:
            return self.stable_checkpoints[-1]
        
        # Fallback to any checkpoint if no stable ones
        return self.find_latest_checkpoint()
    
    def cleanup_debug_checkpoints(self, max_age_hours: int = 24):
        """
        Clean up old debug checkpoints
        
        Args:
            max_age_hours: Maximum age in hours for debug checkpoints
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_count = 0
        
        for item in self.output_dir.iterdir():
            if item.is_dir() and 'debug' in item.name.lower():
                try:
                    # Check creation time
                    if item.stat().st_ctime < cutoff_time:
                        shutil.rmtree(item)
                        removed_count += 1
                        self.logger.debug(f"Removed old debug checkpoint: {item}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove debug checkpoint {item}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old debug checkpoints")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_keep limit."""
        # Sort by step number (extract from directory name)
        sorted_checkpoints = []
        for cp_path in self.regular_checkpoints:
            try:
                if cp_path.name == 'checkpoint-recovery':
                    continue  # Never delete recovery checkpoints
                step = int(cp_path.name.split('-')[1])
                sorted_checkpoints.append((step, cp_path))
            except (ValueError, IndexError):
                continue
        
        sorted_checkpoints.sort()
        
        # Remove oldest checkpoints beyond max_keep
        while len(sorted_checkpoints) > self.max_keep:
            step, old_checkpoint = sorted_checkpoints.pop(0)
            
            # Don't delete stable checkpoints
            if old_checkpoint in self.stable_checkpoints:
                continue
                
            try:
                shutil.rmtree(old_checkpoint)
                self.regular_checkpoints.remove(old_checkpoint)
                self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
    
    def _discover_existing_checkpoints(self):
        """Discover and categorize existing checkpoints in output directory."""
        for item in self.output_dir.iterdir():
            if not item.is_dir() or not item.name.startswith('checkpoint-'):
                continue
                
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('is_stable', False):
                        self.stable_checkpoints.append(item)
                    
                    if 'recovery' in item.name:
                        self.recovery_checkpoint = item
                    else:
                        self.regular_checkpoints.append(item)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata for {item}: {e}")
        
        # Sort stable checkpoints by step
        self.stable_checkpoints.sort(key=lambda x: self._extract_step(x))
    
    def _extract_step(self, checkpoint_path: Path) -> int:
        """Extract step number from checkpoint path."""
        try:
            if checkpoint_path.name == 'checkpoint-recovery':
                return float('inf')
            return int(checkpoint_path.name.split('-')[1])
        except (ValueError, IndexError):
            return 0
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get summary information about managed checkpoints."""
        return {
            'output_dir': str(self.output_dir),
            'regular_checkpoints': len(self.regular_checkpoints),
            'stable_checkpoints': len(self.stable_checkpoints),
            'has_recovery_checkpoint': self.recovery_checkpoint is not None,
            'max_keep': self.max_keep,
            'recovery_interval': self.recovery_interval
        }