"""
Legacy constraint scheduling system (DEPRECATED).

⚠️ This module is deprecated. Use the new architecture instead:
- Replace ConstraintScheduleManager with Scheduler (from schedulers/)
- See legacy/MIGRATION_GUIDE.md for migration instructions

This module is kept for backward compatibility only.

Original description:
Constraint scheduling system for soft ↔ hard constraint transitions.

This module provides flexible scheduling for constraint management:
- Soft constraint weight scheduling (alpha, beta)
- Hard constraint clearance scheduling
- Constraint activation/deactivation scheduling
- Support for soft → hard, hard → soft, and fixed constraints
- Forward and reverse indexing modes for compatibility with diffusion

Indexing Modes:
===============

Forward Mode (default, reverse_mode=False):
  - step = 0: final state (clean trajectory)
  - step = total_steps: initial noise state
  - Progress: step / total_steps (0 → 1, final → initial)

Reverse Mode (reverse_mode=True):
  - step = 0: initial noise state (early diffusion, noisy)
  - step = total_steps: final state (late diffusion, clean)
  - Progress: step / total_steps (0 → 1, initial → final)
  - This matches EDOC diffusion index convention

Usage Example:
==============

# Forward mode (default)
schedule = ConstraintScheduleManager.create_soft_to_hard(
    soft_alpha_start=1.0,  # At step=0 (final)
    soft_alpha_end=0.0,    # At step=total_steps (initial)
    reverse_mode=False
)

# Reverse mode (syncs with diffusion)
schedule = ConstraintScheduleManager.create_soft_to_hard(
    soft_alpha_start=1.0,  # At step=0 (initial noise)
    soft_alpha_end=0.0,    # At step=total_steps (final)
    reverse_mode=True
)

# In EDOC, use reverse mode with diffusion index:
# for i in range(Ndiffuse - 1, 0, -1):
#     # Convert diffusion index to constraint step
#     # i = Ndiffuse-1 (initial noise) → step = 0
#     # i = 1 (near final) → step = Ndiffuse - 2 ≈ total_steps
#     constraint_step = (Ndiffuse - 1) - i
#     alpha = schedule.get_soft_alpha(step=constraint_step, total_steps=Ndiffuse - 1)
#
# Alternative: Direct mapping (if i range matches step range)
# for i in range(Ndiffuse - 1, 0, -1):
#     # Direct use if you adjust total_steps accordingly
#     alpha = schedule.get_soft_alpha(step=i - 1, total_steps=Ndiffuse - 2)
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np


class ConstraintSchedule(ABC):
    """
    Abstract base class for constraint scheduling.
    
    A schedule defines how a constraint parameter (e.g., weight, clearance)
    changes over diffusion steps.
    """
    
    @abstractmethod
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """
        Compute scheduled value at given step.
        
        Args:
            step: Current step (0 = final, total_steps = initial noise)
            total_steps: Total number of steps
            
        Returns:
            Scheduled value (float)
        """
        pass
    
    def get(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Alias for __call__ for clarity."""
        return self.__call__(step, total_steps)


class ConstantSchedule(ConstraintSchedule):
    """
    Constant schedule: value never changes.
    
    Used for fixed constraints (all soft or all hard).
    """
    
    def __init__(self, value: float):
        """
        Initialize constant schedule.
        
        Args:
            value: Constant value to return
        """
        self.value = float(value)
    
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Return constant value."""
        return self.value


class LinearSchedule(ConstraintSchedule):
    """
    Linear schedule: value changes linearly from start to end.
    
    value(step) = start + (end - start) * progress
    where progress = step / total_steps
    """
    
    def __init__(self, start: float, end: float):
        """
        Initialize linear schedule.
        
        Args:
            start: Value at step 0 (final)
            end: Value at step total_steps (initial)
        """
        self.start = float(start)
        self.end = float(end)
    
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Compute linear interpolation."""
        if step is None or total_steps is None or total_steps == 0:
            return self.start
        
        progress = step / total_steps
        return self.start + progress * (self.end - self.start)


class ExponentialSchedule(ConstraintSchedule):
    """
    Exponential schedule: value changes exponentially.
    
    value(step) = start * (end / start) ^ progress
    """
    
    def __init__(self, start: float, end: float):
        """
        Initialize exponential schedule.
        
        Args:
            start: Value at step 0 (final)
            end: Value at step total_steps (initial)
        """
        self.start = float(start)
        self.end = float(end)
    
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Compute exponential interpolation."""
        if step is None or total_steps is None or total_steps == 0:
            return self.start
        
        progress = step / total_steps
        if abs(self.start) < 1e-10:
            return self.start
        
        ratio = self.end / self.start
        return self.start * (ratio ** progress)


class CosineSchedule(ConstraintSchedule):
    """
    Cosine schedule: smooth transition using cosine interpolation.
    
    value(step) = start + (end - start) * (1 - cos(π * progress)) / 2
    """
    
    def __init__(self, start: float, end: float):
        """
        Initialize cosine schedule.
        
        Args:
            start: Value at step 0 (final)
            end: Value at step total_steps (initial)
        """
        self.start = float(start)
        self.end = float(end)
    
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Compute cosine interpolation."""
        if step is None or total_steps is None or total_steps == 0:
            return self.start
        
        progress = step / total_steps
        smooth_progress = (1.0 - np.cos(np.pi * progress)) / 2.0
        return self.start + smooth_progress * (self.end - self.start)


class PiecewiseSchedule(ConstraintSchedule):
    """
    Piecewise linear schedule with breakpoints.
    
    Allows defining custom piecewise linear transitions.
    """
    
    def __init__(self, breakpoints: list[Tuple[float, float]]):
        """
        Initialize piecewise schedule.
        
        Args:
            breakpoints: List of (progress, value) pairs, where progress in [0, 1]
                        progress=0 is step 0 (final), progress=1 is step total_steps (initial)
                        Must be sorted by progress.
        """
        if len(breakpoints) < 2:
            raise ValueError("PiecewiseSchedule requires at least 2 breakpoints")
        
        # Sort by progress
        breakpoints = sorted(breakpoints, key=lambda x: x[0])
        
        # Ensure coverage
        if breakpoints[0][0] != 0.0:
            raise ValueError(f"First breakpoint progress must be 0.0, got {breakpoints[0][0]}")
        if breakpoints[-1][0] != 1.0:
            raise ValueError(f"Last breakpoint progress must be 1.0, got {breakpoints[-1][0]}")
        
        self.breakpoints = breakpoints
    
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Compute piecewise linear interpolation."""
        if step is None or total_steps is None or total_steps == 0:
            return self.breakpoints[0][1]
        
        progress = step / total_steps
        
        # Find the segment containing progress
        for i in range(len(self.breakpoints) - 1):
            p1, v1 = self.breakpoints[i]
            p2, v2 = self.breakpoints[i + 1]
            
            if p1 <= progress <= p2:
                # Linear interpolation within segment
                if p2 == p1:
                    return v1
                t = (progress - p1) / (p2 - p1)
                return v1 + t * (v2 - v1)
        
        # Should not reach here, but return last value
        return self.breakpoints[-1][1]


class CustomSchedule(ConstraintSchedule):
    """
    Custom schedule defined by a callable function.
    
    Allows maximum flexibility for user-defined scheduling.
    """
    
    def __init__(self, func: Callable[[Optional[int], Optional[int]], float]):
        """
        Initialize custom schedule.
        
        Args:
            func: Function (step, total_steps) -> value
        """
        self.func = func
    
    def __call__(self, step: Optional[int], total_steps: Optional[int]) -> float:
        """Call custom function."""
        return float(self.func(step, total_steps))


class ConstraintScheduleManager:
    """
    Manager for scheduling soft and hard constraint parameters.
    
    Supports all scheduling modes:
    1. Scheduled soft → hard: soft weight decreases, hard clearance tightens
    2. All soft: fixed soft constraints, no hard
    3. All hard: no soft, fixed hard constraints
    4. All soft + hard: both fixed, no scheduling
    5. Hard → soft: reverse schedule (hard → soft transition)
    
    Supports two indexing modes:
    - Forward mode (default): step=0 is final state, step=total_steps is initial noise
    - Reverse mode: step=0 is initial noise, step=total_steps is final state (syncs with diffusion)
    """
    
    def __init__(
        self,
        soft_alpha_schedule: Optional[ConstraintSchedule] = None,
        soft_beta_schedule: Optional[ConstraintSchedule] = None,
        hard_clearance_schedule: Optional[ConstraintSchedule] = None,
        soft_active_schedule: Optional[Callable[[Optional[int], Optional[int]], bool]] = None,
        hard_active_schedule: Optional[Callable[[Optional[int], Optional[int]], bool]] = None,
        reverse_mode: bool = False,
    ):
        """
        Initialize constraint schedule manager.
        
        Args:
            soft_alpha_schedule: Schedule for soft constraint alpha (barrier strength)
                                If None, soft constraints use default alpha
            soft_beta_schedule: Schedule for soft constraint beta (barrier sharpness)
                               If None, soft constraints use default beta
            hard_clearance_schedule: Schedule for hard constraint clearance
                                    If None, hard constraints use default clearance
            soft_active_schedule: Function (step, total_steps) -> bool
                                 Returns True if soft constraints should be active
                                 If None, soft constraints are always active if present
            hard_active_schedule: Function (step, total_steps) -> bool
                                 Returns True if hard constraints should be active
                                 If None, hard constraints are always active if present
            reverse_mode: If True, uses reverse indexing:
                         - step=0: initial noise (early diffusion)
                         - step=total_steps: final state (late diffusion)
                         This matches EDOC diffusion index convention.
                         If False (default), uses forward indexing:
                         - step=0: final state (clean)
                         - step=total_steps: initial noise
        """
        self.soft_alpha_schedule = soft_alpha_schedule
        self.soft_beta_schedule = soft_beta_schedule
        self.hard_clearance_schedule = hard_clearance_schedule
        self.soft_active_schedule = soft_active_schedule
        self.hard_active_schedule = hard_active_schedule
        self.reverse_mode = bool(reverse_mode)
    
    def _convert_step(self, step: Optional[int], total_steps: Optional[int]) -> Optional[int]:
        """
        Convert step between indexing modes.

        IMPORTANT:
        - Schedules in this codebase are defined in the *same* indexing convention that
          users supply to `get_*` methods.
        - Therefore, we do NOT remap `step` internally.

        The two supported conventions are:
        - Forward mode (reverse_mode=False): step=0 is final/clean, step=total_steps is initial/noisy
        - Reverse mode (reverse_mode=True): step=0 is initial/noisy, step=total_steps is final/clean

        It is the caller's responsibility to pass `step` in the intended convention.
        EDOC's diffusion loop typically uses reverse mode (step increases from noisy → clean).
        
        Args:
            step: Step in user's convention
            total_steps: Total steps
            
        Returns:
            Converted step for internal schedule (forward mode)
        """
        return step
    
    def get_soft_alpha(
        self, 
        default: float = 1.0,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """Get scheduled soft constraint alpha."""
        if self.soft_alpha_schedule is None:
            return default
        forward_step = self._convert_step(step, total_steps)
        return self.soft_alpha_schedule(forward_step, total_steps)
    
    def get_soft_beta(
        self,
        default: float = 10.0,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """Get scheduled soft constraint beta."""
        if self.soft_beta_schedule is None:
            return default
        forward_step = self._convert_step(step, total_steps)
        return self.soft_beta_schedule(forward_step, total_steps)
    
    def get_hard_clearance(
        self,
        default: float = 0.0,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """Get scheduled hard constraint clearance."""
        if self.hard_clearance_schedule is None:
            return default
        forward_step = self._convert_step(step, total_steps)
        return self.hard_clearance_schedule(forward_step, total_steps)
    
    def is_soft_active(
        self,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> bool:
        """Check if soft constraints should be active."""
        if self.soft_active_schedule is None:
            return True  # Default: always active if present
        forward_step = self._convert_step(step, total_steps)
        return bool(self.soft_active_schedule(forward_step, total_steps))
    
    def is_hard_active(
        self,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> bool:
        """Check if hard constraints should be active."""
        if self.hard_active_schedule is None:
            return True  # Default: always active if present
        forward_step = self._convert_step(step, total_steps)
        return bool(self.hard_active_schedule(forward_step, total_steps))
    
    @staticmethod
    def create_soft_to_hard(
        soft_alpha_start: float = 1.0,
        soft_alpha_end: float = 0.0,
        soft_beta_start: float = 10.0,
        soft_beta_end: float = 10.0,
        hard_clearance_start: float = 0.5,
        hard_clearance_end: float = 0.1,
        schedule_type: str = "linear",
        reverse_mode: bool = False,
    ) -> "ConstraintScheduleManager":
        """
        Create a soft → hard transition schedule.
        
        Soft constraints fade out (alpha decreases) while hard constraints
        tighten (clearance decreases).
        
        Args:
            soft_alpha_start: Initial soft alpha
                            - Forward mode: at step 0 (final state)
                            - Reverse mode: at step 0 (initial noise)
            soft_alpha_end: Final soft alpha
                          - Forward mode: at step total_steps (initial noise)
                          - Reverse mode: at step total_steps (final state)
            soft_beta_start: Initial soft beta
            soft_beta_end: Final soft beta
            hard_clearance_start: Initial hard clearance (relaxed)
            hard_clearance_end: Final hard clearance (strict)
            schedule_type: "linear", "exponential", "cosine", or "piecewise"
            reverse_mode: If True, uses reverse indexing (syncs with diffusion)
            
        Returns:
            ConstraintScheduleManager configured for soft → hard transition
        """
        schedule_class = {
            "linear": LinearSchedule,
            "exponential": ExponentialSchedule,
            "cosine": CosineSchedule,
        }.get(schedule_type, LinearSchedule)
        
        return ConstraintScheduleManager(
            soft_alpha_schedule=schedule_class(soft_alpha_start, soft_alpha_end),
            soft_beta_schedule=schedule_class(soft_beta_start, soft_beta_end) if soft_beta_start != soft_beta_end else ConstantSchedule(soft_beta_start),
            hard_clearance_schedule=schedule_class(hard_clearance_start, hard_clearance_end),
            reverse_mode=reverse_mode,
        )
    
    @staticmethod
    def create_hard_to_soft(
        soft_alpha_start: float = 0.0,
        soft_alpha_end: float = 1.0,
        soft_beta_start: float = 10.0,
        soft_beta_end: float = 10.0,
        hard_clearance_start: float = 0.1,
        hard_clearance_end: float = 0.5,
        schedule_type: str = "linear",
        reverse_mode: bool = False,
    ) -> "ConstraintScheduleManager":
        """
        Create a hard → soft transition schedule (reverse of soft → hard).
        
        Hard constraints relax while soft constraints become stronger.
        
        Args:
            soft_alpha_start: Initial soft alpha (low)
            soft_alpha_end: Final soft alpha (high)
            soft_beta_start: Initial soft beta
            soft_beta_end: Final soft beta
            hard_clearance_start: Initial hard clearance (strict)
            hard_clearance_end: Final hard clearance (relaxed)
            schedule_type: "linear", "exponential", "cosine"
            reverse_mode: If True, uses reverse indexing (syncs with diffusion)
            
        Returns:
            ConstraintScheduleManager configured for hard → soft transition
        """
        schedule_class = {
            "linear": LinearSchedule,
            "exponential": ExponentialSchedule,
            "cosine": CosineSchedule,
        }.get(schedule_type, LinearSchedule)
        
        return ConstraintScheduleManager(
            soft_alpha_schedule=schedule_class(soft_alpha_start, soft_alpha_end),
            soft_beta_schedule=schedule_class(soft_beta_start, soft_beta_end) if soft_beta_start != soft_beta_end else ConstantSchedule(soft_beta_start),
            hard_clearance_schedule=schedule_class(hard_clearance_start, hard_clearance_end),
            reverse_mode=reverse_mode,
        )
    
    @staticmethod
    def create_all_soft(
        alpha: float = 1.0,
        beta: float = 10.0,
        reverse_mode: bool = False,
    ) -> "ConstraintScheduleManager":
        """
        Create a fixed all-soft schedule (no hard constraints).
        
        Args:
            alpha: Fixed soft constraint alpha
            beta: Fixed soft constraint beta
            reverse_mode: If True, uses reverse indexing (syncs with diffusion)
            
        Returns:
            ConstraintScheduleManager for all-soft mode
        """
        return ConstraintScheduleManager(
            soft_alpha_schedule=ConstantSchedule(alpha),
            soft_beta_schedule=ConstantSchedule(beta),
            reverse_mode=reverse_mode,
        )
    
    @staticmethod
    def create_all_hard(
        clearance: float = 0.1,
        reverse_mode: bool = False,
    ) -> "ConstraintScheduleManager":
        """
        Create a fixed all-hard schedule (no soft constraints).
        
        Args:
            clearance: Fixed hard constraint clearance
            reverse_mode: If True, uses reverse indexing (syncs with diffusion)
            
        Returns:
            ConstraintScheduleManager for all-hard mode
        """
        return ConstraintScheduleManager(
            hard_clearance_schedule=ConstantSchedule(clearance),
            reverse_mode=reverse_mode,
        )
    
    @staticmethod
    def create_soft_and_hard(
        soft_alpha: float = 1.0,
        soft_beta: float = 10.0,
        hard_clearance: float = 0.1,
        reverse_mode: bool = False,
    ) -> "ConstraintScheduleManager":
        """
        Create a fixed schedule with both soft and hard constraints active.
        
        Args:
            soft_alpha: Fixed soft constraint alpha
            soft_beta: Fixed soft constraint beta
            hard_clearance: Fixed hard constraint clearance
            reverse_mode: If True, uses reverse indexing (syncs with diffusion)
            
        Returns:
            ConstraintScheduleManager for soft+hard mode
        """
        return ConstraintScheduleManager(
            soft_alpha_schedule=ConstantSchedule(soft_alpha),
            soft_beta_schedule=ConstantSchedule(soft_beta),
            hard_clearance_schedule=ConstantSchedule(hard_clearance),
            reverse_mode=reverse_mode,
        )

