"""Versioned public recovery guard with a measurable liveness release.

The v1 governor latches a retreat until the TCP reaches one absolute height.
That remains the default.  This P5 variant additionally releases the latch
after the current and predicted public pose have both satisfied the existing
keyway predicate for the existing settle clock.  It changes no task threshold,
command bound, force law, observation, or learned weight.
"""


class SafeSettledRetreatReleaseMixin:
    LIVENESS_VERSION="public-safe-settled-retreat-release-v1"

    def _update_retreating(self,phase,p,safe,approaching):
        fb=self.pose_feedback
        self.retreating|=approaching&~safe&(p[:,2]<fb.guard_clear_z_m-.001)
        height_clear=p[:,2]>=fb.guard_clear_z_m
        safe_settled=(phase==3)&safe&(self.clear_count>=fb.guard_settle_ticks)
        self.retreating&=~(height_clear|safe_settled)

