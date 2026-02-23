"""
This module defines a state machine for managing solver stages.

It includes the `Stage` enumeration for different states, the `StateMachine` class
to handle state transitions, and a decorator `solver_trans` for managing transitions
in solver methods.
"""

from enum import Enum, auto


class Stage(Enum):
    """State machine stages for the solver."""
    INIT = auto()
    WAIT_REAL = auto()
    WAIT_QUANT = auto()
    FINAL = auto()


class StateMachine:
    """A simple state machine to manage solver stages."""
    def __init__(self):
        self._state = Stage.INIT

    @property
    def state(self):
        """Current state of the state machine."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def transition_to(self, new_state):
        """Transition to a new state if valid."""
        if self._is_valid_transition(new_state):
            self._state = new_state
        else:
            raise RuntimeError(f"invalid state {self._state.name} -> {new_state.name}")
        self._state = new_state

    def _is_valid_transition(self, new_state):
        valid = {
            Stage.INIT: [Stage.WAIT_REAL],
            Stage.WAIT_REAL: [Stage.WAIT_QUANT],
            Stage.WAIT_QUANT: [Stage.WAIT_REAL, Stage.FINAL],
            Stage.FINAL: [],
        }
        return new_state in valid[self._state]



def solver_trans(from_stages, to_stage):
    """Decorator to manage state transitions in the solver.
    Args:
        from_stages (list or Stage): List of stages from which the function can be called.
        to_stage (Stage): The stage to transition to after the function call.
    Returns:
        function: Decorated function that checks the current state and transitions.
    """
    if not isinstance(from_stages, (list, tuple)):
        from_stages = [from_stages]

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.fsm.state not in from_stages:
                raise RuntimeError(
                    f"Invalid call to {func.__name__}() in current state: {self.fsm.state.name}"
                )
            self.fsm.transition_to(to_stage)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
