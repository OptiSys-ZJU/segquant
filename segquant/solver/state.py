from enum import Enum, auto


class Stage(Enum):
    INIT = auto()
    WAIT_REAL = auto()
    WAIT_QUANT = auto()
    FINAL = auto()


class StateMachine:
    def __init__(self):
        self._state = Stage.INIT

    @property
    def state(self):
        return self._state

    def transition_to(self, new_state):
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
