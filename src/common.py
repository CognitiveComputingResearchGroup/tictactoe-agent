import collections
import random

from env.environment import Board, Move


class Module(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        Update internal state of module using supplied arguments.
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Returns a read-only iterator over the module's content.

        A KeyError is raised if the key doesn't exist.
        A TypeError is raised if the key is of an incorrect type.
        """
        raise NotImplementedError()

    def __next__(self):
        """
        Returns next module state.  Corresponds to the next item that would be returned from iterating over this module.
        """
        raise NotImplementedError()


class Environment(Module):
    def __init__(self):
        super().__init__()
        from src.env.environment import Board
        self._board = Board.blank_board()
        self._mark = 'X'

    def move_possible(self, move):
        return 0 <= move < 9 and self._board.is_blank(move)

    def _make_move(self, pos, mark):
        if self.move_possible(pos):
            self._board[pos] = mark
            return True
        return False

    def __call__(self, motor_plan):
        for action in motor_plan:
            if action is not None:
                self._make_move(action[0], action[1])

    def __next__(self):
        return self._board


class PerceptualAssociativeMemory(Module):
    def __init__(self):
        super().__init__()

        self.pam_contents = []

    def __call__(self, cue_content):
        content = self.cue(cue_content)
        return (cue_content, content)

    def cue(self, cue_content):
        return 'happy_dummy'


class SensoryMemory(Module):
    def __init__(self):
        super().__init__()

        self.sensory_memory = []

    def __call__(self, *args, **kwargs):
        self.sensory_memory.append(args[0])

    def __next__(self):
        return self.sensory_memory[-1]


class AttentionCodelet(Module):
    def __init__(self, is_match=lambda x: True):
        super().__init__()
        self._match_content = is_match
        self.coalition = []

    def __call__(self, module):
        for content in module:
            if self._match_content(content):
                self.coalition.append(content)

    def __next__(self):
        coalition = self.coalition
        self.coalition = []
        return coalition


class StructureBuildingCodelet(Module):
    def __init__(self, select=lambda x: True, transform=lambda x: x):
        super().__init__()

        self.select = select
        self.transform = transform

        self._structures = []

    def __call__(self, workspace):
        new_structures = map(self.transform, filter(self.select, workspace))
        self._structures.extend(new_structures)

    def __next__(self):
        structures = self._structures
        self._structures = []
        return structures


def is_board(structure):
    return isinstance(structure, Board)


def create_move(board):
    if board.is_full():
        return None, board

    move = Move(position=random.choice(board.blanks), mark='X')

    new_board = board.copy()
    new_board[move.position] = move.mark

    return move, new_board


class Workspace(Module):
    def __init__(self):
        super().__init__()

        self.workspace_content = []

    def __call__(self, content):
        if isinstance(content, list):
            self.workspace_content.extend(content)
        else:
            self.workspace_content.append(content)

    def __next__(self):
        return self.workspace_content[-1]

    def __iter__(self):
        return iter(self.workspace_content)


class CueingProcess(Module):
    def __init__(self):
        super().__init__()

        self.cued_content = []

    def __call__(self, content, module):
        # cue module
        cued_content = module.cue(content)

        # receive activated content from cued module
        self.cued_content.append(cued_content)

    def __next__(self):
        cued_content = self.cued_content
        self.cued_content = []
        return cued_content


class GlobalWorkspace(Module):
    def __init__(self):
        super().__init__()

        self.coalitions = []

    def __call__(self, coalition):
        if isinstance(coalition, list):
            self.coalitions.extend(coalition)
        else:
            self.coalitions.append(coalition)

    def __next__(self):
        return self.coalitions[-1]


class Scheme(object):
    def __init__(self, context=None, action=None, result=None):
        self.context = context
        self.action = action
        self.result = result

        self.activation = 0.0


class ProceduralMemory(Module):
    def __init__(self, initial_schemes=None, context_match=None, activation_threshold=1.0):
        super().__init__()

        self._schemes = [] if initial_schemes is None else list(initial_schemes)
        self._context_match = context_match
        self._activation_threshold = activation_threshold

    def __call__(self, broadcast):
        if broadcast is None:
            return

        # Increase activation of schemes with context match
        for scheme in self._schemes:
            if self._context_match(broadcast, scheme):
                scheme.activation = 1.0

    def __next__(self):
        # Find schemes with activation >= activation_threshold
        active_schemes = list(filter(lambda s: s.activation >= self._activation_threshold, self._schemes))

        # Return a random scheme if no schemes above activation threshold
        if len(active_schemes) == 0 and len(self._schemes) > 0:
            return random.choice(self._schemes)

        return active_schemes


def exact_match_context_by_move(content, scheme):
    # Case 1: Content is a Move
    if isinstance(content, Move):
        return scheme.context == content

    # Case 2: Content is a non-Move iterable (special consideration for strings to avoid infinite recursion)
    if isinstance(content, collections.Iterable) and not isinstance(content, str):
        return any(exact_match_context_by_move(c, scheme) for c in content)

    # Case 3: Content is a non-Move, non-Iterable
    return False


def board_position_after_move(curr_board, move):
    new_board = curr_board.copy()
    new_board[move.position] = move.mark
    return new_board


class ActionSelection(Module):
    def __init__(self):
        super().__init__()
        self.behaviors = []

    def __call__(self, behavior):
        self.behaviors.append(behavior)

    def __next__(self):
        maximally_active_behavior = sorted(self.behaviors, key=lambda behavior: behavior.activation)[-1]

        # TODO: This may need to be updated to treat the behavior's result as a function.  (See
        # TODO: initial schemes in agent.py)
        expectation_codelet = AttentionCodelet(
            lambda x: x in maximally_active_behavior.result or x == maximally_active_behavior.result)
        return maximally_active_behavior, expectation_codelet


class SensoryMotorSystem(Module):
    def __init__(self):
        super().__init__()

        self.motor_plan = []

    def __call__(self, behavior):
        self.motor_plan.append(behavior.action)

    def __next__(self):
        motor_plans = self.motor_plan
        self.motor_plan = []
        return motor_plans
