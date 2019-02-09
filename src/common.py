import collections

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax

from env.environment import Board, Move


class Module(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def receive_broadcast(self, broadcast):
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

    move = Move(position=np.random.choice(board.blanks), mark='X')

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

    def __init__(self, context=None, action=None, result=None, current_activation=0.0, base_level_activation=0.0):
        self.context = context
        self.action = action
        self.result = result

        self.current_activation = current_activation
        self.base_level_activation = base_level_activation

    @property
    def activation(self):
        return self.current_activation + self.base_level_activation


class ProceduralMemory(Module):

    def __init__(self, initial_schemes=None, context_match=lambda s, b: 0.0, result_match=lambda s, b: 0.0,
                 initial_base_level_activation=0.2, activation_threshold=1.0):
        super().__init__()

        self._schemes = [] if initial_schemes is None else list(initial_schemes)
        self._context_match = context_match
        self._result_match = result_match

        # Activation parameters
        self._initial_base_level_activation = initial_base_level_activation
        self._activation_threshold = activation_threshold

    @property
    def candidate_behaviors(self):
        # Find schemes with activation >= activation_threshold
        candidate_behaviors = list(filter(lambda s: s.activation >= self._activation_threshold, self._schemes))

        # if no schemes above activation threshold return a random scheme
        if len(candidate_behaviors) == 0 and len(self._schemes) > 0:
            return [np.random.choice(self._schemes, p=softmax([s.activation for s in self._schemes]))]

        return candidate_behaviors

    def receive_broadcast(self, broadcast):
        if broadcast is None:
            return

        def similarity(scheme):
            return self._context_match(scheme, broadcast) + self._result_match(scheme, broadcast)

        similarities = np.array(list(map(similarity, self._schemes)))
        current_activations = sigmoid(similarities + np.array([s.current_activation for s in self._schemes]))

        for index, activation in enumerate(current_activations):
            self._schemes[index].current_activation = activation

        self._learn(broadcast)

    def create_scheme(self, context=None, action=None, result=None):
        return Scheme(context, action, result,
                      current_activation=0.0, base_level_activation=self._initial_base_level_activation)

    def _learn(self, broadcast):
        pass


def exact_match_context_by_move(content, scheme):
    # Case 1: Content is a Move
    if isinstance(content, Move) and scheme.context == content:
        return 1.0

    # Case 2: Content is a non-Move iterable (special consideration for strings to avoid infinite recursion)
    if isinstance(content, collections.abc.Iterable) \
            and not isinstance(content, str) \
            and any(exact_match_context_by_move(c, scheme) for c in content):
        return 1.0

    # Case 3: Content is a non-Move, non-Iterable
    return 0.0


def board_position_after_move(curr_board, move):
    new_board = curr_board.copy()
    new_board[move.position] = move.mark
    return new_board


class ActionSelection(Module):
    def __init__(self):
        super().__init__()
        self.behaviors = []

    def __call__(self, behavior):
        self.behaviors.extend(behavior)

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
