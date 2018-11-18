import random


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

    def __call__(self, action):
        if action:
            self._make_move(action[1], action[0])

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
    def __init__(self):
        super().__init__()

        self.codelets = []

    def __call__(self, module):
        coalitions = []
        for codelet in self.codelets:
            coalition = codelet(module)
            coalitions.append(coalition)
        return coalitions


class StructureBuildingCodelet(Module):
    def __init__(self, is_match=lambda x: True, transform=lambda x: x):
        super().__init__()

        self.is_match = is_match
        self.transform = transform
        self.structures = []

    def __call__(self, workspace):
        new_structures = map(self.transform, filter(self.is_match, workspace))
        self.structures.extend(new_structures)

    def __next__(self):
        structures = self.structures
        self.structures = []
        return structures


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
        module(content)

        # receive activated content from cued module
        self.cued_content.append(content)

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


class ProceduralMemory(Module):
    def __init__(self, initial_schemes=None):
        super().__init__()

        self.schemes = [] if initial_schemes is None else list(initial_schemes)

    # ignore conscious broadcast for now
    def __call__(self, broadcast):
        pass

    def __next__(self):
        return None if len(self.schemes) == 0 else random.choice(self.schemes)


class ActionSelection(Module):
    def __int__(self):
        super().__init__()


class SensoryMotorSystem(Module):
    def __init__(self):
        super().__init__()

        self.motor_plans = []

    def __call__(self, behavior):
        self.motor_plans.append(behavior.action)

    def __next__(self):
        motor_plans = self.motor_plans
        self.motor_plans = []
        return motor_plans
