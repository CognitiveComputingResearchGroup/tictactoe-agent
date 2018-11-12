class Module(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        Update internal state of module using supplied arguments.
        """
        raise NotImplementedError()

    def __getitem__(self, key):
        """
        Retrieve the specified item.

        A KeyError is raised if the key doesn't exist.
        A TypeError is raised if the key is of an incorrect type.
        """
        raise NotImplementedError()

    def __delitem__(self, key):
        """
        Removes the specified item.

        A KeyError is raised if the key doesn't exist.
        A TypeError is raised if the key is of an incorrect type.
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

    def __contains__(self, item):
        """
        Returns True if item is in module; False otherwise.
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
    def __init__(self):
        super().__init__()


class Workspace(Module):
    def __init__(self):
        super().__init__()

        self.workspace_content = []

    def __call__(self, content):
        self.workspace_content.append(content)

    def __next__(self):
        return self.workspace_content[-1]

    def __iter__(self):
        return iter(self.workspace_content)


class CueingProcess(Module):
    def __init__(self):
        super().__init__()


class GlobalWorkspace(Module):
    def __int__(self):
        super().__init__()


class ProceduralMemory(Module):
    def __int__(self):
        super().__init__()


class ActionSelection(Module):
    def __int__(self):
        super().__init__()


class SensoryMotorMemory(Module):
    def __int__(self):
        super().__init__()
