class Module(object):
    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()

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
        super(Environment, self).__init__()


class PerceptualAssociativeMemory(Module):
    def __init__(self):
        super(PerceptualAssociativeMemory, self).__init__()


class SensoryMemory(Module):
    def __init__(self):
        super(SensoryMemory, self).__init__()

        self.sensory_memory = []

    def __call__(self, *args, **kwargs):
        self.sensory_memory.append(args[0])  # TODO: check correct?

    def __next__(self):
        return self.sensory_memory[-1]


class AttentionCodelet(Module):
    def __init__(self):
        super(AttentionCodelet, self).__init__()


class StructureBuildingCodelet(Module):
    def __init__(self):
        super(StructureBuildingCodelet, self).__init__()


class Workspace(Module):
    def __init__(self):
        super(Workspace, self).__init__()

        self.workspace_content = []

    def __call__(self, content):
        self.workspace_content.append(content)

    def __next__(self):
        return self.workspace_content[-1]

    def __iter__(self):
        return iter(self.workspace_content)


class CueingProcess(Module):
    def __init__(self):
        super(CueingProcess, self).__init__()
