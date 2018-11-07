from common import *

# Number of cognitive cycles to execute (-1 -> forever)
N_STEPS = 1

# Module initialization
environment = Environment()
sensory_memory = SensoryMemory()
workspace = Workspace()
sb_codelet = StructureBuildingCodelet()
pam = PerceptualAssociativeMemory()
attn_codelet = AttentionCodelet()
cue = CueingProcess()


def running(step, last=None):
    """
    A boolean function for the running state of the agent.  If a last step is specific, then it will return True when
    step is less than or equal to last and False otherwise.
    :param step: current step
    :param last:  last step to run
    :return: returns True if agent should continue running; False otherwise.
    """
    return True if not last else step <= last


def run(n=None):
    """
    Main control loop of agent.  Runs for n cognitive cycles.  If n is specified, it will run forever.
    :param n: number of cognitive cycles to execute
    """
    action = None

    count = 0
    while running(count, n):
        # Update environment based on action
        environment(action)

        # Update sensory memory from next environment state
        sensory_memory(next(environment))

        # Update the pre-conscious workspace with next sensory memory state
        workspace(next(sensory_memory))

        # Update SBC from workspace
        sb_codelet(workspace)

        # Update workspace with next structure from SBC
        workspace(next(sb_codelet))

        # Cue PAM from next workspace content
        cue(next(workspace), pam)

        # Update workspace from next result from cue
        workspace(next(cue))

        # Update attention codelet from workspace
        attn_codelet(workspace)

        count += 1


if __name__ == '__main__':
    run(N_STEPS)
