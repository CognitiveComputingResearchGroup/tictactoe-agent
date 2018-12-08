from common import *

# Number of cognitive cycles to execute (None -> forever)
N_STEPS = 1

# Module initialization
environment = Environment()
sensory_memory = SensoryMemory()
workspace = Workspace()
pam = PerceptualAssociativeMemory()
cue = CueingProcess()
global_workspace = GlobalWorkspace()
procedural_memory = ProceduralMemory(initial_schemes=[Scheme(action=(pos, 'X')) for pos in range(9)])
action_selection = ActionSelection()
sensory_motor_system = SensoryMotorSystem()

sb_codelets = [StructureBuildingCodelet()]
attn_codelets = [AttentionCodelet()]


def running(step, last=None):
    """
    A boolean function for the running state of the agent.  If a last step is specified, then it will return True when
    step is less than or equal to last and False otherwise.
    :param step: current step
    :param last:  last step to run
    :return: returns True if agent should continue running; False otherwise.
    """
    return True if last is None else step < last


def run(n=None):
    """
    Main control loop of agent.  Runs for n cognitive cycles.  If n is not specified, it will run forever.
    :param n: number of cognitive cycles to execute
    """
    count = 0
    while running(count, n):
        # Update sensory memory from next environment state
        sensory_memory(next(environment))

        # Update the pre-conscious workspace with next sensory memory state
        workspace(next(sensory_memory))

        # Structure building codelets scan and update pre-conscious workspace
        for codelet in sb_codelets:
            codelet(workspace)
            workspace(next(codelet))

        # Cue PAM from next workspace content
        cue(next(workspace), pam)

        # Update pre-conscious workspace with cued memories
        for content in next(cue):
            workspace(content)

        # Attention codelets scan pre-conscious workspace and add coalitions to global workspace
        for codelet in attn_codelets:
            codelet(workspace)
            global_workspace(next(codelet))

        # Conscious broadcast retrieved from global workspace
        broadcast = next(global_workspace)

        # Update procedural memory based on conscious broadcast
        procedural_memory(broadcast)

        # Update action selection from procedural memory
        action_selection(next(procedural_memory))

        # Retrieve next action and associated expectation codelet from action selection
        behavior, exp_codelet = next(action_selection)

        # Add expectation codelet to set of attention codelets
        if exp_codelet:
            attn_codelets.append(exp_codelet)

        # Update sensory motor memory based on selected action
        sensory_motor_system(behavior)

        # Update environment from sensory motor system's motor plan
        environment(next(sensory_motor_system))

        count += 1

    return count


if __name__ == '__main__':
    run(N_STEPS)
