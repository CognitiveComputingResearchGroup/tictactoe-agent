from common import *

import gym
import gym_tictactoe # Needed to add 'TicTacToe-v0' into gym registry

# Number of cognitive cycles to execute (None -> forever)
N_STEPS = None

# Module initialization
sensory_memory = SensoryMemory()
workspace = Workspace()
pam = PerceptualAssociativeMemory()
global_workspace = GlobalWorkspace()
procedural_memory = ProceduralMemory(
    initial_schemes=[Scheme(context=None, action=position, result=None) for position in range(9)],
    context_match=exact_match_by_board)
action_selection = ActionSelection()

# Create Codelets
sb_codelets = []
attn_codelets = [AttentionCodelet()]

cueable_modules = []
broadcast_recipients = []

# Initialize Cueing Process
cue = CueingProcess(cueable_modules)
coalition_manager = CoalitionManager()


def running(step, last=None):
    """
    A boolean function for the running state of the agent.  If a last step is specified, then it will return True when
    step is less than or equal to last and False otherwise.
    :param step: current step
    :param last:  last step to run
    :return: returns True if agent should continue running; False otherwise.
    """
    return True if last is None else step < last


def run(environment, n=None, render=True):
    """
    Main control loop of agent.  Runs for n cognitive cycles.  If n is not specified, it will run forever.
    :param n: number of cognitive cycles to execute
    """
    count = 0

    while running(count, n):

        # Display environment state for human consumption
        if render:
            print(environment.render())

        # Process sensors into modality specific representations
        sensory_memory.receive_sensors(environment)

        # Integrate sensory scene into workspace
        workspace.csm.receive_content(sensory_memory.sensory_scene)

        # Structure building codelets scan the workspace, potentially creating new content
        sbc_content = []
        for codelet in sb_codelets:
            sbc_content.append(codelet.process(workspace))
        workspace.csm.receive_content(sbc_content)

        # Cueing process
        cued_content = cue.process(workspace)
        workspace.csm.receive_content(cued_content)

        # Attention codelets scan workspace and select content of interest
        for codelet in attn_codelets:
            coalition_manager.receive(codelet, codelet.process(workspace))

        global_workspace.receive_coalitions(coalition_manager.coalitions)

        # Conscious broadcast retrieved from global workspace
        broadcast = global_workspace.broadcast
        if broadcast is not None:

            # Broadcast sent to all broadcast recipients
            for module in broadcast_recipients:
                module.receive_broadcast(broadcast)

            action_selection.receive_behaviors(procedural_memory.candidate_behaviors)

            # Retrieve next action and associated expectation codelet from action selection
            selected_behavior = action_selection.selected_behavior
            if selected_behavior is not None:

                # Add expectation codelet for selected behavior
                attn_codelets.append(AttentionCodelet(select= lambda x: x == selected_behavior.result))

                # Execute action against environment
                environment.step(selected_behavior.action)

        count += 1

    return count


if __name__ == '__main__':
    environment = gym.make('TicTacToe-v0')
    environment.reset()

    run(environment, n=N_STEPS)
