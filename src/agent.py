from common import *

import gym
import sys
sys.path.append("..")
import gym_tictactoe  # Needed to add 'TicTacToe-v0' into gym registry

# Number of cognitive cycles to execute (None -> forever)
N_STEPS = None

# Module initialization
feature_detectors = [FeatureDetector("win", lambda x: (x[1] > 0)*1.0),
                     FeatureDetector("lose", lambda x: (x[1] < 0)*1.0)
                    ]
mark_dict = {1: 'X', -1: 'O', 0: 'B'}

def lambda_board_detector(mark, pos):
    return lambda x: (x[0][pos] == mark)*1.0

board_detectors = [FeatureDetector(mark_dict[mark]+'_'+str(pos),
                                   lambda_board_detector(mark, pos))
                       for pos in range(9) for mark in [1, -1, 0]
                  ]

feature_detectors += board_detectors

sensory_memory = SensoryMemory(feature_detectors=feature_detectors)
workspace = Workspace()

# Feature Detectors

pam = PerceptualAssociativeMemory(initial_concepts={"board": CognitiveContent("board"),
                                                    "happy": CognitiveContent("happy", affective_valence=1.0),
                                                    "sad": CognitiveContent("sad", affective_valence=-1.0),
                                                    "gameover": CognitiveContent("gameover"),
                                                    "win": CognitiveContent("win"),
                                                    "lose": CognitiveContent("lose"),
                                                    "draw": CognitiveContent("draw")})
global_workspace = GlobalWorkspace()

# Initial Schemes
move_schemes = [Scheme(context=None, action=Action('move', position), result=None) for position in range(9)]

procedural_memory = ProceduralMemory(initial_schemes= move_schemes, context_match=exact_match_by_board)
action_selection = ActionSelection()

# Motor Plan Templates
reset_mpt = MotorPlanTemplate(motor_commands=[MotorCommand(actuator='reset', value=None)],
                              triggers=[lambda mc: True],
                              choice_function=lambda mcs: random.choice(mcs))

mp_templates = {i: MotorPlanTemplate(motor_commands=[MotorCommand(actuator='move', value=i)],
                             triggers=[lambda mc: True],
                             choice_function=lambda mcs: random.choice(mcs))
                                    for i in range(10) }

mp_templates['reset'] = reset_mpt

sensory_motor_system = SensoryMotorSystem(motor_plan_templates=mp_templates)

# Create Codelets
sb_codelets = []
attn_codelets = [AttentionCodelet(lambda x: [x.content, ] if x.content == "happy"
                                                                                else None),
                 AttentionCodelet(lambda x: [x.content, ] if x.content == "sad"
                                                                                else None)
                ]
position_nodes = [mark_dict[mark]+'_'+str(pos)
                  for mark in [1,0,-1] for pos in range(9)]

def lambda_position_attn_codelet(pos_code):
    return lambda x: x if x.content == pos_code and x.activation > .99 else None

position_attn_codelets = [AttentionCodelet(lambda_position_attn_codelet(pos_code))
                          for pos_code in position_nodes
                          ]
attn_codelets+=position_attn_codelets
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
            environment.render()

        # Process sensors into modality specific representations
        sensory_memory.receive_sensors(environment)

        # Integrate sensory scene into workspace
        workspace.csm.receive_sensory_scene(sensory_memory.sensory_scene)

        # Structure building codelets scan the workspace, potentially creating new content
        sbc_content = []
        for codelet in sb_codelets:
            sbc_content.append(codelet.apply(workspace))
        workspace.csm.receive_content(sbc_content)

        # Cueing process
        cued_content = cue.process(workspace)
        workspace.csm.receive_content(cued_content)

        # Attention codelets scan workspace and select content of interest
        for codelet in attn_codelets:
            coalition_manager.receive(codelet, codelet.apply(workspace))

        global_workspace.receive_coalitions(coalition_manager.coalitions)

        # Conscious broadcast retrieved from global workspace
        broadcast = global_workspace.broadcast
        print(broadcast)
        if broadcast is not None:

            # Broadcast sent to all broadcast recipients
            for module in broadcast_recipients:
                module.receive_broadcast(broadcast)

            action_selection.receive_behaviors(procedural_memory.candidate_behaviors)

            # Process selected behavior
            selected_behavior = action_selection.selected_behavior
            if selected_behavior is not None:
                # Expectation codelet created from selected behavior
                attn_codelets.append(AttentionCodelet(select=lambda x: x == selected_behavior.result))

                sensory_motor_system.receive_selected_behavior(selected_behavior)

                motor_plan = sensory_motor_system.motor_plan
                motor_command = motor_plan.choose_motor_command(sensory_memory.sensory_scene)

                # Action execution - conceptually we can think of this as 2 actuators:
                # a move actuator and a reset actuator
                if motor_command.actuator == 'reset':
                    environment.reset()
                elif motor_command.actuator == 'move':
                    environment.step(motor_command.value)
                else:
                    raise ValueError('Unexpected actuator value')

                Decay(workspace.csm.content)
        count += 1

    return count


if __name__ == '__main__':
    environment = gym.make('TicTacToe-v0')
    environment.reset()

    run(environment, n=N_STEPS)
