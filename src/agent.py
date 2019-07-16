from common import *
from graphics import initialize, draw

import gym
import sys
sys.path.append("..")

import gym_tictactoe  # Needed to add 'TicTacToe-v0' into gym registry

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Number of cognitive cycles to execute (None -> forever)
N_STEPS = 1000

# TODO: can be removed if we figure out logging and another control on graphics.
experimenting = False

# Module initialization
#TODO: 'happy' and 'sad' should be interpretive feeling nodes (like 'sweetness' related feeling node)
feature_detectors = [FeatureDetector("happy", lambda x: x[1] if x[1] > 0 else 0.0),
                     FeatureDetector("sad", lambda x: abs(x[1]) if x[1] < 0 else 0.0)
                    ]
mark_dict = {1: 'X', -1: 'O', 0: 'B'}
mark_dict_r = {'X':1 , 'O':-1, 'B':0}

def lambda_mark_detector(mark, pos):
    return lambda x: (x[0][pos] == mark)*1.0

mark_detectors = [FeatureDetector(mark_dict[mark]+'_'+str(pos),
                                   lambda_mark_detector(mark, pos))
                       for pos in range(9) for mark in [1, -1, 0]
                  ]

feature_detectors += mark_detectors

sensory_memory = SensoryMemory(feature_detectors=feature_detectors)
workspace = Workspace()

# Feature Detectors

pam = PerceptualAssociativeMemory(initial_concepts=[
                                                    #TODO: affective_valence is really valence
                                                    FeelingNode("happy", valence=1.0),
                                                    FeelingNode("sad", valence=-1.0),
                                                    ])
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
attn_codelets = [AttentionCodelet(lambda x: x.content == "happy",
                                  tag="happy", domain=workspace.csm),
                 AttentionCodelet(lambda x: x.content == "sad",
                                  tag="sad", domain=workspace.csm)
                ]
position_nodes = [mark_dict[mark]+'_'+str(pos)
                  for mark in [1, 0, -1] for pos in range(9)]


def lambda_mark_attn_codelet(pos_code):
    return lambda x: x.content == pos_code and x.activation > .99


default_attn_codelet = [AttentionCodelet(lambda x: x.activation > .99,
                                         tag='default_attn_codelet',
                                         base_level_activation=.9,
                                         domain=workspace.csm
                                         )]

mark_attn_codelets = [AttentionCodelet(lambda_mark_attn_codelet(pos_code),
                                       tag=pos_code, domain=workspace.csm)
                      for pos_code in position_nodes
                     ]
#attn_codelets += mark_attn_codelets
attn_codelets += default_attn_codelet
cueable_modules = [pam]
broadcast_recipients = [procedural_memory]

# Initialize Cueing Process
cue_process = CueingProcess(cueable_modules)
coalition_manager = CoalitionManager()

removal_activation = {CurrentSituationalModel: 'current_activation',
                      GlobalWorkspace: 'activation',
                      ProceduralMemory: 'base_level_activation',
                      list: 'base_level_activation' # for attention codelets
                     }


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
    total_reward = 0

    motor_command = None

    while running(count, n):

        # Display environment state for human consumption
        if render:
            environment.render()

        obs, reward, done, info = environment.step(motor_command)
        total_reward += reward if abs(reward) > .99 else 0.0
        # Process sensors into modality specific representations
        #draw(sender=environment, receiver=sensory_memory, content= (obs,reward))
        sensory_memory.receive_sensors((obs, reward))

        pam.receive_features(sensory_memory.detected_features)


        # Integrate sensory scene into workspace
        workspace.csm.receive_sensory_scene(sensory_memory.sensory_scene)

        workspace.csm.receive_content(pam.percept)
        '''
        # Structure building codelets scan the workspace, potentially creating new content
        sbc_content = []
        for codelet in sb_codelets:
            sbc_content.append(codelet.apply(workspace))
        workspace.csm.receive_content(sbc_content)
        '''
        # Cueing process
        cued_content = cue_process.process(workspace)
        workspace.csm.receive_content(cued_content)

        # Attention codelets scan workspace and select content of interest
        for codelet in attn_codelets:
            coalition_manager.receive(codelet, codelet.apply())

        global_workspace.receive_coalitions(coalition_manager.coalitions)

        # Conscious broadcast retrieved from global workspace
        broadcast = global_workspace.broadcast
        if broadcast is not None:

            # Broadcast sent to all broadcast recipients
            for module in broadcast_recipients:
                module.receive_broadcast(broadcast)

            candidate_behaviors = procedural_memory.candidate_behaviors
            action_selection.receive_behaviors(candidate_behaviors)

            # Process selected behavior
            selected_behavior = action_selection.selected_behavior

            procedural_memory.receive_selected_behavior(selected_behavior)

            if selected_behavior is not None:
                # Expectation codelet created from selected behavior
                if selected_behavior.result is not None:
                    attn_codelets.append(ExpectationCodelet(scheme=selected_behavior,
                                                            select=lambda x: x in selected_behavior.result and x.activation > 0.0,
                                                            tag='exp',
                                                            domain=workspace.csm.perceptual_scene))

                sensory_motor_system.receive_selected_behavior(selected_behavior)

                motor_plan = sensory_motor_system.motor_plan
                motor_command = motor_plan.choose_motor_command(sensory_memory.sensory_scene)

                # Action execution - conceptually we can think of this as 2 actuators:
                # a move actuator and a reset actuator

        logger.debug(('broadcast: ', broadcast))
        logger.debug(('attn_codelets: ', attn_codelets))
        logger.debug(('selected_behavior: ', selected_behavior))
        logger.debug(('motor_command: ', motor_command))
        if render:
            print('attn_codelets: ', attn_codelets)
            print('broadcast: ', broadcast)
            print('selected_behavior: ', selected_behavior)
            print('motor_command: ', motor_command)
            print('Strategy:')
            for i, scheme in enumerate(sorted(procedural_memory.content, key=lambda x: x.base_level_activation, reverse=True)):
                if i>5:
                    break
                print(get_board_string_from_context(scheme.context), scheme.action.value)
        logger.log(logging.NOTSET, ('schemes: '))
        for scheme in procedural_memory._schemes:
            if scheme.current_activation > .5:
                logger.log(logging.NOTSET, scheme)

        if render:
            #draw([sensory_memory, workspace.csm, global_workspace])
            import time
            time.sleep(2)

        #housekeeping
        Decay(workspace.csm.content)
        Decay(attn_codelets)
        Decay(pam.content)
        Forget(procedural_memory.content)
        #Forget(procedural_memory.content, function=lambda x: norm.pdf(x, loc=0.5, scale=0.12)*(1.0/50))
        for module in [workspace.csm, global_workspace, attn_codelets, procedural_memory]:
            GarbageCollector(module, removal_activation[type(module)])


        count += 1
        print(count)

    return count, total_reward

def get_board_string_from_context(context):
    board = [0]*9
    if context is not None:
        for elem in context:
            if not isinstance(elem, FeelingNode):
                board[int(str(elem)[2])] = mark_dict_r[str(elem)[0]]
    from gym_tictactoe.envs.tictactoe_env import Board
    return str(Board.from_list(board))

if __name__ == '__main__':
    environment = gym.make('TicTacToe-v0')
    environment.reset()
    print(run(environment, n=N_STEPS, render=not experimenting))


