import collections
import random

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax

SensoryScene = collections.namedtuple('SensoryScene', ['observation', 'outcome', 'gameover'])

EPSILON = 0.000001

class SensoryMemory:
    def __init__(self, feature_detectors):
        self.sensory_scene = []
        self._feature_detectors = feature_detectors

    def receive_sensors(self, environment):
        obs, reward, done, info = environment.step(action=None)

        sensory_scene_content = []

        for fd in self._feature_detectors:
            sensory_scene_content.append(fd.apply((obs, reward)))

        self.sensory_scene = SensoryScene(observation=sensory_scene_content,
                                          outcome=reward, gameover=done)


class FeatureDetector:
    def __init__(self, concept, similarity_metric):
        self._concept = concept
        self._similarity_metric = similarity_metric

    @property
    def concept(self):
        return self._concept

    def apply(self, content):
        return CognitiveContent(self._concept,
                                activation=self._similarity_metric(content))


class PerceptualAssociativeMemory:
    def __init__(self, initial_concepts):
        self._concepts = initial_concepts

    def receive_broadcast(self, broadcast):
        pass

    def receive_cue(self, content):
        cued_content = []

        # "Feature Detectors"
        if isinstance(SensoryScene, content):
            board = SensoryScene.observation
            gameover = SensoryScene.gameover
            outcome = SensoryScene.outcome

            cued_content.append([self._concepts["board"], board])
            cued_content.append([])


            return CognitiveContent()

        # Recognize winning board (add feeling node with positive affective valence)

        # Recognize losing board (add feeling node with negative affective valence)

        return None


class CognitiveContent:

    def __init__(self, content, affective_valence=0.0, activation=0.0):
        self.content = content

        # Activation and Incentive Salience Parameters
        self.base_level_activation = 0.0
        self.base_level_incentive_salience = 0.0
        self.current_activation = activation
        self.current_incentive_salience = 0.0

        # Affective Valence Sign (either +1.0 or -1.0 when set)
        # --- only used for feeling nodes
        self.affective_valence = affective_valence

        # Metadata
        self.virtual = False
        self.sbc_id = None

    @property
    def salience(self):
        return self.activation + self.incentive_salience

    @property
    def current_activation(self):
        # TODO: Need to iterate over all content to calculate the total activation
        return self._current_activation

    @current_activation.setter
    def current_activation(self, curr_act):
        self._current_activation = curr_act

    @property
    def base_level_activation(self):
        # TODO: Need to iterate over all content to calculate the total activation
        return self._bla

    @base_level_activation.setter
    def base_level_activation(self, bla):
        self._bla = bla

    @property
    def activation(self):
        # TODO: Need to iterate over all content to calculate the total activation
        return self.current_activation + self.base_level_activation

    @property
    def incentive_salience(self):
        # TODO: Need to iterate over all content to calculate the total incentive salience
        return self.current_incentive_salience + self.base_level_incentive_salience

    def __iter__(self):
        return iter(self.content)

    def __str__(self):
        return str(self.content.__class__)+'('+str(self.content)+')'

class StructureBuildingCodelet:
    def __init__(self, select=lambda x: True, transform=lambda x: x, id=None):
        self._select = select
        self._transform = transform

        self.id = id

    def apply(self, workspace):
        return list(map(self._transform, filter(self._select, workspace.csm)))

class Decay:
    def __init__(self, content):
       for elem in content:
           elem.current_activation -= .5

class GarbageCollector:
    def __init__(self, content, activation_type=CognitiveContent.base_level_activation):
        for elem in content:
            # TODO: replace current activation with activation_type
            # TODO:     'CognitiveContent' object has no attribute 'activation_type'
            if elem and activation_type.__get__(elem) < EPSILON :
                content.remove(elem)


class CurrentSituationalModel:
    def __init__(self):
        self._content = []

        self.perceptual_scene = None

    def receive_content(self, content):

        GarbageCollector(self._content, CognitiveContent.current_activation)

        if content is None or len(content) == 0:
            return

        self._content.extend(content)

    @property
    def content(self):
        return self._content

    def receive_sensory_scene(self, scene):

        # TODO: Should be distinct from the sensory scene -- containing both real and virtual content
        # TODO: On calling receive_sensory_scene, the content from the sensory_scene should be integrated
        # TODO: Into the perceptual scene, not overwrite it
        self.receive_content(scene.observation)

    def __iter__(self):
        return iter(self._content)


class ConsciousContentsQueue:
    def receive_broadcast(self, broadcast):
        pass


class Workspace:
    initial_current_activation = 0.5

    def __init__(self):
        self.csm = CurrentSituationalModel()
        self.ccq = ConsciousContentsQueue()


class CueingProcess:
    def __init__(self, cueable_modules=None):
        self.cueable_modules = cueable_modules or []

    def process(self, workspace):
        # TODO: Currently implemented as 2 passes: 1 for perceptual scene and
        # another for generated content in the csm.  Need to rethink this later.
        for module in self.cueable_modules:
            module.receive_cue(workspace.csm.perceptual_scene)

        for content in workspace.csm:
            for module in self.cueable_modules:
                cued_content = module.receive_cue(content)

                if cued_content is not None:
                    workspace.csm.receive_cue([content, cued_content])


class AttentionCodelet:
    def __init__(self, select=lambda x: True):
        self._select = select

    def apply(self, workspace):
        return list(filter(self._select, workspace.csm.content))


class Coalition:
    def __init__(self, content, attn_codelets):
        self.content = content
        self.attn_codelets = attn_codelets


    @property
    def activation(self):
        # TODO: calculate activation on coalition...
        # TODO: need to verify formula, but I think it minimally includes:
        # TODO: (1) base-level-activation (of attn codelet)
        # TODO: (2) similarity of content to attn codelet's concerns
        # TODO: (3) salience of cognitive content in structure
        return sum([elem.salience for elem in self.content])

    def __repr__(self):
        def recursive_parse(content):
            if isinstance(content, list):
                return str([recursive_parse(elem) for elem in content])
            elif isinstance(content, CognitiveContent) or isinstance(content, Coalition):
                return str(content.__class__)+'('+recursive_parse(content.content)+')'
            elif isinstance(content, str):
                return content
            else:
                return str(content)
        return recursive_parse(self)


class CoalitionManager:
    def __init__(self):
        # Each element will be an attention codelet + the content that the attn codelet is advocating for
        self._candidates = []

    def receive(self, attn_codelet, content):
        if content is None or len(content) == 0:
            return

        self._candidates.append(Coalition(content, attn_codelet))

    @property
    def coalitions(self):
        # TODO: add more sophisticated implementation based on shared content / concerns
        coalitions = self._candidates
        new_content = []
        for coalition in self._candidates:
            if coalition.activation > .99:
                new_content.extend(coalition.content)
        coalitions.append(Coalition(new_content, AttentionCodelet(lambda x: True)))
        self._candidates=[]
        return coalitions


class GlobalWorkspace:
    def __init__(self):
        self.coalitions = []

    def receive_coalitions(self, coalitions):
        if coalitions is None or len(coalitions) == 0:
            return

        self.coalitions.extend(coalitions)

        GarbageCollector(self.coalitions, Coalition.activation)

    @property
    def broadcast(self):
        # TODO: Check Triggers
        if len(self.coalitions) == 0:
            return None

        return max(self.coalitions, key=lambda c: c.activation)


Action = collections.namedtuple('Action', ['type', 'value'])


class Scheme:
    def __init__(self, context=None, action=None, result=None, current_activation=0.0, base_level_activation=0.0):
        self.context = context
        self.action = action
        self.result = result

        self.current_activation = current_activation
        self.base_level_activation = base_level_activation

    @property
    def activation(self):
        return self.current_activation + self.base_level_activation


class ProceduralMemory:

    def __init__(self, initial_schemes=None, context_match=lambda s, b: 0.0, result_match=lambda s, b: 0.0,
                 activation_threshold=0.6):
        self._schemes = [] if initial_schemes is None else list(initial_schemes)
        self._context_match = context_match
        self._result_match = result_match

        # Activation parameters
        self.activation_threshold = activation_threshold

        # Use for learning
        self.recently_selected_behaviors = collections.deque(maxlen=3)

    @property
    def candidate_behaviors(self):
        # Find schemes with activation >= activation_threshold
        candidate_behaviors = list(filter(lambda s: s.activation >= self.activation_threshold, self._schemes))

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

    def receive_selected_behavior(self, behavior):
        self.recently_selected_behaviors.append(behavior)

    def create_scheme(self, context=None, action=None, result=None):
        return Scheme(context, action, result, current_activation=0.0, base_level_activation=0.2)

    def _learn(self, broadcast):

        # TODO: Need to compare broadcast to results in recently selected behaviors
        pass


def exact_match_by_board(content, scheme):
    return True

    # # Case 1: Content is a Board
    # if isinstance(content, Move) and scheme.context == content:
    #     return 1.0
    #
    # # Case 2: Content is a non-Move iterable (special consideration for strings to avoid infinite recursion)
    # if isinstance(content, collections.abc.Iterable) \
    #         and not isinstance(content, str) \
    #         and any(exact_match_by_board(c, scheme) for c in content):
    #     return 1.0
    #
    # # Case 3: Content is a non-Move, non-Iterable
    # return 0.0


class ActionSelection:
    def __init__(self):
        self.behaviors = []

    def receive_behaviors(self, behaviors):
        self.behaviors = behaviors

    @property
    def selected_behavior(self):
        # TODO: Check Triggers
        if len(self.behaviors) == 0:
            return None

        # TODO: need more sophisticated action selection that includes incentive salience of results
        # TODO: to determine the most "valueable" action to choose.
        return max(self.behaviors, key=lambda c: c.activation)


MotorCommand = collections.namedtuple('MotorCommand', ['actuator', 'value'])


class MotorPlanTemplate:
    def __init__(self, motor_commands, triggers, choice_function):
        self.motor_commands = motor_commands
        self.triggers = triggers
        self.choice_function = choice_function

    # No potential for online control for this agent, but I am including sensory_scene in the method
    # signature for completeness
    def choose_motor_command(self, sensory_scene):
        # Apply triggers on motor commands
        candidates = [command for command in self.motor_commands for trigger in self.triggers if trigger(command)]

        # Choose single motor command
        return self.choice_function(candidates)

    def instantiate(self, value):
        self.choice_function = lambda x: value


class MotorPlanTemplate:
    def __init__(self, motor_commands, triggers, choice_function):
        self.motor_commands = motor_commands
        self.triggers = triggers
        self.choice_function = choice_function

    # No potential for online control for this agent, but I am including sensory_scene in the method
    # signature for completeness
    def choose_motor_command(self, sensory_scene):
        # Apply triggers on motor commands
        candidates = [command for command in self.motor_commands for trigger in self.triggers if trigger(command)]

        # Choose single motor command
        return self.choice_function(candidates)

    def instantiate(self, value):
        return self


class SensoryMotorSystem:
    def __init__(self, motor_plan_templates):
        self.motor_plan_templates = motor_plan_templates
        self.motor_plan = None

    def receive_selected_behavior(self, behavior):
        if behavior is None:
            raise ValueError('Selected behavior cannot be None')

        # Instantiate a motor plan
        if behavior.action.type == 'move':
            self.motor_plan = self.motor_plan_templates[behavior.action.value].instantiate(behavior.action.value)
        else:
            self.motor_plan = self.motor_plan_templates[behavior.action.type].instantiate(behavior.action.value)
