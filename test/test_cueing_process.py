from unittest import TestCase
from unittest.mock import MagicMock

from common import CueingProcess, PerceptualAssociativeMemory


class TestCueingProcess(TestCase):
    def test_init(self):
        try:
            c = CueingProcess()
        except Exception as e:
            self.fail(e)

    def test_cue(self):

        pam = PerceptualAssociativeMemory()
        pam.cue = MagicMock(side_effect=lambda c: [c, 'happy'])

        content = (1, 'X')
        cue = CueingProcess()

        # Execute cue operation
        cue(content, pam)

        # Test that CueingProcess calls pam once
        pam.cue.assert_called_once_with(content)

        cued_content = next(cue)
        self.assertEqual([content, 'happy'], cued_content[0])
