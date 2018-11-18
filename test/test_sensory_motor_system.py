from unittest import TestCase

from common import SensoryMotorSystem, Scheme


class TestSensoryMotorSystem(TestCase):
    def test_init(self):
        try:
            sms = SensoryMotorSystem()
        except Exception as e:
            self.fail(e)

    def test_call(self):
        try:
            sms = SensoryMotorSystem()

            # invoke __call__ several times (testing for exceptions)
            for a in range(100):
                sms(Scheme(action=a))

        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            sms = SensoryMotorSystem()

            # next should return a list containing the action from the behavior passed
            # in the last call to sms
            for a in range(10):
                sms(Scheme(action=a))  # Update SensoryMotorSystem
                actual = next(sms)
                self.assertTrue(a in actual)
        except Exception as e:
            self.fail(e)
