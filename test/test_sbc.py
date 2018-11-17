from unittest import TestCase

from common import StructureBuildingCodelet, Workspace


class TestStructureBuildingCodelet(TestCase):
    def test_init(self):
        try:
            c = StructureBuildingCodelet()
            c = StructureBuildingCodelet(is_match=lambda x: False)
            c = StructureBuildingCodelet(action=lambda x: x + 1)
            c = StructureBuildingCodelet(is_match=lambda x: False, action=lambda x: x + 1)
        except Exception as e:
            self.fail(e)

    def test_call(self):

        # With default is_match and action
        try:
            c = StructureBuildingCodelet()
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                c(w)

        except Exception as e:
            self.fail(e)

        # With custom is_match and action
        try:
            c = StructureBuildingCodelet(is_match=lambda x: x < 3, action=lambda x: x + 1)
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            # invoke __call__ several times (testing for exceptions)
            for v in range(100):
                c(w)

        except Exception as e:
            self.fail(e)

    def test_next(self):
        try:
            w = Workspace()

            # Add content to the Workspace
            for i in range(10):
                w(i)

            for v in range(10):
                c = StructureBuildingCodelet(is_match=lambda x: x == v, action=lambda x: x + 1)
                c(w)
                actual = next(c)
                self.assertEqual(v + 1, actual)
        except Exception as e:
            self.fail(e)
