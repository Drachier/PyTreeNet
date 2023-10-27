import unittest

import pytreenet as ptn

class TestTreeStructureInit(unittest.TestCase):
    def test_init(self):
        ts = ptn.TreeStructure()
        self.assertEqual(None, ts.root_id)
        self.assertEqual(0, len(ts.nodes))