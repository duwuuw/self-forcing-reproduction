import unittest

from scripts.self_forcing_runtime_alignment import NativeCheckpointPolicy


class NativeCheckpointPolicyTests(unittest.TestCase):
    def test_cross_attention_cache_is_disabled_only_inside_checkpoint(self):
        policy = NativeCheckpointPolicy()

        self.assertTrue(policy.disable_cross_attention_cache(True))
        self.assertFalse(policy.disable_cross_attention_cache(False))
        self.assertEqual(policy.cross_attention_bypasses, 1)


if __name__ == "__main__":
    unittest.main()
