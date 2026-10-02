"""Semantic and metric regressions for synchronized sources and new cases."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from agent_search import Session, ROOT
from pilot import FAST, evaluate_one
from search_contract import build_input, stats, unpack, pack, channel_from_rep, equivalent, SCORE_ID


class SyncedBaselineTests(unittest.TestCase):
    def test_upstream_default_preserved(self):
        path = ROOT / 'originals/local_20260917/channelIR_gatecost_rewrite_search.py'
        spec = importlib.util.spec_from_file_location('upstream_gatecost', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        ir = channel_from_rep(build_input('random_pauli', n=2, seed=4))
        expected = old.gatecost_rewrite_search(ir, max_steps=3, beam_width=2)
        actual = ir.gatecost_rewrite_search(max_steps=3, beam_width=2)
        self.assertEqual(actual['termination'], expected['termination'])
        np.testing.assert_array_equal(actual['A_final'], expected['A_final'])
        self.assertEqual((actual['initial_support'], actual['final_support']), (18, 17))

    def test_fractional_score_not_truncated(self):
        ir = channel_from_rep(build_input('hrw', n=2))
        result = ir.gatecost_rewrite_search(max_steps=2, beam_width=2,
            score_fn=lambda A: float(np.sum(np.sum(abs(A), axis=1)**2)) / 1000,
            score_model_id='test_fractional_lambda')
        self.assertGreater(result['final_cost'], 0)
        self.assertLess(result['final_cost'], result['initial_cost'])
        self.assertAlmostEqual(result['final_cost'], .002)

    def test_scrambled_has_same_upper_landscape(self):
        original = stats(build_input('hrw', n=4))
        for seed in range(5):
            rep = build_input('hrw_scrambled', n=4, seed=seed)
            self.assertEqual(stats(rep), original)
            ir = channel_from_rep(rep)
            labels = ir._coeff_matrix()[1]
            result = ir.gatecost_rewrite_search(max_steps=4, beam_width=2,
                score_fn=lambda A: stats(pack(A, labels))['upstream_score'], score_model_id=SCORE_ID)
            final = pack(result['A_final'], result['labels'])
            equivalent(final, rep)
            self.assertLess(stats(final)['upstream_score'], original['upstream_score'])

    def test_paper_score_scale_includes_lambda_once(self):
        rep = build_input('hrw', n=2)
        A, labels = unpack(rep)
        self.assertAlmostEqual(stats(pack(2*A, labels))['upstream_score'],
                               4*stats(rep)['upstream_score'])

    def test_hrw_full_nam_pair_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Session.create(Path(folder) / 'run', family='hrw', n=2,
                                     wall_seconds=180, cpu_seconds=120, lowering_calls=2)
            root = session.call('observe')['root_repr_id']
            args = {'repr_id': root, 'beam_width': 2, 'max_steps': 2}
            result = session.call('run_cost_guided_rewrite', args)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['termination']['score_model_id'], SCORE_ID)
            self.assertTrue(session.call('lookup_cache', {'operation': 'run_cost_guided_rewrite',
                                                         'args': args})['hit'])
            before = evaluate_one(session, root, FAST, 0, 60)
            after = evaluate_one(session, result['repr_id'], FAST, 0, 60)
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            self.assertLess(after['expected_gate_count'], before['expected_gate_count'])
            print(json.dumps({'smoke': 'HRW-2', 'C_before': before['expected_gate_count'],
                              'C_after': after['expected_gate_count']}), flush=True)


if __name__ == '__main__':
    unittest.main()
