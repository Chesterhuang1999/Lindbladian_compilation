"""Contract and small-instance semantic tests; run in the remote runtime."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from agent_search import ROOT, Session
from src.search_contract import build_input, equivalent, pack, stats, unpack
from src.search_worker import evaluate, search
from pilot import FAST


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='channelir-agent-test-')
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def session(self, **kw):
        return Session.create(self.base / 'run', wall_seconds=300, cpu_seconds=300,
                              lowering_calls=10, **kw)

    def compile_direct(self, rep):
        args = {'rep': rep, 'root': rep, 'kraus_ids': 'all', 'config': FAST, 'seed_index': 0}
        bundle = search(args)
        args.update(plans=bundle['plans'], artifact_dir=str(self.base))
        # Exact evaluation may never invoke either entry point for a new search.
        with patch('block_encoding_new_fixed.search_for_pauli_placement', side_effect=AssertionError('hidden search')), \
             patch('block_encoding_new_fixed.search_placement_monitored', side_effect=AssertionError('hidden monitored search')):
            result = evaluate(args)
        self.assertTrue(result['verified'])
        self.assertEqual(result['semantic_validation']['full_channel_check'], 'exhaustive_input_basis')
        self.assertAlmostEqual(result['expected_gate_count'], result['gates']/result['success_probability'])
        return bundle, result

    def test_tfim_physical_coefficients_and_channel_semantics(self):
        from qiskit.quantum_info import SparsePauliOp
        from search_contract import TOL
        rep = build_input(n=2)
        A, labels = unpack(rep)
        mats = [SparsePauliOp.from_list([(p,c) for p,c in zip(labels,row) if abs(c)>TOL]).to_matrix() for row in A]
        X = np.array([[0,1],[1,0]], complex)
        Y = np.array([[0,-1j],[1j,0]], complex)
        Z = np.diag([1,-1])
        H = -np.kron(X,np.eye(2))-np.kron(np.eye(2),X)-np.kron(Z,Z)
        jumps = [np.sqrt(.1)/2*np.kron(np.eye(2),X-1j*Y), np.sqrt(.1)/2*np.kron(X-1j*Y,np.eye(2))]
        expected = np.eye(4)-1j*.01*H-.01/2*sum(L.conj().T@L for L in jumps)
        np.testing.assert_allclose(mats[0], expected, atol=1e-14)
        _, result = self.compile_direct(rep)
        self.assertGreater(result['tau'], 1.)  # finite-dt Euler input is not assumed TP

    def test_complex_nonidentity_zero_address(self):
        rep = pack([[.7+.1j, .2-.3j]], ['X', 'Y'])
        bundle, _ = self.compile_direct(rep)
        self.assertIn('0', [a for _,a in bundle['plans']['0']['placement']])

    def test_padding_and_single_identity(self):
        rep = pack([[.2+.1j,.3-.2j,.1j,.4,.2j]], ['IX','IY','IZ','XI','YI'])
        bundle, _ = self.compile_direct(rep)
        self.assertGreater(bundle['plans']['0']['debug']['padding_count'], 0)
        self.compile_direct(pack([[1j]], ['I']))

    def test_legacy_default_unchanged(self):
        from search_contract import channel_from_rep, matrixsum
        from block_encoding_new_fixed import BlockEncoding
        rep = build_input(n=2)
        ir = channel_from_rep(rep)
        a = ir.rewrite_search(strategy='beam', beam_width=2, max_steps=2)
        events = []
        b = ir.rewrite_search(strategy='beam', beam_width=2, max_steps=2,
                              observer=lambda reason, **kwargs: events.append(reason))
        np.testing.assert_array_equal(a['A_final'], b['A_final'])
        self.assertTrue(events)
        spec = importlib.util.spec_from_file_location('original_fixed', ROOT/'originals/deployed/block_encoding_new_fixed.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        A, labels = unpack(rep)
        op = matrixsum(A[0], labels)
        old = original.BlockEncoding(op)
        new = BlockEncoding(op)
        self.assertEqual(old.find_optimal_order_matrices(), new.find_optimal_order_matrices())
        self.assertEqual(old.additional_modes, new.additional_modes)

    def test_interfaces_cache_and_frontier(self):
        s = self.session(n=1)
        rid = s.call('observe')['root_repr_id']
        args = {'repr_id':rid, 'config':FAST, 'seed_index':0}
        self.assertFalse(s.call('lookup_cache', {'operation':'search_placements','args':args})['hit'])
        placement = s.call('search_placements', args)
        self.assertEqual(placement['status'], 'completed', placement)
        hit = s.call('lookup_cache', {'operation':'search_placements','args':args})
        self.assertTrue(hit['hit'])
        self.assertEqual(hit['source_event'], placement['event_id'])
        self.assertFalse(s.call('lookup_cache', {'operation':'search_placements','args':dict(args,seed_index=1)})['hit'])
        evargs = {'repr_id':rid, 'placement_bundle_id':placement['placement_bundle_id']}
        result = s.call('evaluate', evargs)
        self.assertEqual(result['status'], 'completed', result)
        self.assertTrue(s.call('evaluate', evargs)['cache_hit'])
        self.assertEqual(s.call('inspect_ledger')['budget']['used']['lowering_calls'],1)
        before = s.call('observe')['best_evaluation_id']
        s.call('update_frontier', {'archive':[rid]})
        s = Session(s.path)
        self.assertEqual(s.call('observe')['best_evaluation_id'],before)
        self.assertEqual(s.call('observe')['candidates'],[])
        s.call('update_frontier', {'activate':[rid]})
        self.assertEqual(s.call('finish')['best_evaluation_id'],before)

    def test_export_equivalence_cursor_and_incomplete_bundle(self):
        s = self.session(n=2)
        rid = s.call('observe')['root_repr_id']
        args = {'parent_repr_id':rid,'limits':{'max_pairs':1,'max_unitaries':8,'page_size':2}}
        proposal = s.call('propose_rewrites', args)
        self.assertEqual(proposal['status'],'completed',proposal)
        root = s.call('observe',{'repr_id':rid,'include_structure':True})['candidates'][0]['representation']
        self.assertTrue(proposal['candidates'])
        for row in proposal['candidates']:
            rep = s.call('observe',{'repr_id':row['repr_id'],'include_structure':True})['candidates'][0]['representation']
            self.assertLess(equivalent(rep,root),1e-10)
        if proposal['next_cursor']:
            next_page = s.call('propose_rewrites',dict(args,cursor=proposal['next_cursor']))
            self.assertEqual(next_page['status'],'completed',next_page)
        partial = s.call('search_placements',{'repr_id':rid,'kraus_ids':[0],'config':FAST})
        self.assertTrue(partial['missing_kraus_ids'])
        with self.assertRaises(ValueError):
            s.call('evaluate',{'repr_id':rid,'placement_bundle_id':partial['placement_bundle_id']})
        completed = s.call('search_placements',{'repr_id':rid,'kraus_ids':partial['missing_kraus_ids'],
                    'base_bundle_id':partial['placement_bundle_id'],'config':FAST})
        self.assertEqual(completed['missing_kraus_ids'],[])

    def test_timeout_charged_and_no_best(self):
        s = self.session(n=1)
        rid = s.call('observe')['root_repr_id']
        b = s.call('search_placements',{'repr_id':rid,'config':FAST})
        result = s.call('evaluate',{'repr_id':rid,'placement_bundle_id':b['placement_bundle_id'],
                                  'limits':{'wall_seconds':.001}})
        self.assertEqual(result['status'],'wall_timeout')
        ledger = s.call('inspect_ledger',{'detail':True})
        self.assertEqual(ledger['budget']['used']['lowering_calls'],1)
        self.assertGreater(result['spent']['wall_seconds'],0)
        self.assertIsNone(s.call('observe')['best_evaluation_id'])
        self.assertFalse(s.call('lookup_cache',{'operation':'evaluate','args':{'repr_id':rid,
                                'placement_bundle_id':b['placement_bundle_id']}})['hit'])

    def test_diagnostic_trace_does_not_change_legacy_winner(self):
        s = self.session(n=2)
        rid = s.call('observe')['root_repr_id']
        a = s.call('run_legacy_rewrite', {'repr_id':rid, 'beam_width':2, 'max_steps':1})
        b = s.call('run_legacy_rewrite', {'repr_id':rid, 'beam_width':2, 'max_steps':1, 'trace':True})
        self.assertEqual(a['status'], 'completed', a)
        self.assertEqual(b['status'], 'completed', b)
        self.assertEqual(a['repr_id'], b['repr_id'])
        self.assertIsNone(a['trace_ref'])
        self.assertTrue(Path(b['trace_ref']).read_text())

    def test_content_identity_preserves_order_and_phase(self):
        from search_contract import stable_hash
        A = np.array([[.7,.2j],[.1,.4]])
        a, b = pack(A,['X','Y']), pack(A[::-1],['X','Y'])
        self.assertAlmostEqual(equivalent(a,b),0)
        self.assertNotEqual(stable_hash(a),stable_hash(b))
        c = pack(A * np.array([[1j],[1]]),['X','Y'])
        self.assertAlmostEqual(equivalent(a,c),0)
        self.assertNotEqual(stable_hash(a),stable_hash(c))
        self.assertTrue(stats(a)['score_includes_lambda'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
