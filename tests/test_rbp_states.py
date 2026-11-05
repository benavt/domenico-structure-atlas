import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rbp_states import core_feature_indices, native_call_bounds


class ContractTests(unittest.TestCase):
    def test_core_is_sequence_ordinal_not_native_number(self):
        self.assertEqual(core_feature_indices(5, [0, 2, 4]), [1, 3, 8])
        for core in ([1, 0], [0, 0], [-1, 2], [0, 5], [0], [False, 1]):
            with self.subTest(core=core), self.assertRaises(ValueError):
                core_feature_indices(5, core)

    def test_native_ceiling_includes_rejected_columns_screening_and_audit(self):
        self.assertEqual(native_call_bounds(10926)['directional_call_ceiling'], 185616)
        with self.assertRaises(ValueError): native_call_bounds(49)


try:
    import numpy as np
    from src.analysis.tm_state_geometry import PROTOCOL_VERSION, TM_ARGUMENTS, assign_membership
    from src.analysis.ensemble_sampling import energy_distribution
    NATIVE = True
except ImportError:
    NATIVE = False


@unittest.skipUnless(NATIVE, 'Pinned scientific imports optional in source CI')
class MockedProtocolTests(unittest.TestCase):
    def cohort(self):
        n = 60
        cohort = dict(target='RBP', range_id='frozen-core', models=[f'm{i}' for i in range(n)],
            methods=['first' if i < 42 else 'second' for i in range(n)], energies=list(range(n)),
            sequence_sha256='a'*64, construct_residues=4, conditioning_t=.1, energy_scope='full_construct')
        clustering = dict(primary_labels=['p1']*n, hdbscan_labels=['fine']*55+['noise']*5,
            hdbscan_noise_labels=['noise'], clustering_positions=[1, 2, 3, 4],
            saved_source_views={name:dict(path=f'{name}.npz', sha256='b'*64) for name in ['pca', 'umap', 'tsne']})
        return cohort, clustering

    def scorer(self, names, score=.9, bad_mean=False):
        class Scorer:
            directional_calls = 0
            protocol = dict(version=PROTOCOL_VERSION, arguments=list(TM_ARGUMENTS),
                correspondence='same canonical residue index', normalization='average length',
                aggregation='mean of two directions', executable_sha256='b'*64)
            def score(self, left, right):
                self.directional_calls += 2
                return dict(protocol=self.protocol, fingerprint=dict(canonical_positions=[1, 3], sequence_sha256='a'*64),
                    forward=dict(source_model=left, target_model=right, score=score),
                    reverse=dict(source_model=right, target_model=left, score=score), score=.1 if bad_mean else score)
        return Scorer()

    def test_native_oracle_validates_bidirectional_mean_and_memoizes(self):
        from rbp_states import NativePairOracle
        cohort, _ = self.cohort(); names = cohort['models']
        with tempfile.TemporaryDirectory() as tmp:
            scorer = self.scorer(names)
            oracle = NativePairOracle(names, scorer, [0, 2], 'a'*64, Path(tmp)/'ok', lambda: None)
            self.assertEqual(oracle([(1, 0), (0, 1), (0, 0)]), [.9, .9, 1.])
            self.assertEqual(scorer.directional_calls, 2)
            self.assertEqual(oracle([(0, 1)]), [.9])
            self.assertEqual(scorer.directional_calls, 2)
            self.assertEqual(len(list((Path(tmp)/'ok').glob('checkpoint-*.json'))), 2)
            bad = NativePairOracle(names, self.scorer(names, bad_mean=True), [0, 2], 'a'*64, Path(tmp)/'bad', lambda: None)
            with self.assertRaisesRegex(ValueError, 'bidirectional mean'): bad([(0, 1)])
            oracle.limit = 2
            with self.assertRaisesRegex(ValueError, 'before scoring'): oracle([(2, 3)])
            self.assertEqual(scorer.directional_calls, 2)

    def test_native_oracle_routes_new_pairs_through_bounded_bulk_owner(self):
        from rbp_states import NativePairOracle
        cohort,_=self.cohort();names=cohort['models'];scorer=self.scorer(names);single=scorer.score;seen=[]
        def many(pairs,*,workers):
            seen.append((pairs,workers));return [single(*pair) for pair in pairs]
        scorer.score_many=many
        def forbidden(*args):raise AssertionError('Bulk native owner must control wave commits')
        scorer.score=forbidden
        with tempfile.TemporaryDirectory() as tmp:
            oracle=NativePairOracle(names,scorer,[0,2],'a'*64,Path(tmp)/'waves',lambda:None,workers=4)
            self.assertEqual(oracle([(1,0),(0,1),(0,0),(2,0)]),[.9,.9,1.,.9])
            self.assertEqual(seen,[([('m0','m1'),('m0','m2')],4)])
            self.assertEqual(scorer.directional_calls,4)

    def test_native_oracle_releases_rich_proofs_between_ordered_25_pair_chunks(self):
        import weakref
        from rbp_states import NativePairOracle
        cohort,_=self.cohort();names=cohort['models'];scorer=self.scorer(names);single=scorer.score
        requests=[(i,0) for i in reversed(range(1,len(names)))]+[(0,1),(1,0),(0,0)]
        seen,previous=[],[]
        class Proof(dict):pass
        def many(pairs,*,workers):
            self.assertTrue(all(ref() is None for ref in previous), 'Prior rich proofs survived into another wave')
            seen.append((list(pairs),workers));results=[Proof(single(*pair)) for pair in pairs]
            previous[:]=[weakref.ref(result) for result in results]
            return results
        scorer.score_many=many
        with tempfile.TemporaryDirectory() as tmp:
            oracle=NativePairOracle(names,scorer,[0,2],'a'*64,Path(tmp)/'grouped',lambda:None,workers=4)
            baseline_scorer=self.scorer(names)
            baseline=NativePairOracle(names,baseline_scorer,[0,2],'a'*64,Path(tmp)/'baseline',lambda:None,workers=4)
            self.assertEqual(oracle(requests),baseline(requests))
            self.assertEqual([len(pairs) for pairs,_ in seen],[25,25,9])
            self.assertEqual([pair for pairs,_ in seen for pair in pairs],[('m0',f'm{i}') for i in range(1,len(names))])
            self.assertTrue(all(workers==4 for _,workers in seen))
            self.assertTrue(all(ref() is None for ref in previous))
            self.assertEqual(scorer.directional_calls,baseline_scorer.directional_calls)
            self.assertEqual(scorer.directional_calls,2*(len(names)-1))

    def adapters(self, observed, invalid_panel=False):
        def search(names, features, groups, previous, cached, oracle, settings):
            self.assertEqual((previous, cached), ([], {}))
            self.assertEqual(settings['feature_count'], 1)
            scores = np.asarray(oracle([(0, i) for i in range(len(names))]))[:, None]
            assignment = assign_membership(names, [names[0]], scores, cutoff=.8, min_state_size=50)
            return dict(anchors=[0], scores=scores, assignment=assignment, cached_anchor_indices=[],
                baseline=dict(anchors=[]), budgets=dict(new_full_columns=1, noncached_trials=1,
                    screen_pair_requests=60, completion_pair_requests=0), stop_reason='mocked')
        def chains(logw, **kwargs):
            observed['logw'] = logw.copy(); observed['chains_kwargs'] = kwargs
            runs = []
            for number in range(4):
                # Deliberately different later chains prove that production cannot
                # cherry-pick the lowest-energy or apparently best-mixing chain.
                final = np.arange(50) if number or len(logw) == 50 else np.arange(10, 60)
                runs.append(dict(final=final, seed=kwargs['seed']+1000+number,
                    proposals=0 if len(logw) == 50 else 10000, k=50,
                    degenerate_single_subset=len(logw) == 50, group_keys=['fine', 'noise'], method_keys=['first', 'second']))
            if invalid_panel: runs[0]['final'] = np.zeros(50, dtype=int)
            return dict(law='product_weight_fixed_cardinality_subset', production_chain=0,
                production_iteration=10000, chains=runs, transition_seeds=[1000+i for i in range(4)],
                diagnostics=dict(status='mocked_not_measured'))
        def disco(logw, energies, labels, **kwargs):
            observed['disco_logw'] = logw.copy()
            return dict(indices=np.arange(50), law='fine_cluster_min_energy_seeds_then_plackett_luce_control',
                seed=kwargs['seed'], k=50)
        return dict(run_search=search, run_subset_chains=chains, disco_subset=disco,
                    audit_pairs=lambda assignment, names: [])

    def test_saved_geometry_global_prior_and_fixed_production_panel(self):
        import json
        from rbp_states import run_states
        cohort, clustering = self.cohort(); observed = {}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'states'
            catalog = run_states(cohort, clustering, np.zeros((60, 1), dtype=np.float32), [0, 2], output,
                self.scorer(cohort['models']), source_bindings=dict(plan_sha256='c'*64), seed=7,
                adapters=self.adapters(observed))
            self.assertEqual(catalog['canonical_positions'], [1, 3])
            self.assertEqual(catalog['summary']['new_clustering_fits'], 0)
            self.assertEqual(catalog['states'][0]['methods']['mcmc50']['indices'], list(range(10, 60)))
            self.assertEqual(catalog['states'][0]['methods']['disco50']['indices'], list(range(50)))
            tau = np.quantile(cohort['energies'], .75)-np.quantile(cohort['energies'], .25)
            expected = np.asarray(energy_distribution(cohort['energies'], clustering['hdbscan_labels'], .5,
                tau, ['noise'])['log_probabilities']); expected -= expected.max()
            np.testing.assert_array_equal(observed['logw'], expected)
            np.testing.assert_array_equal(observed['disco_logw'], expected)
            self.assertEqual(observed['chains_kwargs']['steps'], 10000)
            self.assertEqual(observed['chains_kwargs']['burn'], 1000)
            summary = json.loads(Path(catalog['states'][0]['summary']['path']).read_text())
            self.assertIn('60-model', summary['prior_scope'])
            self.assertNotIn('11000', json.dumps(catalog))
            self.assertEqual(len(list((output/'geometry-checkpoints').glob('column-*.npz'))), 1)

    def test_phase_checkpoint_sees_completed_geometry_or_sampling_writes_before_stopping(self):
        import json
        from rbp_states import run_states
        for stopped_phase in ['tm-state-geometry', 'tm-state-sampling']:
            with self.subTest(stopped_phase=stopped_phase), tempfile.TemporaryDirectory() as tmp:
                cohort, clustering = self.cohort()
                observed, checkpoints = {}, []
                output = Path(tmp)/'states'
                def checkpoint(checkpoint_id):
                    checkpoints.append(checkpoint_id)
                    for name in ['pair-audit.json', 'search.json', 'membership.json', 'pair-query-ledger.json']:
                        self.assertIsInstance(json.loads((output/name).read_text()), dict)
                    with np.load(output/'scores.npz', allow_pickle=False) as saved:
                        self.assertEqual(saved['model_to_anchor'].shape, (60, 1))
                    if checkpoint_id.startswith('tm-state-sampling:'):
                        summary = json.loads((output/'sampling/S1/summary.json').read_text())
                        self.assertEqual(summary['status'], 'complete')
                        self.assertEqual(summary['production_chain'], 0)
                        with np.load(summary['arrays']['path'], allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved['production_indices'], np.arange(10, 60))
                    if checkpoint_id.startswith(stopped_phase+':'):
                        raise RuntimeError('mock stop at saved phase boundary')
                with self.assertRaisesRegex(RuntimeError, 'saved phase boundary'):
                    run_states(cohort, clustering, np.zeros((60, 1), dtype=np.float32), [0, 2], output,
                        self.scorer(cohort['models']), source_bindings=dict(plan_sha256='c'*64), seed=7,
                        adapters=self.adapters(observed), checkpoint=checkpoint)
                self.assertTrue(checkpoints[0].startswith('tm-state-geometry:'))
                self.assertFalse((output/'catalog.json').exists())
                if stopped_phase == 'tm-state-geometry':
                    self.assertEqual(len(checkpoints), 1)
                    self.assertNotIn('chains_kwargs', observed)
                else:
                    self.assertEqual(checkpoints[1], 'tm-state-sampling:S1')
                    self.assertEqual(observed['chains_kwargs']['seed'], 7)

    def test_prior_keeps_full_cohort_cluster_sizes_before_state_conditioning(self):
        from rbp_states import run_states
        cohort, clustering = self.cohort(); observed = {}
        clustering['hdbscan_labels'] = ['fine']*40+['other']*15+['noise']*5
        scorer = self.scorer(cohort['models']); original_score = scorer.score
        def score(left, right):
            result = original_score(left, right)
            if int(left[1:]) >= 50 or int(right[1:]) >= 50:
                result['forward']['score'] = result['reverse']['score'] = result['score'] = .7
            return result
        scorer.score = score
        with tempfile.TemporaryDirectory() as tmp:
            catalog = run_states(cohort, clustering, np.zeros((60, 1), dtype=np.float32), [0, 2], Path(tmp)/'states',
                scorer, source_bindings=dict(plan_sha256='c'*64), seed=7, adapters=self.adapters(observed))
            self.assertEqual(catalog['unassigned_indices'], list(range(50, 60)))
            tau = np.quantile(cohort['energies'], .75)-np.quantile(cohort['energies'], .25)
            full = np.asarray(energy_distribution(cohort['energies'], clustering['hdbscan_labels'], .5,
                tau, ['noise'])['log_probabilities'])[:50]; full -= full.max()
            local = np.asarray(energy_distribution(cohort['energies'][:50], clustering['hdbscan_labels'][:50], .5,
                tau, ['noise'])['log_probabilities']); local -= local.max()
            np.testing.assert_array_equal(observed['logw'], full)
            self.assertFalse(np.allclose(full, local))

    def test_invalid_panel_and_mismatched_feature_core_fail(self):
        from rbp_states import run_states
        cohort, clustering = self.cohort()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, '50 distinct'):
                run_states(cohort, clustering, np.zeros((60, 1), dtype=np.float32), [0, 2], Path(tmp)/'bad-panel',
                    self.scorer(cohort['models']), source_bindings=dict(plan_sha256='c'*64), seed=7,
                    adapters=self.adapters({}, invalid_panel=True))
            with self.assertRaisesRegex(ValueError, 'features differ'):
                run_states(cohort, clustering, np.zeros((60, 2), dtype=np.float32), [0, 2], Path(tmp)/'bad-core',
                    self.scorer(cohort['models']), source_bindings=dict(plan_sha256='c'*64), seed=7,
                    adapters=self.adapters({}))


if __name__ == '__main__': unittest.main()
