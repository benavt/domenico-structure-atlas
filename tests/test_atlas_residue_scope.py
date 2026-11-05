"""Saved metadata and mock renderer only; no coordinates, fits or exports."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_atlas as module
from snapshot import sha256


class ResiduePresentationTests(unittest.TestCase):
    def test_saved_target_masks_keep_construct_and_retained_counts_distinct(self):
        for count, start, end, full in [(271, 1, 271, True), (523, 4, 438, False), (993, 58, 951, False)]:
            retained = list(range(start-1, end))
            detection = dict(retained_reference_positions=retained,
                excluded_indices=sorted(set(range(count))-set(retained)))
            with self.subTest(count=count):
                value = module.residue_presentation(detection, count)
                self.assertEqual(value['positions'], list(range(start, end+1)))
                self.assertEqual(value['full'], full)
                self.assertEqual(value['reference_ranges'], f'construct positions {start}–{end}')
                if full:
                    self.assertEqual(value['range_name'], 'All residues')
                    self.assertEqual(value['mask_note'], 'All 271 supplied construct residues retained; native numbering preserved. Biological disorder is not inferred.')
                else:
                    self.assertIn(f'{len(retained)} of {count}', value['mask_note'])
                    self.assertIn(f'{start}–{end}', value['range_name'])
                    self.assertNotIn('every supplied', value['ensemble_interpretation'])

    def test_discontinuous_saved_masks_do_not_become_contiguous_ranges(self):
        value = module.residue_presentation(dict(retained_reference_positions=[0, 1, 3, 5, 6]), 8)
        self.assertEqual(value['ranges'], '1–2, 4, 6–7')
        self.assertEqual(value['positions'], [1, 2, 4, 6, 7])

    def test_ambiguous_or_contradictory_masks_are_refused(self):
        for retained in [[], [0, 1], [0, 1, 1], [2, 1, 0], [False, 1, 2], [-1, 0, 1], [0, 1, 4]]:
            with self.subTest(retained=retained), self.assertRaises(ValueError):
                module.residue_presentation(dict(retained_reference_positions=retained), 4)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            module.residue_presentation(dict(retained_reference_positions=[0, 1, 2], excluded_indices=[]), 4)

    def test_public_payloads_use_saved_masks_without_opening_models_or_exporting(self):
        renderer = types.ModuleType('src.analysis.atlas_presentation')
        renderer.renderer_identity = lambda: dict(revision='b'*40, uncommitted_source=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def save(name, value):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value)); return path
            def entry(path):
                return dict(path=str(path), sha256=sha256(path))
            targets, workflows, expected, names_by_target, rows, scores = {}, {}, {}, {}, [], []
            for target, count, first, last in [('RBP_P02925', 271, 1, 271), ('5NTase_P07024', 523, 4, 438), ('SERCA_P04191', 993, 58, 951)]:
                names = [target+'__native__'+str(i) for i in range(3)]; names_by_target[target] = names
                retained = list(range(first-1, last)); expected[target] = (count, first, last)
                target_folder = root/target; target_folder.mkdir()
                for view in ('primary', 'density'):
                    with (target_folder/(view+'.csv')).open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=['model', 'source_path', 'cluster_label', 'color']); writer.writeheader()
                        for i, name in enumerate(names):
                            writer.writerow(dict(model=name, source_path='/models-must-not-be-read/'+name+'.pdb',
                                cluster_label='root.unassigned' if view=='density' and i==2 else 'C1',
                                color='#BBBBBB' if view=='density' and i==2 else '#4477AA'))
                report = save(target+'/report.json', dict(clusters=[]))
                manifest = save(target+'/manifest.json', dict(identifier=target, analysis_status='complete',
                    knn_clustering=dict(labels_csv=str(target_folder/'primary.csv')), cluster_ensembles=dict(report=str(report)),
                    terminal_detection=dict(retained_reference_positions=retained, excluded_indices=sorted(set(range(count))-set(retained)), pair_mask_sha256='a'*64),
                    visualize_distogram=dict(images=[])))
                workflows[target] = dict(manifest=str(manifest), manifest_sha256=sha256(manifest), density_artifacts={'saved_fixture': True})
                targets[target] = dict(title=target, models=3, residues=count, sequence_sha256='a'*64)
                for i, name in enumerate(names):
                    rows.append(dict(model_id=name, target=target, method='native', sha256='c'*64,
                        native_physicality=['retained', 'rejected', 'unavailable'][i],
                        energy_t010=float(i), **{f'energy_t{k:02d}':float(i+k) for k in range(16)}))
                    scores.append(dict(model_id=name, coordinate_sha256='c'*64, mean_plddt=80.+i, ptm=.7))
            native = save('native.json', dict(status='validated', models=rows))
            confidence = save('confidence.json', dict(status='validated', models=scores, source_manifest_sha256=sha256(native)))
            study = save('config/study.json', dict(targets=targets, library_commit='d'*40, snapshot=dict(manifest_sha256=sha256(native))))
            save('config/model-types.json', dict(rules=[dict(key='native', name='Native model', color='#4477AA')]))
            context = save('context.json', dict(retrieved_at_utc='2026-11-05T00:00:00Z', targets={
                target:dict(protein='Saved metadata fixture', organism='Fixture', modeled_canonical_range=[1, count], canonical_length=count,
                    source='https://example.org/reference') for target, (count, _, _) in expected.items()}))
            vendors = {}
            for filename in ('plotly.min.js', '3Dmol-min.js'):
                path = root/filename; path.write_text('// Static fixture'); vendors[filename] = entry(path)
            plan = save('plan.json', dict(targets=workflows, study=entry(study), native=entry(native), confidence=entry(confidence),
                construct_context=entry(context), vendor_assets=vendors, renderer_commit='b'*40,
                source=str(root/'prepared'), output=str(root/'not-exported')))
            def projections(manifest, expected_rows):
                names = list(expected_rows)
                return names, {method:dict(xy=[[0., 0.], [1., 0.], [0., 1.]]) for method in ('pca', 'umap', 'tsne')}
            def views(frozen, manifest, target, names, **kwargs):
                return [dict(id=view, name=view, data=view+'.json', labels_csv=str(root/target/(view+'.csv')),
                    ensemble_report=manifest['cluster_ensembles']['report'], method_note='Saved fixture only') for view in ('primary', 'density')]
            with patch.dict(sys.modules, {'src.analysis.atlas_presentation': renderer}), \
                 patch.object(module, 'projections', side_effect=projections), patch.object(module, 'saved_views', side_effect=views), \
                 patch.object(module, 'copy_representatives', return_value=[]), patch.object(module, 'saved_region_scope', return_value=({}, None)), \
                 patch.object(module, 'add_supplementary'), patch.object(module, 'export_prepared') as export, \
                 patch('native_ca.write_ca', side_effect=AssertionError('No coordinate exports in metadata tests')), patch('sys.stdout', new_callable=io.StringIO):
                prepared = module.build(plan, export_site=False)
            export.assert_not_called(); self.assertFalse((root/'not-exported').exists())
            source = Path(prepared['source'])
            scopes = json.loads((source/'clustering-residues.json').read_text())['targets']
            ranges = json.loads((source/'residue-ranges.json').read_text())['targets']
            contexts = json.loads((source/'target-context.json').read_text())['targets']
            for target, (count, first, last) in expected.items():
                positions = list(range(first, last+1)); full = len(positions)==count
                scope = scopes[target]['views']['density']
                self.assertEqual(scope['canonical_positions'], positions)
                self.assertEqual(scope['expected_ca_count'], count)
                self.assertEqual(scope['analysis_uses_full_construct'], full)
                self.assertEqual(scope['reference_ranges'], f'construct positions {first}–{last}')
                option = ranges[target]['residue_ranges'][0]
                self.assertEqual(option['id'], 'all'); self.assertEqual(option['default_view'], 'density')
                self.assertEqual(option['canonical_positions'], positions)
                self.assertEqual(option['name']=='All residues', full)
                data = json.loads((source/target/'density.json').read_text())
                self.assertEqual(data['canonical_positions'], positions)
                self.assertEqual(data['clustering_positions'], positions)
                self.assertEqual(data['residues'], count)
                self.assertEqual(data['construct_residues'], count)
                self.assertEqual(data['clustering_residue_count'], len(positions))
                self.assertEqual(data['range_id'], 'all')
                self.assertEqual(data['models'], names_by_target[target])
                self.assertEqual(data['clusters'][-1], 'root.unassigned')
                self.assertEqual(data['coordinate_delivery'], 'representatives')
                self.assertNotIn('model_files', data)
                self.assertEqual(data['mask_note'].startswith('All '), full)
                self.assertEqual('every supplied' in contexts[target]['project_writeup']['ensemble_interpretation'], full)
                self.assertEqual(data['schemes'][2]['name'], 'ProteinEBM energy · t=0.10')
                self.assertEqual(data['schemes'][-1]['values'], ['retained', 'rejected', 'unavailable'])
            publication = json.loads(Path(prepared['publication_manifest']).read_text())
            methods = publication['presentation']['methods']
            self.assertNotIn('Full-construct Cα distances', methods)
            self.assertIn('5NTase_P07024 435/523 construct residues (4–438)', methods)
            self.assertIn('SERCA_P04191 894/993 construct residues (58–951)', methods)
            # A single trimmed view still needs a truthful core selector;
            # the legacy RBP full-construct/default partition stays unchanged.
            for target in ('5NTase_P07024', 'RBP_P02925'):
                with self.subTest(single_view=target):
                    single = json.loads(plan.read_text())
                    single.update(targets={target:workflows[target]}, pending_targets=[t for t in workflows if t!=target],
                        source=str(root/(target+'-single')), output=str(root/(target+'-not-exported')))
                    def one_view(frozen, manifest, name, names, **kwargs):
                        return [dict(views(frozen, manifest, name, names)[0], id='whole')]
                    with patch.dict(sys.modules, {'src.analysis.atlas_presentation': renderer}), \
                         patch.object(module, 'projections', side_effect=projections), patch.object(module, 'saved_views', side_effect=one_view), \
                         patch.object(module, 'copy_representatives', return_value=[]), patch.object(module, 'saved_region_scope', return_value=({}, None)), \
                         patch.object(module, 'add_supplementary'), patch.object(module, 'export_prepared') as export, \
                         patch('native_ca.write_ca', side_effect=AssertionError('No coordinate exports in metadata tests')), patch('sys.stdout', new_callable=io.StringIO):
                        result = module.build(save(target+'-single-plan.json', single), export_site=False)
                    export.assert_not_called()
                    presentation = json.loads(Path(result['publication_manifest']).read_text())['presentation']
                    self.assertNotIn('clustering_order', presentation)
                    self.assertNotIn('HDBSCAN is the default', presentation['methods'])
                    options = Path(result['source'])/'residue-ranges.json'
                    if target=='RBP_P02925':
                        self.assertFalse(options.exists())
                        self.assertIn('Full-construct Cα distances.', presentation['methods'])
                    else:
                        option = json.loads(options.read_text())['targets'][target]['residue_ranges'][0]
                        self.assertEqual(option['default_view'], 'whole')
                        self.assertEqual(option['name'], 'Clustering core · 4–438 (435/523 residues)')
                        self.assertEqual(option['canonical_positions'], list(range(4, 439)))


if __name__=='__main__':
    unittest.main()
