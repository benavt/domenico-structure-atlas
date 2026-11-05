from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rbp_public_states import CORE, attach, digest, encode, logical_hash
from rbp_panel_exports import record

try:
    import numpy as np
    from src.analysis import tm_state_geometry
    from src.analysis.tm_state_geometry import assign_membership
    NATIVE = True
except ImportError:
    NATIVE = False


@unittest.skipUnless(NATIVE, 'Saved-array libraries optional in source CI')
class PublicStatesTests(unittest.TestCase):
    def fixture(self, root):
        source, saved = root/'source', root/'saved'; source.mkdir(); saved.mkdir()
        names = [f'm{i}' for i in range(60)]; sequence_sha = digest(('A'*271).encode())
        bindings = dict(plan_sha256='a'*64, consumer_commit='b'*40, library_commit='c'*40)
        methods = ['methodA']*37+['methodB']*23
        def put(path, value): path.write_bytes(encode(value)); return record(path)
        mapping = [dict(canonical_position=i, chain='Z', native_residue=[' ',
            0 if i == 1 else 1 if i in (2,3) else i-2, 'A' if i == 2 else 'B' if i == 3 else ' ']) for i in CORE]
        def pdb(models):
            lines = []
            for frame, model in enumerate(models, 1):
                lines.extend([f'MODEL     {frame:4d}', 'REMARK 900 SOURCE_MODEL '+model])
                for ordinal, row in enumerate(mapping):
                    _, number, code = row['native_residue']
                    for offset, atom in enumerate(['N', 'CA'], 1):
                        serial = ordinal*2+offset
                        lines.append(f'ATOM  {serial:5d} {atom:^4s} ALA Z{number:4d}{code}   {ordinal:8.3f}{0:8.3f}{1:8.3f}  1.00 50.00           C  ')
                lines.append('ENDMDL')
            return ('\n'.join([*lines, 'END', ''])).encode()
        production, disco = list(range(5,55)), list(range(50))
        state = dict(id='S1', indices=list(range(55)), count=55, anchor=names[0], representative=names[0],
                     supported=True, eligible_for_sampling=True, methods={})
        for method, selected in [('mcmc50', production), ('disco50', disco)]:
            path = saved/(method+'.pdb'); path.write_bytes(pdb([names[i] for i in selected]))
            session = saved/(method+'.pse'); session.write_bytes(pickle.dumps(dict(names=['saved_state_panel'], all_states=1), protocol=1))
            alignment = dict(anchor=names[0], canonical_positions=CORE,
                members=[dict(model=names[i], atom_count=542, canonical_mapping=mapping,
                              alignment_canonical_positions=CORE) for i in selected],
                verification=dict(status='passed', states=50, native_atom_inventory_preserved=True,
                    aligned_coordinates_preserved=True, all_states_visible=True))
            proof = put(saved/(method+'-alignment.json'), alignment)
            state['methods'][method] = dict(indices=selected, models=[names[i] for i in selected], pdb=record(path),
                pse=record(session), alignment=proof, full_atom_frames=50, alignment_canonical_positions=CORE)
        sampling = saved/'sampling.npz'
        np.savez(sampling, cohort_indices=np.arange(55), production_indices=production, disco_indices=disco,
                 chain0_final=np.arange(5,55), log_weights=np.zeros(55))
        summary = dict(status='complete', alpha=.5, k=50, chains=4, steps=10000, burn=1000,
            uniform_mixture=.1, production_chain=0, production_iteration=10000, conditioning_t=.1,
            energy_scope='full_construct', scoring_residues=271, canonical_positions=CORE,
            target='RBP', range_id='all', state_members=55, target_law='product_weight_fixed_cardinality_subset',
            original_target_iqr=17., tau=17.,
            prior_scope='Original HDBSCAN labels and sizes across the complete 60-model cohort; conditioned on state membership.',
            disco=dict(law='fine_cluster_min_energy_seeds_then_plackett_luce_control'), arrays=record(sampling),
            diagnostics=dict(panel_mean_energy=dict(rhat=1.02, ess_bulk=45, ess_tail=30)))
        state['summary'] = put(saved/'summary.json', summary)
        cat = dict(schema_version=1, kind='saved-clustering-residual-states-and-subsets', target='RBP', range_id='all',
            models=names, methods=methods, construct_residues=271, canonical_positions=CORE, clustering_positions=CORE,
            sequence_sha256=sequence_sha, TM_cutoff=.8, min_state_size=50, source_bindings=bindings,
            coordinate_export_status='complete', states=[state], unassigned_indices=list(range(55,60)))
        cat_entry = put(saved/'catalog.json', cat)
        scores = saved/'scores.npz'; values = np.asarray([1.]+[.9]*54+[.7]*5)[:,None]
        np.savez(scores, anchors=np.asarray([names[0]], dtype='U'), model_to_anchor=values)
        raw_views = {}
        for view in ['primary','density']:
            raw = encode(dict(target='RBP_P02925', view_id=view, range_id='all', models=names, methods=methods,
                residues=271, clusters=['P1' if view == 'primary' else 'H1']*60,
                projections={method:dict(xy=[[float(i),float(i+1)] for i in range(60)]) for method in ['pca','umap','tsne']},
                provenance=dict(sequence_sha256=sequence_sha)))
            path = source/(view+'.json.gz'); path.write_bytes(gzip.compress(raw,mtime=0))
            raw_views[view] = dict(file=view+'.json', sha256=digest(raw))
        return dict(root=source, saved=saved, names=names, cat=cat, catalog=cat_entry, scores=record(scores),
            views=raw_views, bindings=bindings, sequence=sequence_sha, put=put)

    def attach(self, f):
        return attach(f['root'], f['catalog'], f['scores'], f['views'], expected_source_bindings=f['bindings'],
            expected_model_order=f['names'], sequence_sha256=f['sequence'], plan_sha256='a'*64, source_commit='b'*40,
            method_labels=f.get('method_labels'))

    def test_full_schema_two_binding_keeps_all_points_and_shared_exact_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp)); before = {p:p.read_bytes() for p in f['root'].iterdir()}
            output = self.attach(f)
            self.assertEqual(len(output['files']), 6)
            payload = json.loads((f['root']/'tm-states/RBP_P02925/full271.json').read_bytes())
            self.assertEqual(payload['schema_version'], 2)
            self.assertEqual(payload['models'], f['names'])
            self.assertEqual(payload['model_order_sha256'], logical_hash(f['names']))
            self.assertEqual(payload['coverage'], dict(assigned_models=55, unassigned_models=5))
            self.assertEqual(payload['states'][0]['members'][0]['tm_score'], 1.)
            selections = payload['states'][0]['selections']['all']
            self.assertEqual(selections['primary'], selections['density'])
            self.assertEqual(selections['primary']['mcmc50']['indices'], list(range(5,55)))
            self.assertEqual(selections['primary']['disco50']['indices'], list(range(50)))
            for row in output['files']:
                self.assertEqual(record(f['root']/row['path'])['sha256'], row['sha256'])
            for path, raw in before.items(): self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(self.attach(f), output)  # identical attachment is idempotent
            self.assertNotIn(str(f['saved']), (f['root']/'tm-states.json').read_text())

    def test_tm_certificate_and_selected_panel_model_order_are_verified(self):
        for change in ['membership', 'panel', 'prior', 'core']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp))
                if change == 'membership':
                    with np.load(f['scores']['path']) as data: values = data['model_to_anchor'].copy()
                    values[1,0] = .7
                    np.savez(f['scores']['path'], anchors=[f['names'][0]], model_to_anchor=values)
                    f['scores'] = record(f['scores']['path'])
                elif change == 'panel': f['cat']['states'][0]['methods']['mcmc50']['models'].reverse()
                elif change == 'core': f['cat']['canonical_positions'] = CORE[:-1]
                else:
                    entry = f['cat']['states'][0]['summary']; summary = json.loads(Path(entry['path']).read_bytes())
                    summary['prior_scope'] = '11000-model cohort'
                    f['cat']['states'][0]['summary'] = f['put'](Path(entry['path']), summary)
                f['catalog'] = f['put'](Path(f['catalog']['path']), f['cat'])
                with self.assertRaises(ValueError): self.attach(f)
                self.assertFalse((f['root']/'tm-states.json').exists())

    def test_changed_saved_hash_or_private_session_blocks_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp)); Path(f['catalog']['path']).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'hash/size'): self.attach(f)
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp)); panel = f['cat']['states'][0]['methods']['mcmc50']
            path = Path(panel['pse']['path']); path.write_bytes(pickle.dumps(dict(session_file='/home/private/run.pse'), protocol=1))
            panel['pse'] = record(path); f['catalog'] = f['put'](Path(f['catalog']['path']), f['cat'])
            with self.assertRaisesRegex(ValueError, 'Private'): self.attach(f)
            self.assertFalse((f['root']/'tm-states.json').exists())

    def test_empty_bounded_search_keeps_all_points_unassigned_without_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp)); f['cat']['states'] = []; f['cat']['unassigned_indices'] = list(range(60))
            f['catalog'] = f['put'](Path(f['catalog']['path']), f['cat'])
            np.savez(f['scores']['path'], anchors=np.asarray([],dtype='U'), model_to_anchor=np.empty((60,0)))
            f['scores'] = record(f['scores']['path'])
            output = self.attach(f)
            self.assertEqual(len(output['files']), 2)
            payload = json.loads((f['root']/'tm-states/RBP_P02925/full271.json').read_bytes())
            self.assertEqual(payload['states'], [])
            self.assertEqual(payload['coverage'], dict(assigned_models=0,unassigned_models=60))
            self.assertFalse(output['target_entry']['ranges']['all']['views']['density']['available'])

    def test_explicit_native_key_to_human_label_mapping_preserves_scientific_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp)); native = ['afcluster']*37+['bioemu']*23
            labels = {'afcluster':'AF-Cluster', 'bioemu':'BioEMU', 'boltz2':'Boltz2'}
            f['cat']['methods'] = native
            f['catalog'] = f['put'](Path(f['catalog']['path']), f['cat'])
            for view in ['primary','density']:
                path = f['root']/(view+'.json.gz'); data = json.loads(gzip.decompress(path.read_bytes()))
                data['methods'] = [labels[key] for key in native]; raw = encode(data)
                path.write_bytes(gzip.compress(raw,mtime=0)); f['views'][view]['sha256'] = digest(raw)
            with self.assertRaisesRegex(ValueError, 'Scientific view'): self.attach(f)
            f['method_labels'] = labels
            before = {path:path.read_bytes() for path in [Path(f['catalog']['path']), *(f['root']/(v+'.json.gz') for v in ['primary','density'])]}
            self.attach(f)
            payload = json.loads((f['root']/'tm-states/RBP_P02925/full271.json').read_bytes())
            proof = payload['provenance']['method_label_aliases']
            self.assertEqual(proof['key_to_display_name'], {'afcluster':'AF-Cluster','bioemu':'BioEMU'})
            self.assertEqual(proof['native_method_order_sha256'], logical_hash(native))
            self.assertNotEqual(proof['native_method_order_sha256'], proof['display_method_order_sha256'])
            self.assertTrue(proof['scientific_method_keys_preserved'])
            self.assertEqual(payload['states'][0]['selections']['all']['density']['mcmc50']['indices'], list(range(5,55)))
            for path, raw in before.items(): self.assertEqual(path.read_bytes(), raw)

    def test_explicit_method_mapping_rejects_missing_keys_and_ambiguous_labels(self):
        for labels in ({'methodA':'A'}, {'methodA':'Family','methodB':'Family'}):
            with self.subTest(labels=labels), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp)); f['method_labels'] = labels
                with self.assertRaisesRegex(ValueError, 'one-to-one'): self.attach(f)
                self.assertFalse((f['root']/'tm-states.json').exists())

    @unittest.skipUnless(shutil.which('node') or shutil.which('nodejs'), 'Node is optional for pinned JS compatibility check')
    def test_pinned_browser_validator_accepts_both_views_and_shared_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp)); self.attach(f)
            script = Path(tm_state_geometry.__file__).resolve().parents[2]/'src/app/templates/ais_explorer_tm_states.js'
            code = """
const fs=require('fs'),path=require('path'),zlib=require('zlib');
const root=process.argv[1],api=require(process.argv[2]);
const cat=JSON.parse(fs.readFileSync(path.join(root,'tm-states.json')));
const target=cat.targets.RBP_P02925,range=target.ranges.all;
for(const view of ['primary','density']){
 const entry=range.views[view],data=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(root,view+'.json.gz'))));
 data.visible_view_policy=true;data.policy_view_id='checked-policy';data._source_sha=entry.source_data_sha256;
 data.views=[{id:'checked-policy',source_view_id:view,source_data_sha256:entry.source_data_sha256,canonical_positions:Array.from({length:271},(_,i)=>i+1)}];
 const payload=JSON.parse(fs.readFileSync(path.join(root,entry.file)));
 api.validatePayload(payload,{...entry,effective_scope_id:range.effective_scope_id},target,data);
 payload.active_binding={range_id:'all',view_id:view};
 const selected=api.select(payload,{state_id:'S1',method:'mcmc50'});
 if(selected.orderedIndices.length!==50||selected.members!==55)throw Error('Wrong saved subset');
}
process.stdout.write('both schema2 views verified');
"""
            done = subprocess.run([shutil.which('node') or shutil.which('nodejs'), '-e', code, str(f['root']), str(script)],
                check=True, capture_output=True, text=True)
            self.assertEqual(done.stdout, 'both schema2 views verified')


if __name__ == '__main__': unittest.main()
