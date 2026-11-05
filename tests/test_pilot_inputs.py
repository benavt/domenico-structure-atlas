import hashlib,json,struct,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_inputs as p

class InputTests(unittest.TestCase):
 def fixture(self,root,compressed=False):
  rows=[];native=[]
  for i in range(6):
   path=root/('m'+str(i)+'.pdb');path.write_bytes(('original'+str(i)).encode())
   rows.append(dict(model_id='m'+str(i),path=str(path),sha256=p.sha(path),method='a'if i<3 else'b'))
   native.append(dict(rows[-1],target='T',native_physicality='rejected'if i==0 else'unavailable',energy_t010=i))
  def put(name,data):
   path=root/name;path.write_text(json.dumps(data));return p.record(path)
  ids=[r['model_id']for r in rows];mask='a'*64
  spec=dict(target='T',source_models=6,model_count=4,selection_seed=42,retained_reference_positions=[0,1,2],pair_mask_sha256=mask,
   intake=put('intake.json',dict(models=rows)),native_models=put('native.json',dict(models=native)),
   feature_proof=put('proof.json',dict(models_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),retained_reference_positions=[0,1,2],pair_mask_sha256=mask)),
   terminal_report=put('terminal.json',dict(retained_reference_positions=[0,1,2],pair_mask_sha256=mask)))
  axis=repr(dict(descr='<i4',fortran_order=False,shape=(3,))).encode();axis+=b' '*((64-(10+len(axis)+1)%64)%64)+b'\n'
  cache=root/'cache.npz'
  with zipfile.ZipFile(cache,'w',compression=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED)as z:
   z.writestr('flattened.npy',p.make_header((6,3))+b''.join(struct.pack('<3e',i,i+1,i+2)for i in range(6)))
   z.writestr('reference_positions.npy',b'\x93NUMPY\x01\x00'+struct.pack('<H',len(axis))+axis+struct.pack('<3i',0,1,2))
   z.writestr('filenames.npy',b'UNTRUSTED OBJECT PAYLOAD MUST NOT BE DESERIALIZED')
  spec['cache']=p.record(cache);return spec
 def test_preparation_copies_only_selected_raw_rows_and_originals(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);spec=self.fixture(root);out=root/'out';r=p.prepare(spec,out)
   self.assertEqual(r['shape'],[4,3]);self.assertFalse(r['object_arrays_deserialized']);self.assertEqual(r['scientific_fits'],0)
   members=json.loads((out/'members-pax.json').read_text())['members'];self.assertIn('m0',[m['model']for m in members])
   with(out/'features.npy').open('rb')as f:
    header,_=p.npy_header(f);body=f.read()
   expected=b''.join(struct.pack('<3e',i,i+1,i+2)for i in r['selection']['source_indices'])
   self.assertEqual(body,expected);self.assertEqual(r['raw_feature_sha256'],hashlib.sha256(expected).hexdigest())
   self.assertEqual(header['shape'],(4,3));self.assertEqual(set(r['selection']['method_counts']),{'a','b'})
   for m in members:self.assertEqual(p.sha(m['local_path']),m['original_sha256'])
   with self.assertRaises(ValueError):p.prepare(spec,out)
 def test_tampered_mask_or_cache_never_creates_output(self):
  for fault in ['mask','cache']:
   with tempfile.TemporaryDirectory()as tmp:
    root=Path(tmp);spec=self.fixture(root);out=root/'out'
    if fault=='mask':spec['pair_mask_sha256']='b'*64
    else:Path(spec['cache']['path']).write_bytes(b'changed')
    with self.assertRaises(ValueError):p.prepare(spec,out)
    self.assertFalse(out.exists())
 def test_compressed_source_requires_explicit_different_reader(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);spec=self.fixture(root,compressed=True)
   with self.assertRaisesRegex(ValueError,'stored-array'):p.prepare(spec,root/'out')
 def test_selection_is_deterministic_and_rejects_invalid_size(self):
  rows=[dict(model_id='m'+str(i),method='m'+str(i%2),native_physicality='rejected'if i==0 else'not_assessed')for i in range(10)]
  self.assertEqual(p.choose(rows,4,42),p.choose(list(reversed(rows)),4,42))
  self.assertIn('m0',p.choose(rows,4,42))
  with self.assertRaises(ValueError):p.choose(rows,0,42)
if __name__=='__main__':unittest.main()
