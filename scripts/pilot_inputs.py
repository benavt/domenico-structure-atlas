"""Materialize identity-preserving pilot inputs from an accepted stored NPZ.

Copies selected raw rows and original files; never unpickles names, parses
coordinates, recomputes distances, imports an engine or fits a model.
"""
import argparse
import ast
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import zipfile


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def record(path):
    path=Path(path).absolute()
    need(path.is_file()and not any(p.is_symlink()for p in [path,*path.parents]),'Regular non-symlink input required')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def bound(entry):
    need(record(entry['path'])==entry,'Bound input changed: '+str(entry['path']))
    return Path(entry['path'])


def identity(path):
    item=Path(path).stat();return item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns


def read(entry):return json.loads(bound(entry).read_text())


def npy_header(stream):
    prefix=stream.read(8);need(prefix[:6]==b'\x93NUMPY'and prefix[6]in(1,2),'Supported NPY header required')
    size=2 if prefix[6]==1 else 4;length=struct.unpack('<H'if size==2 else'<I',stream.read(size))[0]
    need(0<length<=65536,'Bounded NPY header required')
    value=ast.literal_eval(stream.read(length).decode('latin1'))
    need(set(value)=={'descr','fortran_order','shape'}and value['fortran_order']is False,'C-order NPY required')
    need(isinstance(value['shape'],tuple)and all(type(n)is int and n>0 for n in value['shape']),'Positive fixed NPY shape required')
    return value,8+size+length


def make_header(shape):
    text=repr(dict(descr='<f2',fortran_order=False,shape=tuple(shape))).encode('ascii')
    text+=b' '*((64-(10+len(text)+1)%64)%64)+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(text))+text


def choose(rows,count,seed):
    need(type(count)is int and 1<=count<=len(rows),'Supported pilot size required')
    def rank(row):return hashlib.sha256((str(seed)+'|'+row['model_id']).encode()).hexdigest()
    selected={}
    rejected=[r for r in rows if r['native_physicality']=='rejected']
    if rejected:
        item=min(rejected,key=rank);selected[item['model_id']]=item
    groups={method:deque(sorted([r for r in rows if r['method']==method and r['model_id']not in selected],key=rank))
            for method in sorted({r['method']for r in rows})}
    while len(selected)<count:
        for group in groups.values():
            if group and len(selected)<count:
                item=group.popleft();selected[item['model_id']]=item
    return set(selected)


def save(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def prepare(spec,output):
    output=Path(output).absolute();need(not output.exists()and not any(p.is_symlink()for p in [output,*output.parents]),'Fresh pilot input destination required')
    intake=read(spec['intake']);native=read(spec['native_models']);proof=read(spec['feature_proof']);terminal=read(spec['terminal_report'])
    rows=intake['models'];ids=[r['model_id']for r in rows];order=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    need(len(set(ids))==len(ids)==spec['source_models']and order==proof['models_sha256'],'Accepted model order differs')
    retained=spec['retained_reference_positions'];need(retained==proof['retained_reference_positions']==terminal['retained_reference_positions'],'Retained core differs')
    need(spec['pair_mask_sha256']==proof['pair_mask_sha256']==terminal['pair_mask_sha256'],'Retained pair mask differs')
    expected={r['model_id']:r for r in native['models']if r['target']==spec['target']}
    need(set(expected)==set(ids),'Snapshot/intake identities differ')
    for row in rows:
        need(row['sha256']==expected[row['model_id']]['sha256']and row['method']==expected[row['model_id']]['method'],'Snapshot hashes/methods differ')
    selected=choose(list(expected.values()),spec['model_count'],spec['selection_seed'])
    indices=[i for i,name in enumerate(ids)if name in selected];chosen=[rows[i]for i in indices]
    cache=Path(spec['cache']['path']);before=identity(cache);bound(spec['cache'])
    dimensions=len(retained)*(len(retained)-1)//2
    with zipfile.ZipFile(cache)as archive:
        info=archive.getinfo('flattened.npy');need(info.compress_type==zipfile.ZIP_STORED,'Direct stored-array access required')
        with archive.open(info)as source:header,header_bytes=npy_header(source)
        need(header==dict(descr='<f2',fortran_order=False,shape=(len(ids),dimensions)),'Accepted feature shape/dtype differs')
        with archive.open('reference_positions.npy')as source:
            axis,_=npy_header(source);need(axis==dict(descr='<i4',fortran_order=False,shape=(len(retained),)),'Retained axis schema differs')
            need(list(struct.unpack('<'+'i'*len(retained),source.read()))==retained,'Stored reference axis differs')
        with cache.open('rb')as stream:
            stream.seek(info.header_offset);local=stream.read(30)
            need(local[:4]==b'PK\x03\x04','ZIP local header differs')
            flags,compression=struct.unpack('<HH',local[6:10]);need(not flags&1 and compression==0,'Unsupported stored ZIP member')
            name_bytes,extra_bytes=struct.unpack('<HH',local[26:30]);start=info.header_offset+30+name_bytes+extra_bytes+header_bytes
            output.mkdir(parents=True);(output/'models').mkdir();digest=hashlib.sha256()
            with(output/'features.npy').open('xb')as dest:
                dest.write(make_header((len(indices),dimensions)))
                for i in indices:
                    stream.seek(start+i*dimensions*2);row=stream.read(dimensions*2)
                    need(len(row)==dimensions*2,'Truncated selected row');digest.update(row);dest.write(row)
    members=[];scores=[]
    for row in chosen:
        name=row['model_id'];need(re.fullmatch('[A-Za-z0-9_.-]+',name)and name not in('.','..'),'Unsafe analysis ID')
        source=Path(row['path']);need(record(source)['sha256']==row['sha256'],'Original coordinate hash differs')
        destination=output/'models'/(name+source.suffix);shutil.copyfile(source,destination)
        need(sha(destination)==row['sha256'],'Selected coordinate copy differs')
        metadata=expected[name];members.append(dict(model=name,local_path=str(destination),original_sha256=row['sha256'],
            flags=dict(method=row['method'],native_physicality=metadata['native_physicality'],energy_t010=metadata['energy_t010'],energy_is_confidence=False)))
        scores.append(metadata)
    save(output/'members-pax.json',dict(members=members));save(output/'snapshot-metadata.json',dict(models=scores))
    need(identity(cache)==before,'Source cache changed during preparation')
    files=[record(p)for p in sorted(output.rglob('*'))if p.is_file()]
    result=dict(status='prepared_no_science',target=spec['target'],source_models=len(ids),model_count=len(indices),
        selection=dict(rule='one seeded rejected control when available; seeded method-round-robin, source row order',seed=spec['selection_seed'],source_indices=indices,
                       method_counts=dict(Counter(r['method']for r in chosen)),physicality_counts=dict(Counter(expected[r['model_id']]['native_physicality']for r in chosen))),
        shape=[len(indices),dimensions],dtype='float16',raw_feature_sha256=digest.hexdigest(),
        model_order_sha256=hashlib.sha256('\n'.join(r['model_id']for r in chosen).encode()).hexdigest(),
        retained_reference_positions=retained,pair_mask_sha256=spec['pair_mask_sha256'],originals_unchanged=True,
        source=spec,cache_identity_unchanged=True,source_archive_rehashed_once=True,object_arrays_deserialized=False,
        coordinate_parses=0,distance_recalculations=0,scientific_fits=0,files=files)
    save(output/'preparation.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('spec');parser.add_argument('output');args=parser.parse_args()
    print(json.dumps(prepare(json.loads(Path(args.spec).read_text()),args.output),indent=2))


if __name__=='__main__':main()
