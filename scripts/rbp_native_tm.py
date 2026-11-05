"""Bounded group commits around pinned native TM math; no execution entry point.

Rollback transactions use synchronous=EXTRA to sync journal-directory changes.
See sqlite.org/pragma.html#pragma_synchronous and sqlite.org/atomiccommit.html.
Numerical engines and pressure limits are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import gzip
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time

from rbp_states import core_feature_indices, native_call_bounds, need, record, save

LIBRARY_MODULES = ('src.analysis.nested_cluster_candidates', 'src.analysis.structure_io',
                   'src.analysis.tm_state_geometry', 'src.analysis.tm_state_sqlite')
INPUT_BATCH, PAIR_WAVE = 50, 25
STORAGE = 'single-sqlite-extra-wave-v1'


def frozen_source_evidence():
    result = {module:record(importlib.import_module(module).__file__) for module in LIBRARY_MODULES}
    result['consumer.rbp_native_tm'] = record(__file__)
    result['consumer.rbp_states'] = record(Path(__file__).with_name('rbp_states.py'))
    return result


def _bindings(value):
    need(isinstance(value, dict) and re.fullmatch('[0-9a-f]{64}', value.get('plan_sha256', ''))
         and all(re.fullmatch('[0-9a-f]{40}', value.get(key, '')) for key in ('consumer_commit','library_commit')),
         'Frozen plan and source commit bindings required')
    need(value.get('sources') == frozen_source_evidence(), 'Frozen native source evidence changed')


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _regular(path):
    path = Path(path).absolute()
    need('..' not in path.parts and path.is_file() and not any(p.is_symlink() for p in [path,*path.parents]),
         'Regular non-symlink evidence file required')
    return path


def _read_bound(entry):
    need(isinstance(entry,dict) and type(entry.get('bytes')) is int
         and re.fullmatch('[0-9a-f]{64}',entry.get('sha256','')), 'Exact path/size/SHA evidence required')
    path = _regular(entry['path']);need(record(path) == entry, 'Frozen reuse evidence changed')
    return path


def _readonly(path):
    from urllib.parse import quote
    db = sqlite3.connect('file:'+quote(str(_regular(path)))+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    need(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','SQLite evidence integrity failed')
    return db


def _fresh_db(path,schema):
    path=Path(path).absolute()
    need(not path.exists() and '..' not in path.parts and not any(p.is_symlink() for p in [path,*path.parents]),
         'Fresh non-symlink SQLite output required')
    db=sqlite3.connect(path,check_same_thread=False,isolation_level=None);db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=EXTRA')
    need(db.execute('PRAGMA synchronous').fetchone()[0]==3,'EXTRA durability unavailable')
    db.executescript('BEGIN IMMEDIATE;'+schema+'COMMIT;');return db


def _decode_canonical_row(row):
    with gzip.GzipFile(fileobj=io.BytesIO(row['evidence_gzip'])) as stream:raw=stream.read(8*1024*1024+1)
    need(len(raw)<=8*1024*1024 and hashlib.sha256(raw).hexdigest()==row['evidence_sha256'],'Canonical input evidence changed')
    full=json.loads(raw);compact_raw=row['compact_json'].encode()
    need(hashlib.sha256(compact_raw).hexdigest()==row['compact_sha256'],'Compact canonical input changed')
    expected={key:value for key,value in full.items() if key!='mapping'}
    expected.update(mapping_sha256=hashlib.sha256(_json(full['mapping'])).hexdigest(),canonical_input_evidence_sha256=row['evidence_sha256'])
    need(json.loads(compact_raw)==expected,'Compact/full native correspondence differs');return full


def read_canonical_evidence(database,model):
    with closing(_readonly(database)) as db:
        row=db.execute('SELECT * FROM inputs WHERE model=?',(model,)).fetchone()
        need(row is not None,'Missing canonical input evidence');return _decode_canonical_row(row)


class CanonicalEvidenceStore:
    """Pinned canonicalization; ≤50-row durable commits; old evidence read-only.

    reuse_inputs={'scope':record(...),'canonical_inputs':record(...)} must bind the
    same ordered originals/core/sequence/protocol and pinned library math hashes.
    Consumer storage code may differ. Original and saved CIF hashes are rechecked.
    """
    def __init__(self,members,core,sequence_sha256,output,check,*,base_factory=None,
                 reuse_inputs=None,source_bindings=None,protocol=None,checkpoint=None):
        from src.analysis.tm_state_geometry import CanonicalCAStore
        self.members={row['model']:dict(row) for row in members}
        need(len(self.members)==len(members) and self.members,'Distinct frozen source members required')
        self.names,self.check,self.output=list(self.members),check,Path(output)
        self.core,self.sequence_sha256=[i+1 for i in core],sequence_sha256
        self.output.mkdir(parents=True,exist_ok=False);self.database=self.output/'canonical-inputs.sqlite'
        self._lock,self._closed,self._batch=threading.RLock(),False,False
        need(checkpoint is None or callable(checkpoint), 'Optional checkpoint callback must be callable')
        self.checkpoint = checkpoint
        self._stopped = False
        self._reuse,self.reuse_inputs=None,reuse_inputs
        self._db=_fresh_db(self.database,'''
            CREATE TABLE inputs(model TEXT PRIMARY KEY,compact_json TEXT NOT NULL,compact_sha256 TEXT NOT NULL,
                evidence_gzip BLOB NOT NULL,evidence_sha256 TEXT NOT NULL,origin TEXT NOT NULL);
            CREATE TRIGGER immutable_input BEFORE UPDATE ON inputs BEGIN SELECT RAISE(ABORT,'Immutable canonical input'); END;
            CREATE TRIGGER no_input_delete BEFORE DELETE ON inputs BEGIN SELECT RAISE(ABORT,'Canonical input cannot be deleted'); END;
            CREATE TABLE batches(id INTEGER PRIMARY KEY,rows INTEGER NOT NULL,reused_rows INTEGER NOT NULL,
                total_inputs INTEGER NOT NULL,committed_ns INTEGER NOT NULL);
        ''')
        self.base=(base_factory or CanonicalCAStore)(members,self.core,sequence_sha256,self.output/'canonical-ca')
        try:
            if reuse_inputs is not None:
                need(set(reuse_inputs)=={'scope','canonical_inputs'},'Exact canonical reuse descriptors required')
                scope=json.loads(_read_bound(reuse_inputs['scope']).read_text())
                previous=[(r['model'],r['source_path'],r['sha256']) for r in scope['members']]
                current=[(r['model'],r['source_path'],r['sha256']) for r in members]
                need(previous==current and scope['canonical_positions']==self.core
                     and scope['sequence_sha256']==sequence_sha256 and scope['protocol']==protocol,
                     'Reused canonical registry/core/sequence/protocol differs')
                old,new=scope['source_bindings'],source_bindings or {}
                need(old.get('library_commit')==new.get('library_commit') and all(
                    old.get('sources',{}).get(m,{}).get('sha256')==new.get('sources',{}).get(m,{}).get('sha256')
                    and re.fullmatch('[0-9a-f]{64}',new.get('sources',{}).get(m,{}).get('sha256','')) for m in LIBRARY_MODULES),
                    'Reused canonical library math differs')
                self._reuse=_readonly(_read_bound(reuse_inputs['canonical_inputs']))
                need(all(row[0] in self.members for row in self._reuse.execute('SELECT model FROM inputs')),
                     'Reused canonical cache contains unknown models')
        except BaseException:self.close();raise

    def _materialize(self,model):
        self.check();need(not self._closed and not self._stopped and model in self.members,'Unknown/closed/stopped canonical input')
        member=self.members[model]
        need(record(member['source_path'])['sha256']==member['sha256'],'Original model hash changed')
        cached=self._db.execute('SELECT compact_json,compact_sha256 FROM inputs WHERE model=?',(model,)).fetchone()
        if cached:
            raw=cached['compact_json'].encode()
            need(hashlib.sha256(raw).hexdigest()==cached['compact_sha256'],'Canonical evidence hash changed');compact=json.loads(raw)
        else:
            old=None if self._reuse is None else self._reuse.execute('SELECT * FROM inputs WHERE model=?',(model,)).fetchone()
            if old is not None:full,origin=_decode_canonical_row(old),'reuse'
            else:
                try:full,origin=self.base.canonical_record(model),'new'
                finally:self.base._records.clear()
            need(full['model']==model and full['source_sha256']==member['sha256'] and full['canonical_positions']==self.core
                 and full['sequence_sha256']==self.sequence_sha256 and full['ca_count']==len(self.core)
                 and [r['canonical_position'] for r in full['mapping']]==self.core,'Canonical source/core/sequence binding changed')
            need(record(_regular(full['path']))['sha256']==full['sha256'],'Canonical CA file changed')
            raw=_json(full);proof_sha=hashlib.sha256(raw).hexdigest()
            compact={key:value for key,value in full.items() if key!='mapping'}
            compact.update(mapping_sha256=hashlib.sha256(_json(full['mapping'])).hexdigest(),canonical_input_evidence_sha256=proof_sha)
            encoded=_json(compact)
            self._db.execute('INSERT INTO inputs VALUES(?,?,?,?,?,?)',(model,encoded.decode(),
                hashlib.sha256(encoded).hexdigest(),gzip.compress(raw,mtime=0),proof_sha,origin))
        need(record(_regular(compact['path']))['sha256']==compact['sha256'],'Canonical CA file changed')
        self.check();return compact

    def canonical_record(self,model):
        with self._lock:
            if self._batch:return self._materialize(model)
            if self._db.execute('SELECT 1 FROM inputs WHERE model=?',(model,)).fetchone() is None:self.prepare([model])
            return self._materialize(model)

    def prepare(self,models=None):
        names=self.names if models is None else list(models)
        need(len(names)==len(set(names)) and set(names)<=set(self.names),'Invalid canonical preparation registry')
        with self._lock:
            need(not self._batch and not self._closed and not self._stopped,'Canonical batch already active/closed/stopped')
            for offset in range(0,len(names),INPUT_BATCH):
                before=self._db.total_changes;reused_before=self.reused_count
                self._db.execute('BEGIN IMMEDIATE');self._batch=True
                try:
                    for name in names[offset:offset+INPUT_BATCH]:self._materialize(name)
                    changed=self._db.total_changes-before
                    if changed:self._db.execute('INSERT INTO batches(rows,reused_rows,total_inputs,committed_ns) VALUES(?,?,?,?)',
                        (changed,self.reused_count-reused_before,self.prepared_count,time.time_ns()))
                    self._db.execute('COMMIT')
                except BaseException:
                    if self._db.in_transaction:self._db.execute('ROLLBACK')
                    raise
                finally:self._batch=False
                if changed and self.checkpoint is not None:
                    try:
                        self.checkpoint('canonical-inputs:'+str(self.prepared_count))
                    except BaseException:
                        self._stopped = True
                        raise
            return len(names)

    @property
    def reused_count(self):return self._db.execute("SELECT count(*) FROM inputs WHERE origin='reuse'").fetchone()[0]

    @property
    def prepared_count(self):return self._db.execute('SELECT count(*) FROM inputs').fetchone()[0]

    def close(self):
        with self._lock:
            if self._closed:return
            try:
                if self._db.in_transaction:self._db.execute('ROLLBACK')
                need(self._db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','Canonical SQLite integrity failed')
                if self.reuse_inputs is not None:
                    for entry in self.reuse_inputs.values():_read_bound(entry)
            finally:
                if self._reuse is not None:self._reuse.close()
                self._db.close();self._closed=True


class NativeTMRun:
    """Pinned scorer with atomic ≤25-pair waves and exact cumulative reservations.

    The sole native-tm.sqlite commits the full wave intent BEFORE calls. A second
    atomic transaction commits all proofs/directions/queries and the barrier.
    Interrupted waves remain pending and fully charged; no automatic retry exists.
    """
    def __init__(self,members,retained_core_indices,sequence_sha256,output,*,runtime,source_bindings,check,
                 maximum_directional_calls=None,reserve_calls=lambda count:None,reuse_inputs=None,checkpoint=None,
                 _base_factory=None,_runner=None):
        from src.analysis.tm_state_geometry import protocol_descriptor
        from src.analysis.tm_state_sqlite import SQLiteNativeTMScorer
        _bindings(source_bindings);core=list(retained_core_indices)
        core_feature_indices(max(core)+1 if core else 0,core)
        cap=native_call_bounds(len(members))['directional_call_ceiling']
        maximum_directional_calls=cap if maximum_directional_calls is None else maximum_directional_calls
        need(type(maximum_directional_calls) is int and 0<=maximum_directional_calls<=cap,'Native reservation cap differs')
        need(callable(check),'Explicit pressure guard required')
        need(checkpoint is None or callable(checkpoint), 'Optional checkpoint callback must be callable')
        self.checkpoint = checkpoint
        self.output,self.check=Path(output),check;self.check()
        need(not self.output.exists(),'Fresh native run output required');self.output.mkdir(parents=True)
        self.source_bindings,self.runtime=source_bindings,dict(runtime)
        self._wave_lock,self._local,self._closed=threading.RLock(),threading.local(),False
        self._reserved,self._direction_count,self._wave=0,0,None
        self._outer_reserve,self._runner=reserve_calls,_runner or subprocess.run
        self.maximum_directional_calls=maximum_directional_calls
        protocol=protocol_descriptor(runtime['usalign'],runtime['usalign_sha256'])
        self.store=CanonicalEvidenceStore(members,core,sequence_sha256,self.output/'inputs',check,
            base_factory=_base_factory,reuse_inputs=reuse_inputs,source_bindings=source_bindings,protocol=protocol,
            checkpoint=None if checkpoint is None else self._durable_checkpoint)
        self.database=self.output/'native-tm.sqlite';owner=self
        class GroupedSQLiteScorer(SQLiteNativeTMScorer):
            def _transaction(self,statement,parameters):
                with self._db_lock:
                    need(owner._wave is not None and self._connection.in_transaction and not self._closed,
                         'Native proof write outside a durable reserved wave')
                    need(self._connection.execute(statement,parameters).rowcount==1,'Native proof update count differs')
        try:
            self.scorer=GroupedSQLiteScorer(self.store,runtime['usalign'],self.database,maximum_directional_calls,
                executable_sha256=runtime['usalign_sha256'],timeout=runtime.get('timeout_seconds',180),
                reserve_calls=self._consume,on_call=self._begin,check_cancel=check,runner=self._direction)
        except BaseException:self.store.close();self._closed=True;raise
        self.protocol,self.failed=self.scorer.protocol,self.scorer.failed
        self._db,self._lock=self.scorer._connection,self.scorer._db_lock
        self.directions_database=self.waves_database=self.database
        self._db.execute('PRAGMA synchronous=EXTRA')
        need(self._db.execute('PRAGMA synchronous').fetchone()[0]==3,'EXTRA native durability unavailable')
        self._db.executescript('''BEGIN IMMEDIATE;
            CREATE TABLE waves(id INTEGER PRIMARY KEY,state TEXT NOT NULL,intent_json TEXT NOT NULL,
                intent_sha256 TEXT NOT NULL,completion_json TEXT,completion_sha256 TEXT);
            CREATE TRIGGER immutable_wave BEFORE UPDATE ON waves WHEN OLD.completion_json IS NOT NULL
                BEGIN SELECT RAISE(ABORT,'Immutable sealed native wave'); END;
            CREATE TRIGGER no_wave_delete BEFORE DELETE ON waves BEGIN SELECT RAISE(ABORT,'Native wave cannot be deleted'); END;
            CREATE TABLE directions(id INTEGER PRIMARY KEY,wave_id INTEGER NOT NULL,state TEXT NOT NULL,
                command_json TEXT NOT NULL,started_ns INTEGER NOT NULL,result_json TEXT);
            CREATE TRIGGER immutable_direction BEFORE UPDATE ON directions WHEN OLD.state!='PENDING'
                BEGIN SELECT RAISE(ABORT,'Immutable native direction'); END;
            CREATE TRIGGER no_direction_delete BEFORE DELETE ON directions BEGIN SELECT RAISE(ABORT,'Native direction cannot be deleted'); END;
            CREATE TABLE model_pairs(first TEXT NOT NULL,second TEXT NOT NULL,score REAL NOT NULL,
                result_key TEXT NOT NULL,wave_id INTEGER NOT NULL,PRIMARY KEY(first,second));
            CREATE TRIGGER immutable_model_pair BEFORE UPDATE ON model_pairs BEGIN SELECT RAISE(ABORT,'Immutable native model pair'); END;
            CREATE TRIGGER no_model_pair_delete BEFORE DELETE ON model_pairs BEGIN SELECT RAISE(ABORT,'Native model pair cannot be deleted'); END;
            COMMIT;
        ''')
        save(self.output/'native-scope.json',dict(source_bindings=source_bindings,runtime=runtime,
            canonical_positions=[i+1 for i in core],sequence_sha256=sequence_sha256,members=members,
            maximum_directional_calls=maximum_directional_calls,protocol=self.protocol,automatic_retries=0,
            storage=STORAGE,input_batch_size=INPUT_BATCH,maximum_wave_pairs=PAIR_WAVE,reuse_inputs=reuse_inputs))

    def _durable_checkpoint(self, checkpoint_id):
        if self.checkpoint is None:
            return
        try:
            self.checkpoint(checkpoint_id)
        except BaseException:
            self.failed.set()
            raise

    @property
    def directional_calls(self):return self.scorer.directional_calls

    @property
    def reserved_calls(self):return self._reserved

    def _consume(self,count):
        with self._lock:
            self.check();need(self._wave is not None and self._wave['consumed']+count<=self._wave['reserved'],
                              'Native call lacks durable wave reservation')
            self._wave['consumed']+=count

    def _begin(self,command):
        with self._lock:
            self.check();need(self._wave is not None and (command[1],command[2]) in self._wave['directions'],
                              'Native direction is outside the approved wave')
            self._direction_count+=1;need(self._direction_count<=self._reserved,'Native direction lacks reservation')
            cursor=self._db.execute('INSERT INTO directions(wave_id,state,command_json,started_ns) VALUES(?,?,?,?)',
                (self._wave['id'],'PENDING',_json(command).decode(),time.time_ns()))
            self._local.direction_id=cursor.lastrowid

    def _direction(self,command,**kwargs):
        try:
            self.check();result=self._runner(command,**kwargs)
            evidence=dict(stdout=result.stdout,stderr=result.stderr,returncode=getattr(result,'returncode',0),
                          matrix_text=Path(command[-1]).read_text());state='COMPLETE'
        except BaseException as error:
            evidence,state=dict(error_type=type(error).__name__,error=str(error)),'FAILED'
            with self._lock:self._db.execute('UPDATE directions SET state=?,result_json=? WHERE id=? AND state=?',
                (state,_json(evidence).decode(),self._local.direction_id,'PENDING'))
            raise
        with self._lock:
            need(self._db.execute('UPDATE directions SET state=?,result_json=? WHERE id=? AND state=?',
                (state,_json(evidence).decode(),self._local.direction_id,'PENDING')).rowcount==1,'Native direction completion differs')
        self.check();return result

    def _pair_intent(self,pair):
        from src.analysis.tm_state_geometry import digest,validate_cached_pair
        from src.analysis.tm_state_sqlite import decode_complete
        inputs=sorted([self.store.canonical_record(name) for name in pair],key=lambda r:(r['source_sha256'],r['sha256'],r['model']))
        first,second=inputs
        fingerprint=dict(protocol_sha256=self.scorer.protocol_sha256,sequence_sha256=first['sequence_sha256'],
            canonical_positions=first['canonical_positions'],
            original_and_canonical_sha256=sorted([[r['source_sha256'],r['sha256']] for r in inputs]))
        key=digest(fingerprint);existing=self.scorer._lookup(key)
        if existing is not None:
            proof=decode_complete(existing);validate_cached_pair(proof['result'],key,fingerprint,self.protocol)
        needed=0 if existing is not None or first['sha256']==second['sha256'] else 2
        return dict(key=key,fingerprint=fingerprint,inputs=inputs,reserved_calls=needed,existing_complete=existing is not None)

    def _one(self,pair):
        result=self.scorer.score(*pair)
        if pair[0]!=pair[1]:
            with self._lock:
                row=self._db.execute('SELECT score,result_key FROM model_pairs WHERE first=? AND second=?',pair).fetchone()
                if row is None:self._db.execute('INSERT INTO model_pairs VALUES(?,?,?,?,?)',(*pair,result['score'],result['key'],self._wave['id']))
                else:need(row['score']==result['score'] and row['result_key']==result['key'],'Native pair cache differs')
        return result

    def _score_wave(self,pairs,workers):
        need(1<=len(pairs)<=PAIR_WAVE and len(set(pairs))==len(pairs),'Native pair wave exceeds fixed bound')
        self.check();self.store.prepare(sorted({name for pair in pairs for name in pair}))
        intents,queries={},[]
        for pair in pairs:
            intent=self._pair_intent(pair);intents.setdefault(intent['key'],intent)
            queries.append(dict(first=pair[0],second=pair[1],result_key=intent['key'],inputs=intent['inputs']))
        reserved=sum(row['reserved_calls'] for row in intents.values());before=self.directional_calls
        need(self._reserved+reserved<=self.maximum_directional_calls,'Native reservation ceiling exceeded before wave')
        self._outer_reserve(reserved)
        value=dict(storage=STORAGE,protocol_sha256=self.scorer.protocol_sha256,pairs=queries,intents=list(intents.values()),
            reserved_calls=reserved,cumulative_reserved_before=self._reserved,cumulative_reserved_after=self._reserved+reserved,
            maximum_directional_calls=self.maximum_directional_calls)
        raw=_json(value)
        with self._lock:
            cursor=self._db.execute('INSERT INTO waves(state,intent_json,intent_sha256) VALUES(?,?,?)',
                ('PENDING',raw.decode(),hashlib.sha256(raw).hexdigest()))  # EXTRA commit BEFORE any calls.
            self._reserved+=reserved
            directions={(a['path'],b['path']) for query in queries if intents[query['result_key']]['reserved_calls']
                        for a,b in (query['inputs'],list(reversed(query['inputs'])))}
            self._wave=dict(id=cursor.lastrowid,reserved=reserved,consumed=0,directions=directions)
            try:self._db.execute('BEGIN IMMEDIATE')
            except BaseException:
                self.failed.set()
                if self._db.in_transaction:self._db.execute('ROLLBACK')
                self._wave=None;raise
        committed_wave_id = self._wave['id']
        results,error=[],None
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(self._one,pairs))
            need(self._wave['consumed']==reserved and self.directional_calls-before==reserved,
                 'Native wave reservation/actual calls differ')
        except BaseException as caught:error=caught;self.failed.set()
        try:
            complete=[]
            for key in sorted(intents):
                row=self.scorer._lookup(key)
                if row is not None and row['state']=='COMPLETE':
                    complete.append(dict(key=key,proof_sha256=row['proof_sha256'],native_calls=row['native_calls']))
            completion=dict(complete_proofs=complete,actual_calls_before=before,actual_calls_after=self.directional_calls,
                consumed_reservation=self._wave['consumed'],cumulative_reserved=self._reserved,
                error=None if error is None else dict(type=type(error).__name__,message=str(error)))
            raw=_json(completion)
            with self._lock:
                self._commit_barrier(error,raw)
        except BaseException:
            self.failed.set()
            if self._db.in_transaction:self._db.execute('ROLLBACK')
            raise
        finally:self._wave=None
        if error is not None:raise error
        self._durable_checkpoint(f'native-wave:{committed_wave_id}:{reserved}')
        self.check();return results

    def _commit_barrier(self,error,raw):
        self._db.execute('UPDATE waves SET state=?,completion_json=?,completion_sha256=? WHERE id=?',
            ('COMPLETE' if error is None else 'PENDING',raw.decode(),hashlib.sha256(raw).hexdigest(),self._wave['id']))
        self._db.execute('COMMIT')  # Atomic EXTRA proof/direction/query/completion barrier.

    def prepare(self):self.check();return self.store.prepare()

    def score_many(self,pairs,*,workers=1):
        """Ordered results; exact reservations for waves of ≤25 unordered pairs."""
        from src.analysis.tm_state_geometry import preferred_forward_transform
        pairs=[tuple(pair) for pair in pairs]
        need(type(workers) is int and 1<=workers<=8 and all(len(pair)==2 and all(name in self.store.members for name in pair)
             for pair in pairs),'Invalid native wave requests/workers')
        with self._wave_lock:
            need(not self._closed,'Native run already closed')
            if self.failed.is_set():raise RuntimeError('TM batch stopped after a prior failure; no automatic retry')
            unique=sorted({tuple(sorted(pair)) for pair in pairs});results={}
            try:
                for offset in range(0,len(unique),PAIR_WAVE):
                    chunk=unique[offset:offset+PAIR_WAVE];results.update(zip(chunk,self._score_wave(chunk,workers)))
            except BaseException:self.failed.set();raise
            ordered=[]
            for first,second in pairs:
                canonical=tuple(sorted((first,second)));value=dict(results[canonical])
                if (first,second)!=canonical:value['forward'],value['reverse']=value['reverse'],value['forward']
                value['forward']=dict(value['forward'],source_model=first,target_model=second)
                value['reverse']=dict(value['reverse'],source_model=second,target_model=first)
                value['transforms']={key:value[key]['transform'] for key in ('forward','reverse')}
                value['display_transform']=preferred_forward_transform(value);ordered.append(value)
            return ordered

    def score(self,first,second):return self.score_many([(first,second)])[0]

    def close(self,*,require_complete=True):
        if self._closed:return
        error=None;prepared,reused=self.store.prepared_count,self.store.reused_count
        try:
            need(self._wave is None and not self._db.in_transaction,'Unresolved native group transaction')
            if require_complete:need(self._db.execute("SELECT count(*) FROM waves WHERE state!='COMPLETE'").fetchone()[0]==0
                and self._db.execute("SELECT count(*) FROM directions WHERE state!='COMPLETE'").fetchone()[0]==0,
                'Unresolved native wave/directions; no automatic retry')
            self.scorer.close(require_complete=require_complete)
        except BaseException as caught:
            error=caught
            if not self.scorer._closed:self.scorer.close(require_complete=False)
        finally:
            try:self.store.close()
            except BaseException as caught:
                if error is None:error=caught
            self._closed=True
            save(self.output/'evidence.json',dict(schema_version=2,storage=STORAGE,protocol=self.protocol,
                native_evidence=dict(format='canonical-TM-lossless-sqlite-proof-v1',file=record(self.database)),
                canonical_inputs=record(self.store.database),native_directions=record(self.database),native_waves=record(self.database),
                prepared_inputs=prepared,reused_inputs=reused,actual_directional_calls=self.directional_calls,
                reserved_directional_calls=self.reserved_calls,maximum_directional_calls=self.maximum_directional_calls,
                maximum_wave_pairs=PAIR_WAVE,input_batch_size=INPUT_BATCH,complete=require_complete and error is None,
                error=None if error is None else dict(type=type(error).__name__,message=str(error)),source_bindings=self.source_bindings))
        if error is not None:raise error

    def __enter__(self):return self

    def __exit__(self,error_type,error,traceback):self.close(require_complete=error_type is None)


def read_completed_columns(output,*,expected_source_bindings,expected_scope_sha256,return_evidence=False):
    """Verify the committed proof prefix and expose any charged pending wave.

    This grants no execution permission, opens no scorer, and retries no intent.
    Legacy separate databases remain readable. New query evidence requires an
    atomic completion/partial-completion barrier; unsealed pending work is skipped.
    """
    import numpy as np
    from src.analysis.tm_state_geometry import validate_cached_pair
    from src.analysis.tm_state_sqlite import decode_complete
    output=Path(output).absolute()
    need(not any(p.is_symlink() for p in [output,*output.parents]),'Symlinked native output forbidden')
    need(record(output/'native-scope.json')['sha256']==expected_scope_sha256,'Frozen native scope hash differs')
    scope=json.loads((output/'native-scope.json').read_text())
    need(scope['source_bindings']==expected_source_bindings,'Frozen recovery source bindings differ')
    names=[row['model'] for row in scope['members']];index={name:i for i,name in enumerate(names)}
    hashes={};expected={row['model']:row for row in scope['members']}
    with closing(_readonly(output/'inputs/canonical-inputs.sqlite')) as inputs:
        for row in inputs.execute('SELECT * FROM inputs'):
            full=_decode_canonical_row(row)
            need(full['model'] in expected and full['source_sha256']==expected[full['model']]['sha256']
                 and full['sequence_sha256']==scope['sequence_sha256'] and full['canonical_positions']==scope['canonical_positions'],
                 'Frozen canonical source binding differs')
            hashes[full['model']]=[full['source_sha256'],full['sha256']]
    values,degrees,wave_status,charged={},[0]*len(names),[],0
    with closing(_readonly(output/'native-tm.sqlite')) as proofs:
        modern=proofs.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='waves'").fetchone()[0]==1
        queries=proofs if modern else _readonly(output/'native-directions.sqlite')
        try:
            barriers={};logged_queries={}
            if modern:
                known=set();pending_seen=False;last_actual=0
                for row in proofs.execute('SELECT * FROM waves ORDER BY id'):
                    raw=row['intent_json'].encode();need(hashlib.sha256(raw).hexdigest()==row['intent_sha256'],'Native wave intent changed')
                    intent=json.loads(raw)
                    from src.analysis.tm_state_geometry import digest
                    need(not pending_seen and 1<=len(intent['pairs'])<=PAIR_WAVE
                         and len({(q['first'],q['second']) for q in intent['pairs']})==len(intent['pairs'])
                         and intent['protocol_sha256']==digest(scope['protocol'])
                         and intent['reserved_calls']==sum(p['reserved_calls'] for p in intent['intents'])
                         and len({p['key'] for p in intent['intents']})==len(intent['intents'])
                         and intent['cumulative_reserved_before']==charged
                         and intent['cumulative_reserved_after']==charged+intent['reserved_calls']
                         and intent['maximum_directional_calls']==scope['maximum_directional_calls']
                         and intent['cumulative_reserved_after']<=scope['maximum_directional_calls'],'Native wave reservation differs')
                    for planned in intent['intents']:
                        identity=planned['fingerprint']['original_and_canonical_sha256']
                        need(planned['key']==digest(planned['fingerprint'])
                             and planned['fingerprint']['sequence_sha256']==scope['sequence_sha256']
                             and planned['fingerprint']['canonical_positions']==scope['canonical_positions']
                             and (not planned['existing_complete'] or planned['key'] in known)
                             and planned['reserved_calls']==(0 if planned['existing_complete'] or identity[0][1]==identity[1][1] else 2),
                             'Native wave fingerprint/exact reservation differs')
                    planned_keys={p['key'] for p in intent['intents']}
                    need({p['result_key'] for p in intent['pairs']}==planned_keys,'Native logged query/intent coverage differs')
                    logged_queries[row['id']]={(p['first'],p['second'],p['result_key']) for p in intent['pairs']}
                    charged=intent['cumulative_reserved_after'];complete={}
                    if row['completion_json'] is not None:
                        raw=row['completion_json'].encode()
                        need(hashlib.sha256(raw).hexdigest()==row['completion_sha256'],'Native completion barrier changed')
                        completion=json.loads(raw);complete={p['key']:p for p in completion['complete_proofs']}
                        need(completion['cumulative_reserved']==charged and completion['actual_calls_before']==last_actual
                             and 0<=completion['actual_calls_after']-last_actual<=intent['reserved_calls']
                             and 0<=completion['consumed_reservation']<=intent['reserved_calls']
                             and set(complete)<=planned_keys,
                             'Native barrier reservation/actual call bound differs')
                        if row['state']=='COMPLETE':
                            need(completion['actual_calls_after']-last_actual==completion['consumed_reservation']==intent['reserved_calls'],
                                 'Complete native wave exact call accounting differs')
                        last_actual=completion['actual_calls_after']
                        if row['state']=='COMPLETE':need(completion['error'] is None and set(complete)=={p['result_key'] for p in intent['pairs']},
                            'Native complete-wave proof coverage differs')
                    else:need(row['state']=='PENDING','Complete native wave lacks barrier')
                    barriers[row['id']]=complete
                    known.update(complete);pending_seen=row['state']!='COMPLETE'
                    wave_status.append(dict(id=row['id'],state=row['state'],reserved_calls=intent['reserved_calls'],
                                            verified_complete_pairs=len(complete),sealed=row['completion_json'] is not None))
            else:
                charged=proofs.execute('SELECT coalesce(sum(reserved_calls),0) FROM proofs').fetchone()[0]
                need(charged<=scope['maximum_directional_calls'],'Legacy native reservations exceed scope')
            for row in queries.execute('SELECT * FROM model_pairs'):
                if modern:
                    need((row['first'],row['second'],row['result_key']) in logged_queries.get(row['wave_id'],set()),
                         'Model pair is outside its logged native wave queries')
                    barrier=barriers.get(row['wave_id'],{}).get(row['result_key'])
                    if barrier is None:continue
                saved=proofs.execute('SELECT * FROM proofs WHERE pair_key=?',(row['result_key'],)).fetchone()
                need(saved is not None and saved['state']=='COMPLETE','Column lacks complete native proof')
                if modern:need(saved['proof_sha256']==barrier['proof_sha256'] and saved['native_calls']==barrier['native_calls'],
                               'Barrier/native proof differs')
                decoded=decode_complete(saved);fingerprint=decoded['intent']['fingerprint']
                result=validate_cached_pair(decoded['result'],decoded['key'],fingerprint,scope['protocol'])
                left,right=row['first'],row['second']
                need(left in index and right in index and left<right and row['score']==result['score']
                     and fingerprint['canonical_positions']==scope['canonical_positions']
                     and fingerprint['sequence_sha256']==scope['sequence_sha256'] and left in hashes and right in hashes
                     and sorted([hashes[left],hashes[right]])==fingerprint['original_and_canonical_sha256'],'Column source/score binding differs')
                i,j=index[left],index[right];values[tuple(sorted((i,j)))]=float(row['score']);degrees[i]+=1;degrees[j]+=1
        finally:
            if not modern:queries.close()
    columns={i:np.asarray([1. if i==j else values[tuple(sorted((i,j)))] for j in range(len(names))])
             for i,count in enumerate(degrees) if count==len(names)-1}
    if return_evidence:return dict(columns=columns,waves=wave_status,historical_charged_calls=charged,
        pending_waves=[row for row in wave_status if row['state']!='COMPLETE'],automatic_retry=False)
    return columns
