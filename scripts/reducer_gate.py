"""Assemble both closed native proofs for the generic-job gate; never fit."""
import argparse
import json
from pathlib import Path
import sys


def assemble(request,output):
    source=str(Path(__file__).absolute().parent)
    if source not in sys.path:sys.path.insert(0,source)
    import mimas_cpu as cpu
    import reducer_diagnostic as diagnostic
    import reducer_trace_compare as saved
    request=json.loads(Path(request).read_text())
    if request.get('operation')!='assemble-native-pair-gate':raise ValueError('Explicit closed-proof request required')
    master=saved.data(request['master'])
    if saved.read(master['gate_source'])!=Path(__file__).absolute():raise ValueError('Unreviewed gate assembler')
    names={'pax-native','mimas-native'}
    if set(request['runs'])!=names:raise ValueError('Both native proofs required')
    for name,proof in request['runs'].items():
        if proof['plan']['sha256']!=master['host_plans'][name]['sha256']:raise ValueError('Different native plan')
    reference=saved.data(master['host_plans']['pax-native'])
    context=dict(inputs=reference['inputs'],native=reference['native'],portability_helper=reference['portability_helper'],
        native_gate_plan_shas={name:master['host_plans'][name]['sha256']for name in names})
    gate=dict(operation='closed-native-pair-gate',runs=request['runs'],master=request['master'],projection_fits=0)
    diagnostic.verify_native_pair(context,gate)
    output=Path(output)
    if not output.is_absolute()or any(p.is_symlink()for p in [output,*output.parents]):raise ValueError('Owned absolute gate path required')
    cpu.save(output,gate)
    return gate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest');parser.add_argument('output');args=parser.parse_args()
    assemble(args.manifest,args.output);print(json.dumps(dict(status='both_native_proofs_verified',projection_fits=0)))

if __name__=='__main__':main()
