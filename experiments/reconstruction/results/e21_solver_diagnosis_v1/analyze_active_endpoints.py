"""Post-run endpoint analysis only: no fitting, promotion, or source mutation.

Run only after the staged experiment finishes. Reports are separate from the
frozen 22-attempt experiment. Inputs and this script are hashed in the output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.reconstruction import e21_simulation, e21_protocol
from geom_mesh_net.fields import constrained_joint_fit as cj
from geom_mesh_net.fields import solver_diagnostics as sd


ROOT = Path('/Users/rolandbennett/projects/geom-mesh-net')
REPORT = ROOT / 'experiments/reconstruction/results/e21_solver_diagnosis_v1/diagnosis.json'
CASE = ROOT / 'experiments/reconstruction/results/e21_artifacts/v3/cases/57babaa91366f5b10dd88bb3'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slopes(points):
    if len(points) != 4 or not all(p['valid'] for p in points):
        return {'verified': False, 'reason': 'invalid_or_unavailable_bracket', 'directional_cusp_minimum': False}
    a,b,c,d = points
    if b['step'] <= a['step'] or d['step'] <= c['step'] or len({p['parameter_identity'] for p in points}) != 4:
        return {'verified':False,'reason':'bracket_below_float_resolution','directional_cusp_minimum':False}
    left_fd=(b['mean_nll']-a['mean_nll'])/(b['step']-a['step'])
    right_fd=(d['mean_nll']-c['mean_nll'])/(d['step']-c['step'])
    scale=max(1.,*(abs(p['mean_nll']) for p in points))
    left_error=32*np.finfo(float).eps*scale/(b['step']-a['step'])+abs(a['derivative']-b['derivative'])
    right_error=32*np.finfo(float).eps*scale/(d['step']-c['step'])+abs(d['derivative']-c['derivative'])
    left,right=b['derivative'],c['derivative']
    verified=abs(left_fd-left)<=left_error and abs(right_fd-right)<=right_error
    resolved=abs(right-left)>4*(left_error+right_error)
    return {'verified': bool(verified), 'left_derivative': left,'right_derivative': right,
            'left_finite_difference':left_fd,'right_finite_difference':right_fd,
            'left_error_allowance':left_error,'right_error_allowance':right_error,
            'derivative_jump':right-left,'resolved_jump':bool(resolved),
            'directional_cusp_minimum':bool(verified and resolved and left < -left_error and right > right_error)}


def analyze(stage,start,*,top_n=3,window=1e-3,side_offset=1e-6,max_evaluations=32):
    begun=time.perf_counter()
    report=json.loads(REPORT.read_text())
    if not report.get('all_diagnostic_stages_complete'):
        raise RuntimeError('main diagnostic experiment is incomplete; do not run probes concurrently')
    rows=[r for r in report['rows'] if r['case']=='observed' and r['stage']==stage and r['start']==start]
    if len(rows)!=1:
        raise ValueError('exactly one requested stage/start is required')
    row=rows[0]
    fit_path=ROOT / row['fit_artifact']
    if file_hash(fit_path)!=row['fit_sha256']:
        raise ValueError('saved fit artifact hash differs from the compact report')
    fit=json.loads(fit_path.read_text());state=fit['starts'][0]['last_accepted_state']
    if state!=row['last_accepted_state']:
        raise ValueError('terminal state differs from the verified fit artifact')
    metadata=json.loads((CASE/'metadata.json').read_text())
    design={'simulation':metadata['configuration'],**metadata['observation']}
    arrays,metadata=e21_simulation.load_case(CASE,metadata['case'],design)
    if metadata['data_identity']!=row['data_identity']['data_identity']:
        raise ValueError('case data identity differs from the staged fit')
    mask=arrays['fitting'];sample_indices=np.flatnonzero(mask)
    x,y=arrays['x'][mask],arrays['y'][mask].astype(float)
    if e21_protocol.array_hash(x,y)!=row['data_identity']['fitting_identity']:
        raise ValueError('fitting observations differ from the staged fit')
    model=cj.restore_constrained_state(state);vector=model.vector.numpy().copy()
    tx,ty=torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64)
    assignment=model.assign(tx);fixed=cj._fixed_probes(model.box)
    evaluator=cj._Evaluation(model,tx,ty,assignment,1,[],fixed)
    evaluated=evaluator.evaluate(vector)
    lower,upper=model.layout.bounds(model.box,model.radius_bounds)
    constraints=cj.geometry_constraints(model.layout,box=model.box,min_gap=model.min_gap)
    matrix,slack=cj._linearized_constraints(vector,evaluated,constraints,lower,upper)
    subspace=cj._ParameterSubspace(vector,model.layout,fit_centres=row['fit_centres'],fit_radii=row['fit_radii'])
    active=subspace.active
    reduced,lam,certificate=cj._project_direction(-evaluated[1][active],matrix[:,active],slack)
    joint=np.zeros(len(vector));joint[active]=reduced
    radius_indices=model.layout.blocks()['radii'];k=model.layout.count
    radii=vector[radius_indices]*cj.GEOMETRY_SCALE
    with torch.no_grad():
        distances=torch.cdist(tx,model.centres()).numpy()
    owners=assignment.numpy();nearest=distances[np.arange(len(x)),owners]
    ranking=np.argsort(-abs(joint[radius_indices]),kind='stable')[:top_n]
    output={'schema':'active-radius-postrun-probes-v1','scope':'separate diagnostic; no fitting; no convergence or inference certificate',
            'stage':stage,'start':start,'fit_centres':row['fit_centres'],'fit_radii':row['fit_radii'],
            'provenance':{'script_sha256':file_hash(__file__),'main_report_sha256':file_hash(REPORT),
                          'fit_path':str(fit_path),'fit_sha256':file_hash(fit_path),
                          'solver_source_sha256':file_hash(cj.__file__),'diagnostics_source_sha256':file_hash(sd.__file__),
                          'data_identity':row['data_identity'],'execution_fingerprint':row['execution_fingerprint']},
            'initial_state':state,'fitting_identity':sd._identity(x,y,sample_indices),
            'settings':{'top_n':top_n,'nearby_radius_window':window,'side_offset_scaled':side_offset,
                        'max_diagnostic_evaluations':max_evaluations},
            'baseline_nll':evaluated[0],'saved_nll':row['train_loss'],
            'active_projected_gradient_inf':float(max(abs(reduced))),
            'active_complementarity':float(max(abs(lam*slack))),
            'active_projection_certificate':certificate,'active_indices':active.tolist(),
            'active_projected_direction':joint.tolist(),'raw_gradient':evaluated[1].tolist(),
            'particles':[]}
    cache={(vector.tobytes(),owners.tobytes()):(evaluated[0],evaluated[1],evaluator.trace[-1],True)}
    def probe(direction,step,fresh):
        point=vector+step*direction;model.set_vector(point)
        own=model.assign(tx) if fresh else assignment
        key=(point.tobytes(),own.numpy().tobytes())
        if key not in cache:
            if len(cache)>=max_evaluations:
                return {'step':step,'valid':False,'reason':'diagnostic_evaluation_cap','mean_nll':None}
            test=cj._Evaluation(model,tx,ty,own,1,[],fixed)
            loss,gradient,valid,error_text=None,None,False,None
            try:
                result=test.evaluate(point,derivatives=False)
                gradient=torch.autograd.grad(test.cached_loss,test.cached_parameter)[0].detach().numpy()
                loss=result[0];valid=np.isfinite(gradient).all() and min(result[2])>=-cj.DOMAIN_TOLERANCE
            except (cj.DomainError,RuntimeError) as error:
                error_text=f'{type(error).__name__}: {error}'
            guard=dict(test.trace[-1]) if test.trace else {}
            if error_text:guard['diagnostic_error']=error_text
            cache[key]=(loss,gradient,guard,bool(valid))
        loss,gradient,guard,valid=cache[key]
        return {'step':float(step),'valid':valid,'mean_nll':loss,
                'derivative':None if gradient is None else float(gradient@direction),
                'assignment_changes':int((own!=assignment).sum()),'assignment_identity':sd._identity(own.numpy()),
                'parameter_identity':sd._identity(point),'guard':guard}
    def bracket(direction,centre,offset):
        result={'direction':direction.tolist(),'crossing_step':float(centre),'actual_side_offset':float(offset),
                'assignment_modes':{}}
        locations=[centre+factor*offset for factor in (-1.,-.5,.5,1.)]
        for mode in ('frozen','fresh'):
            points=[probe(direction,step,mode=='fresh') for step in locations]
            result['assignment_modes'][mode]={'points':points,**slopes(points)}
        return result
    def first_coupled_event(direction):
        velocity=cj.GEOMETRY_SCALE*direction[radius_indices][owners]
        fractions=np.arange(1,model.layout.knots+1)/model.layout.knots
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            events=(nearest[:,None]/fractions[None,:]-radii[owners,None])/velocity[:,None]
        # The projected unit step certifies the linearized segment [0,1].
        # Actual nonlinear physical guards remain authoritative at every probe.
        positive=np.isfinite(events)&(events>0)&(events<1)
        if not positive.any():return None
        first=float(events[positive].min())
        max_velocity=float(max(abs(cj.GEOMETRY_SCALE*direction[radius_indices])))
        if first*max_velocity>window:return None
        rows,knots=np.where(events==first)
        others=events[np.isfinite(events)&(events!=first)]
        offset=min(side_offset/max(abs(direction)),first/4,(1-first)/4)
        if len(others):offset=min(offset,float(min(abs(others-first)))/4)
        return {'step':first,'offset':offset,'linearized_segment':[0.,1.],
                'event_scope':'frozen-owner radius knots; fresh assignment switches are additional possible events',
                'maximum_radius_move_to_event':first*max_velocity,
                'origins':[{'particle_index':int(owners[i]),'fitting_position':int(i),
                            'sample_index':int(sample_indices[i]),'knot_index':int(j+1)} for i,j in zip(rows,knots)]}
    for particle in ranking:
        coordinate=radius_indices[particle]
        scalar=np.zeros(len(vector));scalar[coordinate]=1.
        crossings=sd._radius_crossings(distances,owners,particle,radii[particle],model.layout.knots,sample_indices,1)
        item={'particle_index':int(particle),'candidate_id':model.candidate_ids[particle],
              'active_radius_component':float(joint[coordinate]),'raw_gradient':float(evaluated[1][coordinate]),
              'endpoint_radius':float(radii[particle]),'nearest_crossing':crossings[0] if crossings else None}
        scalar_valid=False
        if crossings and abs(crossings[0]['distance_from_endpoint'])<=window:
            crossing=crossings[0];centre=crossing['distance_from_endpoint']/cj.GEOMETRY_SCALE
            offset=side_offset
            if crossing['nearest_other_crossing_distance'] is not None:
                offset=min(offset,crossing['nearest_other_crossing_distance']/(4*cj.GEOMETRY_SCALE))
            item['scalar_probe']=bracket(scalar,centre,offset)
            scalar_valid=all(p['valid'] for p in item['scalar_probe']['assignment_modes']['fresh']['points'])
        else:item['scalar_probe']={'status':'no_crossing_within_declared_neighborhood'}
        if not scalar_valid and abs(joint[coordinate])>0:
            desired=np.zeros(k);desired[particle]=np.sign(joint[coordinate])
            coupled,_,cert=cj._project_direction(desired,matrix[:,radius_indices],slack)
            direction=np.zeros(len(vector));direction[radius_indices]=coupled
            event=first_coupled_event(direction) if np.max(abs(direction))>0 else None
            item['coupled_direction_projection']=cert
            if event:
                item['coupled_probe']={'event':event,**bracket(direction,event['step'],event['offset'])}
            else:item['coupled_probe']={'status':'no_positive_radius_event_in_unit_segment_and_neighborhood','direction':direction.tolist()}
        output['particles'].append(item)
    output['diagnostic_evaluations']={'outside_fitting_budget':True,'total':len(cache),
        'likelihood_evaluated':sum(bool(record[2] and record[2]['likelihood_evaluated']) for record in cache.values()),
        'valid':sum(record[3] for record in cache.values())}
    output['seconds']=time.perf_counter()-begun
    return cj._json_safe(output)


def analyze_blocks(stage,start):
    """Fresh actual-assignment certificate, separate from frozen-round fit status."""
    begun=time.perf_counter()
    report=json.loads(REPORT.read_text())
    if not report.get('all_diagnostic_stages_complete'):
        raise RuntimeError('main diagnostic experiment is incomplete; do not run probes concurrently')
    rows=[r for r in report['rows'] if r['case']=='observed' and r['stage']==stage and r['start']==start]
    if len(rows)!=1:raise ValueError('exactly one requested stage/start is required')
    row=rows[0];fit_path=ROOT/row['fit_artifact']
    if file_hash(fit_path)!=row['fit_sha256']:raise ValueError('fit artifact hash changed')
    fit=json.loads(fit_path.read_text());state=fit['starts'][0]['last_accepted_state']
    if state!=row['last_accepted_state']:raise ValueError('saved state mismatch')
    metadata=json.loads((CASE/'metadata.json').read_text())
    arrays,metadata=e21_simulation.load_case(CASE,metadata['case'],{'simulation':metadata['configuration'],**metadata['observation']})
    if metadata['data_identity']!=row['data_identity']['data_identity']:raise ValueError('case identity mismatch')
    mask=arrays['fitting'];x,y=arrays['x'][mask],arrays['y'][mask].astype(float)
    if e21_protocol.array_hash(x,y)!=row['data_identity']['fitting_identity']:
        raise ValueError('fitting observations differ from the staged fit')
    model=cj.restore_constrained_state(state)
    result={'schema':'actual-assignment-postrun-blocks-v1','stage':stage,'start':start,
            'scope':'posthoc endpoint diagnostic only; does not revise the recorded fit outcome',
            'saved_status':row['status'],'saved_objective_assignment_context':row.get('objective_assignment_context'),
            'saved_checkpoint_metadata':row['last_accepted_metadata'],'saved_mean_nll':row['train_loss'],
            'fit_centres':row['fit_centres'],'fit_radii':row['fit_radii'],'initial_state':state,
            'provenance':{'script_sha256':file_hash(__file__),'main_report_sha256':file_hash(REPORT),
                          'solver_source_sha256':file_hash(cj.__file__),'fit_path':str(fit_path),
                          'fit_sha256':file_hash(fit_path),'data_identity':row['data_identity'],
                          'execution_fingerprint':row['execution_fingerprint']},
            'diagnostic_evaluations':{'outside_fitting_budget':True,'stationarity':1,'value':1}}
    try:
        result['fresh_actual_assignment_blocks']=cj.projected_stationarity_diagnostics(
            model,x,y,fit_centres=row['fit_centres'],fit_radii=row['fit_radii'])
    except (cj.DomainError,RuntimeError) as error:
        result['fresh_actual_assignment_blocks']={'status':'failed','error':f'{type(error).__name__}: {error}'}
    tx,ty=torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64)
    evaluator=cj._Evaluation(model,tx,ty,model.assign(tx),1,[],cj._fixed_probes(model.box))
    try:
        evaluated=evaluator.evaluate(model.vector.detach().numpy(),derivatives=False)
        result['fresh_actual_assignment_mean_nll']=evaluated[0]
        result['mean_nll_change_from_saved_objective']=evaluated[0]-row['train_loss']
    except (cj.DomainError,RuntimeError) as error:
        result['value_error']=f'{type(error).__name__}: {error}'
    result['value_guard']=evaluator.trace[-1] if evaluator.trace else None
    result['seconds']=time.perf_counter()-begun
    return cj._json_safe(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='radii',choices=['chemistry','radii','centres','warm_radii','joint','warm_joint'])
    parser.add_argument('--blocks-only',action='store_true')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--top-n',type=int,default=3)
    parser.add_argument('--max-evaluations',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(3)
    if args.output.exists():raise FileExistsError('diagnostic output must be a fresh path')
    if not args.blocks_only and args.stage in ('chemistry','centres'):
        raise ValueError('stages with fixed radii require --blocks-only')
    result=(analyze_blocks(args.stage,args.start) if args.blocks_only else
            analyze(args.stage,args.start,top_n=args.top_n,max_evaluations=args.max_evaluations))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'stage':args.stage,'start':args.start,'active_pg':result.get('active_projected_gradient_inf'),
                      'seconds':result['seconds'],'evaluations':result['diagnostic_evaluations']}))
