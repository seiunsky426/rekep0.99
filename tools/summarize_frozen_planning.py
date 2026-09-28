#!/usr/bin/env python3
"""Summarize measured preview phases without double-counting nested solvers."""
import argparse
import json
from pathlib import Path


def summarize(session):
    session = Path(session)
    report_path = session/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    files = [session/'timings.jsonl', session/'grasp_timings.jsonl']
    files.extend(sorted(session.glob('candidate*_stage*_attempt*/solver_timings.jsonl')))
    phases = []
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'elapsed_s' not in row:
                continue
            phases.append(dict(source=str(path.relative_to(session)), **row))
    # Child SDK and optimizer measurements are breakdowns of parent stages.
    top = [r for r in phases if r['source']=='timings.jsonl'
           and r['stage']!='grasp_pose_to_final_path']
    return dict(session=str(session.resolve()),
        snapshot_wall_s=(report.get('frozen_scene') or {}).get('snapshot_wall_s'),
        requested_stages_passed=report.get('requested_stages_passed', False),
        all_stages_passed=report.get('all_stages_passed', False),
        grasp_pose_to_final_path_s=report.get('grasp_pose_to_final_path_s'),
        measured_wait_s=sum(r['elapsed_s'] for r in top if r['kind']=='wait'),
        measured_work_s=sum(r['elapsed_s'] for r in top if r['kind']=='compute'),
        timing_definition='monotonic wall time; nested rows are breakdowns, not additive',
        stages=report.get('stages', []), phases=phases,
        motion_allowed=False, hardware_commands_sent=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    args = parser.parse_args()
    result = summarize(args.session)
    target = Path(args.session)/'timing_summary.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for row in result['phases']:
        print('{:7s} {:32s} {:10.3f}s {}'.format(
            row['kind'], row['stage'], row['elapsed_s'], row['event']))
    print('Full three-stage preview passed:', result['all_stages_passed'])
    print('Report:', target)


if __name__ == '__main__':
    main()
