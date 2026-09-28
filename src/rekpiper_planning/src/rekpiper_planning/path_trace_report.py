"""Export inspectable cost and joint trends alongside the raw evidence."""

import csv
from pathlib import Path

import numpy as np


def read_costs(directory):
    with (Path(directory) / 'costs.csv').open(newline='', encoding='utf-8') as stream:
        return [{key: float(value) if value else np.nan for key, value in row.items()}
                for row in csv.DictReader(stream)]


def cost_plot(directory, reset_in_total):
    from matplotlib.figure import Figure

    rows = read_costs(directory)
    if not rows:
        return rows
    figure = Figure(figsize=(10, 7), tight_layout=True)
    total, components = figure.subplots(2, 1, sharex=True)
    x = [row['evaluation'] for row in rows]
    for key in ('total_cost', 'best_cost'):
        total.plot(x, [row[key] for row in rows], label=key)
    for key in rows[0]:
        if not key.endswith('_cost') or key in ('total_cost', 'best_cost'):
            continue
        label = key + (' (diagnostic only)' if key == 'reset_reg_cost' and not reset_in_total else '')
        components.plot(x, [row[key] for row in rows], label=label)
    for axis in (total, components):
        axis.set_ylabel('Weighted cost')
        axis.grid(True, alpha=.25)
        axis.legend(fontsize=8)
    total.set_title('PathSolver objective evaluations (includes rejected trials and final debug)')
    components.set_xlabel('Objective evaluation; not optimizer iteration')
    figure.savefig(str(Path(directory) / 'cost_trends.svg'))
    return rows


def joint_plot(directory, report):
    from matplotlib.figure import Figure

    directory = Path(directory)
    joints = np.asarray(report['joint_positions'])
    delta = np.asarray(report['joint_delta_rad'])
    names = report['joint_names']
    with (directory / 'joints.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample'] + [name + '_rad' for name in names] +
                        [name + '_delta_rad' for name in names])
        for index, (q, change) in enumerate(zip(joints, delta)):
            writer.writerow([index] + q.tolist() + change.tolist())
    figure = Figure(figsize=(10, 7), tight_layout=True)
    angles, changes = figure.subplots(2, 1, sharex=True)
    for index, name in enumerate(names):
        angles.plot(np.rad2deg(joints[:, index]), label=name)
        changes.plot(np.rad2deg(delta[:, index]), label=name)
    for axis in (angles, changes):
        axis.set_ylabel('Degrees')
        axis.legend(ncol=3)
        axis.grid(True, alpha=.25)
    angles.set_title('IK preview: {} / {} (no collision validation)'.format(
        report['preview_kind'], 'PASS' if report['valid'] else 'FAILED; prefix only'))
    changes.set_ylabel('Change from live seed (deg)')
    changes.set_xlabel('IK sample; no physical time parameterization')
    figure.savefig(str(directory / 'joint_trends.svg'))
