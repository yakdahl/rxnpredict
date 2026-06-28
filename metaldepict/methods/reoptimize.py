#!/usr/bin/env python3
"""Re-optimize every workflow physics method against the IMPROVED (crowd-aware)
numerical judge, render each, and write results to /tmp/reopt_results.json."""
import sys, importlib.util, json, time
sys.path.insert(0, '.')
import relax_harness as R

METHODS = ['energy', 'minkin', 'rigidtorque', 'forcedir', 'pbd']
MAXITER = 12


def load(name):
    spec = importlib.util.spec_from_file_location(name, f'/tmp/variants/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    out = {}
    for name in METHODS:
        t = time.time()
        try:
            mod = load(name)
            params, loss, per = R.optimize_method(
                mod.relax, mod.BOUNDS, mod.PARAM_NAMES,
                optimizer='de', maxiter=MAXITER)
            # add per-ligand quality_loss
            for key in R.LIGANDS:
                H = R.Harness(key)
                mod.relax(H, params)
                per[key]['loss'] = round(R.quality_loss(H), 4)
            R.make_panel(mod.relax, params, f'/tmp/reopt_{name}_panel.png',
                         tag=f'reopt_{name}', steps_note='crowd-aware judge')
            out[name] = {'mean_loss': round(loss, 4), 'best_params': params,
                         'per_ligand': per, 'panel': f'/tmp/reopt_{name}_panel.png',
                         'secs': round(time.time() - t, 1)}
            print(f'{name:12s} mean_loss={loss:.4f}  ({time.time()-t:.0f}s)', flush=True)
        except Exception as e:
            out[name] = {'error': str(e)}
            print(f'{name:12s} ERROR {e}', flush=True)
        json.dump(out, open('/tmp/reopt_results.json', 'w'), indent=2)
    ranked = sorted([k for k in out if 'mean_loss' in out[k]],
                    key=lambda k: out[k]['mean_loss'])
    print('\nRANKING (crowd-aware judge):')
    for i, k in enumerate(ranked):
        print(f'  {i+1}. {k:12s} {out[k]["mean_loss"]:.4f}')
    out['_ranking'] = ranked
    json.dump(out, open('/tmp/reopt_results.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
