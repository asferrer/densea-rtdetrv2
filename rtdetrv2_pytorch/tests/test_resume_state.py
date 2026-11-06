"""DetSolver resume: best_stat and patience are rebuilt from log.txt.
CPU only. Run: python -m pytest tests/test_resume_state.py (from rtdetrv2_pytorch)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.solver.det_solver import DetSolver  # noqa: E402

PATIENCE, MIN_DELTA = 20, 0.001


def make_solver(out):
    s = object.__new__(DetSolver)  # no cfg: only the selection helpers are used
    s.output_dir = Path(out)
    s.early_stopping_patience = PATIENCE
    s.early_stopping_min_delta = MIN_DELTA
    s.early_stopping_metric = 'coco_eval_bbox'
    s.epochs_without_improvement = 0
    s.best_metric_value = -float('inf')
    return s


def ap(epoch):
    # best at epoch 9 (10th); only gets worse afterwards
    return 0.30 + 0.01 * epoch if epoch < 10 else 0.39 - 0.001 * (epoch - 9)


def write_log(out, n, dup=False):
    lines = [json.dumps({'train_loss': 1.0, 'test_coco_eval_bbox': [ap(e), 0.5],
                         'epoch': e}) for e in range(n)]
    if dup:
        lines.append(lines[-1])
    (Path(out) / 'log.txt').write_text('\n'.join(lines) + '\n')


def step(s, best_stat, epoch, saved):
    """One selection + stopping step, same as fit(); returns True if it stops."""
    ts = {'coco_eval_bbox': [ap(epoch), 0.5]}
    for k in ts:
        s._update_best_stat(best_stat, k, ts[k][0], epoch)
        if best_stat['epoch'] == epoch:
            saved.append(epoch)
    s._update_early_stopping(ts)
    return s.epochs_without_improvement >= PATIENCE


def resumed(tmp_path, n=25, dup=False):
    write_log(tmp_path, n, dup)
    s, best = make_solver(tmp_path), {'epoch': -1}
    s._restore_selection_state(best, n)
    return s, best


def test_counter_and_best_after_resume(tmp_path):
    s, best = resumed(tmp_path)
    assert best['epoch'] == 9
    assert s.best_metric_value == ap(9)
    assert s.epochs_without_improvement == 15


def test_best_not_overwritten_by_worse_epoch(tmp_path):
    s, best = resumed(tmp_path)
    saved = []
    step(s, best, 25, saved)
    assert saved == [] and best['epoch'] == 9


def test_stops_after_5_epochs(tmp_path):
    s, best = resumed(tmp_path)
    saved = []
    stops = [step(s, best, e, saved) for e in range(25, 30)]
    assert stops == [False] * 4 + [True]


def test_equivalent_to_uninterrupted_run(tmp_path):
    s0, b0, saved0 = make_solver(tmp_path), {'epoch': -1}, []
    for e in range(25):
        step(s0, b0, e, saved0)
    s1, b1 = resumed(tmp_path)
    assert (s1.epochs_without_improvement, s1.best_metric_value, b1) == \
           (s0.epochs_without_improvement, s0.best_metric_value, b0)


def test_ignores_later_and_duplicate_epochs(tmp_path):
    write_log(tmp_path, 25, dup=True)
    s, best = make_solver(tmp_path), {'epoch': -1}
    s._restore_selection_state(best, 20)  # resume at epoch 20: only epochs 0..19 count
    assert best['epoch'] == 9 and s.epochs_without_improvement == 10


def test_no_log_starts_from_zero(tmp_path):
    s, best = make_solver(tmp_path), {'epoch': -1}
    s._restore_selection_state(best, 5)
    assert best == {'epoch': -1} and s.epochs_without_improvement == 0


def test_min_delta_best_and_patience_diverge_as_in_fit(tmp_path):
    # 0.3905 improves best_stat (>) but not patience (< best + min_delta)
    vals = [0.30, 0.39, 0.3905]
    lines = [json.dumps({'test_coco_eval_bbox': [v], 'epoch': e}) for e, v in enumerate(vals)]
    (tmp_path / 'log.txt').write_text(chr(10).join(lines) + chr(10))
    s, best = make_solver(tmp_path), {'epoch': -1}
    s._restore_selection_state(best, 3)
    assert best['epoch'] == 2 and s.best_metric_value == 0.39
    assert s.epochs_without_improvement == 1


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def step(self):
        pass

    def finished(self):
        return True

    def set_epoch(self, e):
        pass


def run_fit(tmp_path, start_epoch, until):
    """Real fit() with mocked train/evaluate; returns (epochs, saved, solver)."""
    import torch
    import src.solver.det_solver as ds
    ran, saved = [], []
    ds.train_one_epoch = lambda *a, **k: {'loss': 1.0}
    cur = {}

    def fake_eval(*a, **k):
        e = cur['e']; ran.append(e)
        return {'coco_eval_bbox': [ap(e), 0.5]}, None
    ds.evaluate = fake_eval
    ds.dist_utils.save_checkpoint_fast = lambda state, path: saved.append((cur['e'], Path(path).name))
    s = make_solver(tmp_path)
    s.cfg = _Stub(epoches=until, clip_max_norm=0.1, print_freq=1, checkpoint_freq=10**6)
    s.train = lambda: None
    s.model = torch.nn.Linear(1, 1)
    s.train_dataloader = _Stub()
    s.val_dataloader = s.evaluator = s.postprocessor = s.criterion = s.device = None
    s.optimizer = _Stub(param_groups=[])
    s.lr_scheduler = _Stub(eta_min=0.0)
    s.lr_warmup_scheduler = s.ema = s.scaler = s.writer = None
    s.enable_per_class_metrics = False
    s.last_epoch = start_epoch - 1
    s.state_dict = lambda: {}
    # the current epoch is set by train_one_epoch (called right before evaluate)
    ds.train_one_epoch = lambda m, c, dl, o, d, epoch, **k: cur.update(e=epoch) or {'loss': 1.0}
    s.fit()
    return ran, saved, s


def test_resumed_fit_keeps_best_and_stops_at_5(tmp_path):
    write_log(tmp_path, 25)
    ran, saved, s = run_fit(tmp_path, 25, 200)
    assert ran == [25, 26, 27, 28, 29]  # 20 epochs without improvement in total: stops at epoch 29
    assert [x for x in saved if x[1] == 'best.pth'] == []


def test_uninterrupted_fit_stops_at_the_same_epoch(tmp_path):
    ran, saved, s = run_fit(tmp_path, 0, 200)
    assert ran[-1] == 29
    assert [e for e, n in saved if n == 'best.pth'][-1] == 9


if __name__ == '__main__':  # without pytest: python tests/test_resume_state.py
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory() as d:
                try:
                    fn(Path(d))
                    print('OK', name)
                except Exception as exc:  # noqa: BLE001
                    print('FALLA', name, type(exc).__name__, str(exc)[:80])
