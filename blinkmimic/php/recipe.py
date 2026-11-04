"""Pinned upstream PRESET[main], with explicit PHP/hardware overrides."""
import shlex

UPSTREAM_MAIN = shlex.split('''
--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more
--lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3
--contact_reward 0.25 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon
--pi_ref_skip --wm_only_iters 0 --bin_fix --ref_contact_fix --ckpt_minutes 0
--cuda_graph --eprobe_share 0.15 --graph_early --collect_graph --reset_masked
--pi_batch 2048 --pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300
--iterations 3500 --ckpt_every 100 --ckpt_light --eprobe_graph --eprobe_async 20
--eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7 --lr_decay_by 1400
''')


def main():
    import os
    import sys
    from pathlib import Path
    from datetime import datetime
    from zoneinfo import ZoneInfo
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    # The author explicitly recommends synchronous probes on a shared GPU.
    # Algorithm/probe distribution is retained; only stream overlap is disabled.
    args = UPSTREAM_MAIN + ['--robot','php','--num_envs','2048','--buffer_steps','512',
        '--pi_microbatch','512','--eprobe_batch','64',
        '--eprobe_async','0','--log_dir',str(Path('logs')/f'blinkmimic_php_{stamp}')]
    os.execv(sys.executable, [sys.executable,'-u','-m','blinkmimic.train',*args,*sys.argv[1:]])
