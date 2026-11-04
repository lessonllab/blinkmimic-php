"""CLI wrapper for evaluating the PHP BlinkMimic checkpoint format."""
def main():
    import runpy
    import sys
    from pathlib import Path
    args = sys.argv[1:]
    aliases = {'--checkpoint': '--ckpt', '--dataset-dir': '--motion_dir'}
    args = [aliases.get(value, value) for value in args]
    if '--run_dir' not in args and '--ckpt' in args:
        args += ['--run_dir', str(Path(args[args.index('--ckpt')+1]).parent)]
    sys.argv = ['blinkmimic.evaluate', *args]
    runpy.run_module('blinkmimic.evaluate', run_name='__main__')


if __name__ == '__main__':
    main()
