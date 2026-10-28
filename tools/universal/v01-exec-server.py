import os, sys
env = os.environ.copy()
env['LD_PRELOAD'] = env.pop('L0XRE_VISION_PRELOAD')
os.execvpe(sys.argv[1], sys.argv[1:], env)
