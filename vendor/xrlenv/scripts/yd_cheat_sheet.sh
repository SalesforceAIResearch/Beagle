# on the login node — exposes control-plane:8080 at login-node:8080 (shared)
ssh -fN -L 8080:localhost:8080 node-host
ssh node-host
# ssh -L 8080:localhost:8080 node-host


xrlenv build apply --plan /path/to/xrlenv/xrlenv_plugins/images_build/terminal_bench_2/build_plan_89_full.calibrated.yaml --connect-host 127.0.0.1 --fill-missing
xrlenv build apply --plan /path/to/xrlenv/xrlenv_plugins/images_build/swebench_verified/build_plan_500_full.calibrated.yaml --connect-host 127.0.0.1 --fill-missing --concurrency 96
xrlenv build apply --plan /path/to/xrlenv/xrlenv_plugins/images_build/swebench_pro/build_plan_731_full.partial_calibrated.yaml --connect-host 127.0.0.1 --fill-missing --concurrency 32
xrlenv build calibrate \
    --plan /path/to/xrlenv/xrlenv_plugins/images_build/swebench_pro/build_plan_731_full.partial_calibrated.yaml \
    --output /path/to/xrlenv/xrlenv_plugins/images_build/swebench_pro/build_plan_731_full.partial_calibrated.yaml \
    --connect-host 127.0.0.1

xrlenv build cancel --plan 36bd7fad19ed --connect-host 127.0.0.1
xrlenv build cancel --plan dc69aa04ce9a --connect-host 127.0.0.1
xrlenv build cancel --plan 06b4f4a6d305 --connect-host 127.0.0.1

python -m xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen \
    --all --max-workers 8 --output build_plan_89_full.yaml


sinfo -p your-slurm-partition -N -h -o '%N|%T|%E'




python3 scripts/warm_images.py /path/to/xrlenv/xrlenv_plugins/images_build/swebench_pro/build_plan_731_full.partial_calibrated.yaml --concurrency 32
python3 scripts/warm_images.py /path/to/xrlenv/xrlenv_plugins/images_build/swebench_verified/build_plan_500_full.calibrated.yaml --concurrency 32
python3 scripts/warm_images.py /path/to/xrlenv/xrlenv_plugins/images_build/terminal_bench_2/build_plan_89_full.calibrated.yaml --concurrency 32

 tmux new -d -s metrics 'cd /path/to/xrlenv && bash scripts/eval_metrics_sampler.sh --interval 60 --csv eval_metrics.csv'