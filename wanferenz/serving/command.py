import argparse
import os


def parser_for(runtime):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="cmd", required=True)
    worker = commands.add_parser("stage", help="Run a DeepSeek-V4 layer partition")
    for option in ("stage", "nstages", "lo", "hi"):
        worker.add_argument("--" + option, type=int, required=True)
    worker.add_argument("--port", type=int, default=runtime.ENG_IN)
    worker.add_argument("--next", default=None)
    worker.add_argument(
        "--ret-relay",
        default=None,
        dest="ret_relay",
        help="Local return channel of the final GPU on this machine",
    )
    worker.add_argument(
        "--dspark", action="store_true", help="Enable DSpark on the final partition"
    )
    worker.add_argument("--dir", default=os.environ.get("V4_DIR", "/root/v4"))
    worker.add_argument("--device", default=None)
    worker.add_argument(
        "--bind", default=os.environ.get("WANFERENZ_ENGINE_BIND", "127.0.0.1")
    )
    worker.add_argument("--receipts", action="store_true")
    coordinator = commands.add_parser(
        "coord", help="Read generation requests as JSON lines"
    )
    coordinator.add_argument("--head", default=f"127.0.0.1:{runtime.ENG_IN}")
    coordinator.add_argument("--tail", default=f"127.0.0.1:{runtime.FWD_RET}")
    coordinator.add_argument("--dir", default=os.environ.get("V4_DIR", "/root/v4"))
    coordinator.add_argument("--receipts", action="store_true")
    coordinator.add_argument("--timeout", type=int, default=600)
    coordinator.add_argument(
        "--connect-retry", type=int, default=300, dest="connect_retry"
    )
    commands.add_parser("selftest", help="Check a CPU chain against the model oracle")
    commands.add_parser(
        "selftest-relay", help="Check a CPU chain with a local return relay"
    )
    return parser


def execute(runtime):
    options = parser_for(runtime).parse_args()
    match options.cmd:
        case "stage":
            runtime.run_partition(
                options.stage,
                options.nstages,
                options.lo,
                options.hi,
                options.port,
                options.next,
                ckpt_dir=options.dir,
                device=options.device,
                receipts=options.receipts or runtime.RECEIPTS,
                bind=options.bind,
                ret_relay=options.ret_relay,
                dspark=options.dspark,
            )
        case "coord":
            raise SystemExit(runtime._serve_job_stream(options))
        case "selftest":
            runtime.selftest()
        case "selftest-relay":
            runtime.selftest(nstages=3, tail_box_g=2)
