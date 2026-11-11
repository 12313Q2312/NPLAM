import argparse
import json
from pathlib import Path
import yaml
from .experiment import run, summarize
from .real_data import download, SOURCES


def main():
    parser=argparse.ArgumentParser(description="NPLAM audited research benchmark")
    subs=parser.add_subparsers(dest="action",required=True)
    train=subs.add_parser("run")
    train.add_argument("--config",required=True)
    train.add_argument("--output",required=True)
    train.add_argument("--cache",default="data")
    train.add_argument("--reuse-from",help="reuse verified complete runs with identical method, packages and config")
    fetch=subs.add_parser("fetch")
    fetch.add_argument("dataset",choices=list(SOURCES))
    fetch.add_argument("--cache",default="data")
    summary=subs.add_parser("summarize")
    summary.add_argument("output")
    args=parser.parse_args()
    if args.action=="fetch":
        print(download(args.dataset,args.cache))
    elif args.action=="summarize":
        summarize(args.output)
    else:
        config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not config["seeds"] or len(set(config["seeds"]))!=len(config["seeds"]):
            raise ValueError("seeds must be nonempty and distinct")
        run(config,args.output,args.cache,reuse_from=args.reuse_from)


if __name__=="__main__":
    main()
