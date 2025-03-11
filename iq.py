import logging
import argparse
from argparse import ArgumentParser
from icotq_store import IcoTqStore
from typing import cast

def iq_info(its: IcoTqStore) -> None:
    its.list_sources()

def parse_cmd(its: IcoTqStore, logger: logging.Logger) -> None:
    valid_actions = ['info', 'export']
    parser: ArgumentParser = argparse.ArgumentParser(description="IcoTq")
    _ = parser.add_argument(
        "action",
        nargs="*",
        default="",
        help="Action: " + ', '.join(valid_actions),
    )
    args = parser.parse_args()
    actions: list[str] =  cast(list[str], args.action)
    for action in actions:
        if action not in valid_actions:
            logger.error(f"Invalid action {action}, valid are: {valid_actions}")
            exit(1)
    if 'info' in actions:
        iq_info(its)

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
     
    its = IcoTqStore()
    parse_cmd(its, logger)

if __name__ == "__main__":
    main()