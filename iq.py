import logging
import os
import argparse
from argparse import ArgumentParser
from icotq_store import IcoTqStore, TqSource
from typing import cast

def iq_info(its: IcoTqStore, logger:logging.Logger) -> None:
    its.list_sources()

def iq_export(its: IcoTqStore, logger:logging.Logger) -> None:
    if 'ebook_mirror' not in its.config:
        logger.error(f"Cannot export, destination 'ebook_mirror' not defined in config")
        return
    ebook_mirror_path = os.path.expanduser(its.config['ebook_mirror'])
    if os.path.isdir(ebook_mirror_path) is False:
        logger.error(f"Destination directory {ebook_mirror_path} does not exist, aborting export!")
        return
    print(f"Export to {ebook_mirror_path}")

def iq_import(its: IcoTqStore, logger:logging.Logger):
    its.import_texts()

def iq_help(parser:argparse.ArgumentParser, valid_actions:list[str]):
    parser.print_help()
    print()
    print("Command can either be provided as command-line arguments or at the '> ' prompt.")
    print("Valid commands are: " + ', '.join(valid_actions))
    print("To exit, simply press Enter at the command prompt, or by 'exit' or 'quit'")

def parse_cmd(its: IcoTqStore, logger: logging.Logger) -> None:
    valid_actions = ['info', 'export', 'import', 'help']
    parser: ArgumentParser = argparse.ArgumentParser(description="IcoTq")
    _ = parser.add_argument(
        "action",
        nargs="*",
        default="",
        help="Action: " + ', '.join(valid_actions),
    )
    _ = parser.add_argument(
            "-n",
            "--non-interactive",
            action="store_true",
            help="Non-interactive mode, do not enter repl",
        )        
    args = parser.parse_args()
    quit:bool = False
    while quit is False:
        actions: list[str] =  cast(list[str], args.action)
        for action in actions:
            if action not in valid_actions:
                logger.error(f"Invalid action {action}, valid are: {valid_actions}")
                exit(1)
        if 'info' in actions:
            iq_info(its, logger)
        if 'export' in actions:
            iq_export(its, logger)
        if 'import' in actions:
            iq_import(its, logger)
        if 'help' in actions:
            iq_help(parser, valid_actions)
        if cast(bool, args.non_interactive) is True:
            break
        try:
            cmd = input("> ")
        except (EOFError, KeyboardInterrupt):
            quit = True
            continue
        print(f"{len(cmd)}: >{cmd}<")
        cmd = cmd.strip().replace('  ', ' ')
        if cmd == "" or cmd == 'quit' or cmd == 'exit': 
            quit = True
        else:
            args = parser.parse_args(cmd.split(' '))
    print()

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IQ")
    logger.info("Starting...")
     
    its = IcoTqStore()
    parse_cmd(its, logger)

if __name__ == "__main__":
    main()