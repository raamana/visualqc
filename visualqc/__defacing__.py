from sys import version_info

if version_info.major > 2:
    from visualqc import defacing
    from warnings import catch_warnings, filterwarnings
else:
    raise NotImplementedError('visualqc_defacing requires Python 3 or higher!')

def main():
    "Entry point."

    # disabling all not severe warnings
    with catch_warnings():
        filterwarnings("ignore", category=DeprecationWarning)
        filterwarnings("ignore", category=FutureWarning)

        defacing.cli_run()

if __name__ == '__main__':
    main()
