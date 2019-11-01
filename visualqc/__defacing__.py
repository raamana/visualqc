from sys import version_info

if version_info.major > 2:
    from visualqc import defacing
else:
    raise NotImplementedError('visualqc_defacing requires Python 3 or higher!')

def main():
    "Entry point."

    defacing.cli_run()

if __name__ == '__main__':
    main()
