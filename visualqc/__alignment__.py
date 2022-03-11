from sys import version_info

if version_info.major > 2:
    from visualqc import alignment
else:
    raise NotImplementedError('visualqc_alignment requires Python 3 or higher!')


def main():
    """Entry point."""

    alignment.cli_run()


if __name__ == '__main__':
    main()
