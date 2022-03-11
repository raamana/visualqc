from sys import version_info

if version_info.major > 2:
    from visualqc import functional_mri
else:
    raise NotImplementedError('visualqc_func_mri requires Python 3 or higher!')


def main():
    """Entry point."""

    functional_mri.cli_run()


if __name__ == '__main__':
    main()
