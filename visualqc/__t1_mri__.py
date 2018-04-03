from sys import version_info

if version_info.major > 2:
    from visualqc import t1_mri
else:
    raise NotImplementedError('visualqc_t1_mri requires Python 3 or higher!')

def main():
    "Entry point."

    t1_mri.cli_run()

if __name__ == '__main__':
    main()
