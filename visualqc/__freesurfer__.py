from sys import version_info

if version_info.major > 2:
    from visualqc import freesurfer
else:
    raise NotImplementedError('visualqc_freesurfer requires Python 3 or higher!')

def main():
    "Entry point."

    freesurfer.cli_run()

if __name__ == '__main__':
    main()
