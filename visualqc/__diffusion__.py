from sys import version_info

if version_info.major > 2:
    from visualqc import diffusion
else:
    raise NotImplementedError('visualqc_diffusion requires Python 3 or higher!')

def main():
    "Entry point."

    diffusion.cli_run()

if __name__ == '__main__':
    main()
