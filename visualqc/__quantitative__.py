from sys import version_info

if version_info.major > 2:
    from visualqc import quantitative
else:
    raise NotImplementedError('visualqc_quantitative requires Python 3 or higher!')

def main():
    "Entry point."

    quantitative.cli_run()

if __name__ == '__main__':
    main()
