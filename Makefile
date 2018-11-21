flake8:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 --ignore N802,N806,E501,F841,F401,W504 `find . -name \*.py | grep -v setup.py | grep -v /doc/ | grep -v __init__ | grep -v .eggs`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
	py.test-3 --pyargs nikamap --cov-report term-missing --cov=nikamap --mpl -x

test-pdb:
	py.test-3 --pyargs nikamap --cov-report term-missing --cov=nikamap --mpl -x --pdb --pdbcls=IPython.terminal.debugger:Pdb

png:
	py.test-3 --mpl-generate-path=nikamap/tests/baseline
