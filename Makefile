flake8:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 --ignore N802,N806,E501,F841,F401 `find . -name \*.py | grep -v setup.py | grep -v /doc/ | grep -v __init__`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
	py.test --pyargs nikamap --cov-report term-missing --cov=nikamap --mpl


png:
	py.test --mpl-generate-path=nikamap/tests/baseline
