unit-tests:
	python -m pytest --cov=lorax tests

install:
	pip install pip --upgrade
	pip install -e .

release:
	pip install poetry
	poetry build
	poetry publish