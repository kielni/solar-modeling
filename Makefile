lint:
	black *.py
	flake8 *.py

run:
	black *.py
	python main.py solar-10-battery-30.yml
	cd solar-10-battery-30 ; montage solar_hourly*.png -tile 3x4 -geometry +2+2 solar_monthly.png ; cd ..
	# python main.py --solar 12 --battery 30
	# cd solar-12-battery-30; montage solar_hourly*.png -tile 3x4 -geometry +2+2 solar_monthly.png ; cd ..

run-all:
	black *.py
	python main.py solar-10-battery-30.yml
	python main.py solar-10-battery-45.yml
	python main.py solar-8-battery-30.yml
	python main.py solar-10-battery-30-actual.yml
	python main.py solar-8-battery-30-rate-8.yml
	python main.py arbitrage-limit-10.yml
	python main.py arbitrage-limit-20.yml
	python main.py arbitrage-limit-8.yml
	python main.py arbitrage-max.yml
	python summary.py
	python summary.py arbitrage
 