# Exoplanet Detection API Makefile

.PHONY: help install run test clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make run      - Start the API server"
	@echo "  make test     - Test with HAT-P-7 target"
	@echo "  make clean    - Clean temporary files"

install:
	@echo "Installing API dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

run:
	@echo "Starting Exoplanet Detection API..."
	@echo "API will be available at: http://localhost:5000"
	@echo "Press Ctrl+C to stop"
	python app.py

test:
	@echo "Testing API with HAT-P-7 target..."
	@echo "This may take a few minutes to download and process the light curve..."
	curl -X POST -F "label=HAT-P-7" http://localhost:5000/look-for-exoplanet

test-health:
	@echo "Testing health endpoint..."
	curl -s http://localhost:5000/health | python -m json.tool

clean:
	@echo "Cleaning temporary files..."
	rm -rf temp/*
	@echo "Cleanup complete!"

# Development commands
dev-install: install
	@echo "Development setup complete!"

dev-run:
	@echo "Starting API in development mode..."
	FLASK_ENV=development python app.py
