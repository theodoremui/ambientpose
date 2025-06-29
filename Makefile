# AlphaDetect - Comprehensive Makefile
# This Makefile provides targets for all aspects of the AlphaDetect project lifecycle

# ===== CONFIGURATION VARIABLES =====
SHELL := /bin/bash
.SHELLFLAGS := -ec
.ONESHELL:

# Python / uv settings
UV := uv                     # uv binary (https://github.com/astral-sh/uv)
VENV := .venv                # created by `uv venv`
UV_RUN := $(UV) run          # helper to run cmds inside the venv

PYTEST := $(UV_RUN) pytest
COVERAGE := $(UV_RUN) coverage
BLACK := $(UV_RUN) black
ISORT := $(UV_RUN) isort
RUFF := $(UV_RUN) ruff
MYPY := $(UV_RUN) mypy
PRE_COMMIT := $(UV_RUN) pre-commit

# Node/Frontend settings
NODE_MODULES := frontend/node_modules
NPM := npm

# Docker settings
DOCKER_COMPOSE := docker compose
DOCKER_COMPOSE_FILE := docker-compose.yml

# Project directories
CLI_DIR := cli
SERVER_DIR := server
FRONTEND_DIR := frontend
TEST_DIR := tests
DOCS_DIR := docs
OUTPUT_DIR := outputs
UPLOAD_DIR := uploads
MODEL_DIR := model_files

# AlphaPose model URLs
YOLOX_MODEL_URL := https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.5.0/yolox_x.pth
POSE_MODEL_URL := https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.5.0/fast_res50_256x192.pth
SMPL_MODEL_URL := https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

# Database settings
DB_FILE := alphadetect.db
MIGRATIONS_DIR := $(SERVER_DIR)/migrations

# ===== PHONY TARGETS =====
.PHONY: help all setup install install-dev install-frontend download-models
.PHONY: run run-cli run-server run-frontend run-docker
.PHONY: test test-cli test-server test-integration test-coverage
.PHONY: format lint typecheck
.PHONY: docs docs-serve docs-build docs-deploy
.PHONY: docker-build docker-up docker-down docker-logs docker-gpu
.PHONY: deploy deploy-server deploy-frontend
.PHONY: db-init db-migrate db-upgrade db-downgrade
.PHONY: clean clean-pyc clean-test clean-build clean-outputs clean-all
.PHONY: compile-requirements install-alphapose install-pose-alternatives fix-alphapose-build

# ===== DEFAULT TARGET =====
.DEFAULT_GOAL := help

# ===== HELP TARGET =====
help:
	@echo "AlphaDetect Makefile Help"
	@echo "=========================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup            - Create virtual environment and install dependencies"
	@echo "  make install          - Install Python dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-frontend - Install frontend dependencies"
	@echo "  make download-models  - Download AlphaPose model files"
	@echo "  make compile-requirements - Generate requirements.txt lock file from pyproject.toml"
	@echo "  make install-alphapose      - Install AlphaPose (with build fixes)"
	@echo "  make install-pose-alternatives - Install MediaPipe and other pose detection alternatives"
	@echo "  make fix-alphapose-build    - Troubleshoot AlphaPose installation issues"
	@echo ""
	@echo "Development:"
	@echo "  make run              - Run all components (server and frontend)"
	@echo "  make run-cli          - Run CLI with test image (auto-detects backend)"
	@echo "  make run-server       - Run FastAPI server"
	@echo "  make run-frontend     - Run Next.js frontend"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-cli         - Run CLI tests"
	@echo "  make test-server      - Run server tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-coverage    - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           - Format code with black and isort"
	@echo "  make lint             - Lint code with ruff"
	@echo "  make typecheck        - Type check with mypy"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             - Build documentation"
	@echo "  make docs-serve       - Serve documentation locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build all Docker images"
	@echo "  make docker-up        - Start all Docker containers"
	@echo "  make docker-down      - Stop all Docker containers"
	@echo "  make docker-logs      - View Docker container logs"
	@echo "  make docker-gpu       - Start Docker containers with GPU support"
	@echo ""
	@echo "Database:"
	@echo "  make db-init          - Initialize database"
	@echo "  make db-migrate       - Create a new migration"
	@echo "  make db-upgrade       - Upgrade database to latest migration"
	@echo "  make db-downgrade     - Downgrade database to previous migration"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove Python cache files"
	@echo "  make clean-pyc        - Remove Python cache files"
	@echo "  make clean-test       - Remove test artifacts"
	@echo "  make clean-build      - Remove build artifacts"
	@echo "  make clean-outputs    - Remove output files"
	@echo "  make clean-all        - Remove all generated files"
	@echo ""
	@echo "NOTE: 'pyproject.toml' is the SINGLE source of truth for dependencies."
	@echo "      'requirements.txt' is auto-generated via make compile-requirements."
# ===== SETUP AND INSTALLATION TARGETS =====
# -------------------------------------------------------------------
# SETUP AND INSTALLATION TARGETS (powered by uv)
# -------------------------------------------------------------------

setup: $(VENV) install-dev install-frontend download-models compile-requirements
# create .venv only once
$(VENV):
	@echo "Creating virtual environment with uv..."
	$(UV) venv
	@echo "Virtual environment created at $(VENV)"

# core deps
install: $(VENV)
	@echo "Installing Python dependencies with uv..."
	$(UV) pip install -e .
	@echo "Dependencies installed successfully!"

# dev / test / docs extras
install-dev: install
	@echo "Installing development dependencies..."
	$(UV) pip install -e ".[dev,test,docs]"
	$(PRE_COMMIT) install
	@echo "Development dependencies installed successfully!"

install-frontend: $(NODE_MODULES)
	@echo "Frontend dependencies installed successfully!"

$(NODE_MODULES):
	@echo "Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && $(NPM) install

# -------------------------------------------------------------------
# Dependency lock generation
# -------------------------------------------------------------------
compile-requirements: $(VENV)
	@echo "Compiling requirements.txt lock file from pyproject.toml ..."
	@command -v $(UV) >/dev/null 2>&1 || { echo >&2 '❌ uv not found. Install uv and try again.' ; exit 1; }
	$(UV) pip compile pyproject.toml -o requirements.txt --all-extras
	@echo "requirements.txt updated successfully!"

# -------------------------------------------------------------------
# Pose Detection Installation Alternatives
# -------------------------------------------------------------------

# Install MediaPipe as AlphaPose alternative (recommended for macOS)
install-pose-alternatives: install
	@echo "Installing pose detection alternatives..."
	@echo "Installing MediaPipe (cross-platform, easier installation)..."
	$(UV) pip install mediapipe
	@echo "Installing OpenPose alternative packages..."
	$(UV) pip install ultralytics
	@echo "Pose detection alternatives installed successfully!"
	@echo ""
	@echo "Available alternatives:"
	@echo "  - MediaPipe: Cross-platform, works on CPU, good performance"
	@echo "  - Ultralytics YOLO: Modern object detection with pose estimation"
	@echo "  - Use these instead of AlphaPose if installation fails"

# Fix AlphaPose build issues (installs build dependencies first)
fix-alphapose-build: install
	@echo "Attempting to fix AlphaPose build issues..."
	@echo "Installing build dependencies..."
	$(UV) pip install numpy cython setuptools wheel
	@echo "Installing additional dependencies that may be needed..."
	$(UV) pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "Attempting AlphaPose installation with build fixes..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		echo "Detected macOS - using CPU-only installation"; \
		FORCE_CPU=1 $(UV) pip install --no-build-isolation "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose" || \
		echo "❌ AlphaPose installation failed. Consider using: make install-pose-alternatives"; \
	else \
		$(UV) pip install --no-build-isolation "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose" || \
		echo "❌ AlphaPose installation failed. Consider using: make install-pose-alternatives"; \
	fi
	@echo "Build fix attempt completed!"

# -------------------------------------------------------------------
# AlphaPose installation (requires NumPy to be present)
# -------------------------------------------------------------------
install-alphapose: fix-alphapose-build
	@echo "Installing AlphaPose (may take a while)..."
	$(UV) pip install "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose"
	@echo "AlphaPose installed successfully!"

download-models:
	@echo "Creating model directory..."
	mkdir -p $(MODEL_DIR)
	@echo "Downloading AlphaPose models..."
	@if [ ! -f $(MODEL_DIR)/yolox_x.pth ]; then \
		echo "Downloading YOLOX-X detector model..."; \
		curl -L -o $(MODEL_DIR)/yolox_x.pth $(YOLOX_MODEL_URL); \
	fi
	@if [ ! -f $(MODEL_DIR)/fast_res50_256x192.pth ]; then \
		echo "Downloading FastPose model..."; \
		curl -L -o $(MODEL_DIR)/fast_res50_256x192.pth $(POSE_MODEL_URL); \
	fi
	@echo "Models downloaded successfully!"
	@echo "NOTE: To use 3D pose estimation, manually download the SMPL model from:"
	@echo "      https://smpl.is.tue.mpg.de/ and place it in the model_files directory"

# ===== DEVELOPMENT TARGETS =====
run: run-server run-frontend

run-cli: install
	@echo "Running CLI with example (using auto-detected backend)..."
	@if [ ! -d "test_images" ]; then \
		echo "Creating test image..."; \
		mkdir -p test_images; \
		python -c "import cv2; import numpy as np; img = np.zeros((400, 400, 3), dtype=np.uint8); img.fill(50); cv2.circle(img, (200, 80), 30, (255, 255, 255), -1); cv2.line(img, (200, 110), (200, 250), (255, 255, 255), 8); cv2.line(img, (200, 150), (150, 180), (255, 255, 255), 6); cv2.line(img, (200, 150), (250, 180), (255, 255, 255), 6); cv2.line(img, (200, 250), (170, 350), (255, 255, 255), 6); cv2.line(img, (200, 250), (230, 350), (255, 255, 255), 6); cv2.imwrite('test_images/test_person.jpg', img); print('Test image created')"; \
	fi
	$(UV_RUN) python $(CLI_DIR)/detect.py --image-dir test_images --backend auto

run-server: install
	@echo "Running FastAPI server..."
	$(VENV)/bin/uvicorn $(SERVER_DIR).app:app --reload --host 0.0.0.0 --port 8000

run-frontend: install-frontend
	@echo "Running Next.js frontend..."
	cd $(FRONTEND_DIR) && $(NPM) run dev

run-docker: docker-up

# ===== TESTING TARGETS =====
test: test-cli test-server test-integration

test-cli: install-dev
	@echo "Running CLI tests..."
	$(PYTEST) $(TEST_DIR)/test_cli.py -v

test-server: install-dev
	@echo "Running server tests..."
	$(PYTEST) $(TEST_DIR)/test_server.py -v

test-integration: install-dev
	@echo "Running integration tests..."
	$(PYTEST) $(TEST_DIR)/test_integration.py -v

test-coverage: install-dev
	@echo "Running tests with coverage..."
	$(COVERAGE) run -m pytest $(TEST_DIR)
	$(COVERAGE) report -m
	$(COVERAGE) html
	@echo "Coverage report generated in htmlcov/"

# ===== CODE QUALITY TARGETS =====
format: install-dev
	@echo "Formatting code with black and isort..."
	$(BLACK) $(CLI_DIR) $(SERVER_DIR) $(TEST_DIR)
	$(ISORT) $(CLI_DIR) $(SERVER_DIR) $(TEST_DIR)
	@echo "Code formatted successfully!"

lint: install-dev
	@echo "Linting code with ruff..."
	$(RUFF) check $(CLI_DIR) $(SERVER_DIR) $(TEST_DIR)
	@echo "Linting completed!"

typecheck: install-dev
	@echo "Type checking with mypy..."
	$(MYPY) $(CLI_DIR) $(SERVER_DIR)
	@echo "Type checking completed!"

# ===== DOCUMENTATION TARGETS =====
docs: install-dev
	@echo "Building documentation..."
	cd $(DOCS_DIR) && $(VENV)/bin/mkdocs build
	@echo "Documentation built successfully in $(DOCS_DIR)/site/"

docs-serve: install-dev
	@echo "Serving documentation locally..."
	cd $(DOCS_DIR) && $(VENV)/bin/mkdocs serve

docs-deploy: install-dev
	@echo "Deploying documentation..."
	cd $(DOCS_DIR) && $(VENV)/bin/mkdocs gh-deploy --force
	@echo "Documentation deployed successfully!"

# ===== DOCKER TARGETS =====
docker-build:
	@echo "Building Docker images..."
	$(DOCKER_COMPOSE) build

docker-up:
	@echo "Starting Docker containers..."
	$(DOCKER_COMPOSE) up -d
	@echo "Docker containers started! Access:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Frontend: http://localhost:3000"
	@echo "  - Adminer (DB): http://localhost:8080"
	@echo "  - Redis Commander: http://localhost:8081"

docker-down:
	@echo "Stopping Docker containers..."
	$(DOCKER_COMPOSE) down

docker-logs:
	@echo "Viewing Docker container logs..."
	$(DOCKER_COMPOSE) logs -f

docker-gpu:
	@echo "Starting Docker containers with GPU support..."
	$(DOCKER_COMPOSE) --profile gpu up -d
	@echo "GPU-enabled containers started!"

# ===== DEPLOYMENT TARGETS =====
deploy: deploy-server deploy-frontend

deploy-server:
	@echo "Deploying server..."
	$(DOCKER_COMPOSE) --profile prod up -d api
	@echo "Server deployed successfully!"

deploy-frontend:
	@echo "Building frontend for production..."
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo "Deploying frontend..."
	$(DOCKER_COMPOSE) --profile prod up -d frontend
	@echo "Frontend deployed successfully!"

# ===== DATABASE TARGETS =====
db-init: install
	@echo "Initializing database..."
	$(VENV)/bin/python -c "from sqlmodel import SQLModel; from server.app import engine; SQLModel.metadata.create_all(engine)"
	@echo "Database initialized successfully!"

db-migrate: install-dev
	@echo "Creating a new migration..."
	@read -p "Enter migration name: " name; \
	$(VENV)/bin/alembic -c $(MIGRATIONS_DIR)/alembic.ini revision --autogenerate -m "$$name"
	@echo "Migration created successfully!"

db-upgrade: install-dev
	@echo "Upgrading database to latest migration..."
	$(VENV)/bin/alembic -c $(MIGRATIONS_DIR)/alembic.ini upgrade head
	@echo "Database upgraded successfully!"

db-downgrade: install-dev
	@echo "Downgrading database to previous migration..."
	$(VENV)/bin/alembic -c $(MIGRATIONS_DIR)/alembic.ini downgrade -1
	@echo "Database downgraded successfully!"

# ===== CLEANUP TARGETS =====
clean: clean-pyc clean-test

clean-pyc:
	@echo "Removing Python cache files..."
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -delete
	find . -name '.pytest_cache' -delete
	@echo "Python cache files removed!"

clean-test:
	@echo "Removing test artifacts..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	@echo "Test artifacts removed!"

clean-build:
	@echo "Removing build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf $(FRONTEND_DIR)/.next
	rm -rf $(FRONTEND_DIR)/out
	@echo "Build artifacts removed!"

clean-outputs:
	@echo "Removing output files..."
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(UPLOAD_DIR)/*
	@echo "Output files removed!"

clean-all: clean clean-build clean-outputs
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Removing node_modules..."
	rm -rf $(NODE_MODULES)
	@echo "Removing database..."
	rm -f $(DB_FILE)
	@echo "All generated files removed!"

# ===== ALL TARGET =====
all: setup test format lint docs
	@echo "All targets completed successfully!"
