.PHONY: serve open help

PORT ?= 8000

help: ## Show available targets
	@echo Agent DJ targets:
	@echo   make serve   start the frontend dev server on http://localhost:$(PORT)
	@echo   make open    open the frontend in your default browser

serve: ## Start the frontend dev server
	cd frontend && python -m http.server $(PORT)

open: ## Open the frontend in your default browser
	python -m webbrowser http://localhost:$(PORT)
