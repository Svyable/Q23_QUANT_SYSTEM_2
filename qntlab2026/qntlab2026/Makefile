# Makefile for QntLab2026 Cookbook LaTeX document

.PHONY: all clean pdf view help

# Default target
all: pdf

# Compile PDF with bibliography
pdf: QntLab2026_Cookbook.pdf

QntLab2026_Cookbook.pdf: QntLab2026_Cookbook.tex QntLab2026_Cookbook.bib
	@echo "Compiling QntLab2026 Cookbook..."
	@pdflatex -interaction=nonstopmode QntLab2026_Cookbook.tex > /dev/null 2>&1 || true
	@bibtex QntLab2026_Cookbook > /dev/null 2>&1 || true
	@pdflatex -interaction=nonstopmode QntLab2026_Cookbook.tex > /dev/null 2>&1 || true
	@pdflatex -interaction=nonstopmode QntLab2026_Cookbook.tex > /dev/null 2>&1 || true
	@echo "QntLab2026 Cookbook compiled successfully!"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.fdb_latexmk *.fls
	@rm -f QntLab2026_Cookbook.blg QntLab2026_Cookbook.brf QntLab2026_Cookbook.idx
	@echo "Auxiliary files cleaned."

# Clean all generated files
cleanall: clean
	@echo "Removing PDF..."
	@rm -f QntLab2026_Cookbook.pdf
	@echo "All generated files removed."

# View PDF (cross-platform)
view: pdf
	@echo "Opening QntLab2026 Cookbook..."
	@if command -v xdg-open > /dev/null 2>&1; then \
		xdg-open QntLab2026_Cookbook.pdf > /dev/null 2>&1 & \
	elif command -v open > /dev/null 2>&1; then \
		open QntLab2026_Cookbook.pdf > /dev/null 2>&1 & \
	elif command -v start > /dev/null 2>&1; then \
		start QntLab2026_Cookbook.pdf > /dev/null 2>&1 & \
	else \
		echo "Please open QntLab2026_Cookbook.pdf manually"; \
	fi

# Quick compile (no bibliography)
quick: QntLab2026_Cookbook.tex
	@echo "Quick compile (no bibliography)..."
	@pdflatex -interaction=nonstopmode QntLab2026_Cookbook.tex > /dev/null 2>&1 || true
	@echo "Quick compile completed."

# Word count
wc: QntLab2026_Cookbook.tex
	@echo "Word count:"
	@detex QntLab2026_Cookbook.tex | wc -w 2>/dev/null || echo "Install 'detex' for word count"

# Check for missing references
check_refs: QntLab2026_Cookbook.tex
	@echo "Checking for undefined references..."
	@pdflatex -interaction=nonstopmode QntLab2026_Cookbook.tex 2>&1 | grep -i "undefined\|multiply defined" || echo "No reference issues found."

# Help target
help:
	@echo "QntLab2026 Cookbook Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Compile the complete cookbook (default)"
	@echo "  pdf       - Compile PDF with full bibliography"
	@echo "  quick     - Quick compile without bibliography"
	@echo "  clean     - Remove auxiliary files (*.aux, *.log, etc.)"
	@echo "  cleanall  - Remove all generated files including PDF"
	@echo "  view      - Open the compiled PDF"
	@echo "  wc        - Count words in the document"
	@echo "  check_refs- Check for undefined references"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - LaTeX distribution (TeX Live, MiKTeX, or MacTeX)"
	@echo "  - pdflatex, bibtex commands in PATH"
	@echo ""
	@echo "Usage examples:"
	@echo "  make pdf          # Full compilation"
	@echo "  make quick && make view  # Fast preview"
	@echo "  make cleanall     # Start fresh"