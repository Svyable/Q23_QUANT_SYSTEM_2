# Makefile for Max Dama 2026 LaTeX document

.PHONY: all clean pdf view

# Default target
all: pdf

# Compile PDF
pdf: Max_Dama_2026_Automated_Trading.pdf

Max_Dama_2026_Automated_Trading.pdf: Max_Dama_2026_Automated_Trading.tex references.bib
	pdflatex Max_Dama_2026_Automated_Trading.tex
	bibtex Max_Dama_2026_Automated_Trading
	pdflatex Max_Dama_2026_Automated_Trading.tex
	pdflatex Max_Dama_2026_Automated_Trading.tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.fdb_latexmk *.fls

# Clean all generated files
cleanall: clean
	rm -f Max_Dama_2026_Automated_Trading.pdf

# View PDF (requires viewer)
view: pdf
	xdg-open Max_Dama_2026_Automated_Trading.pdf 2>/dev/null || \
	open Max_Dama_2026_Automated_Trading.pdf 2>/dev/null || \
	start Max_Dama_2026_Automated_Trading.pdf 2>/dev/null || \
	echo "Please open Max_Dama_2026_Automated_Trading.pdf manually"