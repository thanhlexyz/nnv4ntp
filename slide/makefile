MAIN_FILE = main
PDF_VIEWER = okular
all: pdf view
pdf:
	pdflatex $(MAIN_FILE)
	biber $(MAIN_FILE)
	pdflatex $(MAIN_FILE)
	pdflatex $(MAIN_FILE)
compile:
	pdflatex $(MAIN_FILE) # > /dev/null 2>&1
biber:
	biber $(MAIN_FILE) # > /dev/null 2>&1
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.acn *.bcf *.glo *.ist *.nav *.run.xml *.snm
view:
	$(PDF_VIEWER) $(MAIN_FILE).pdf
