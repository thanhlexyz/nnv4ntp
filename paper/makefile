MAIN_FILE = main
PDF_VIEWER = okular
all: pdf view
compile:
	pdflatex $(MAIN_FILE) > /dev/null 2>&1
biber:
	biber -quiet $(MAIN_FILE)
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.acn *.bcf *.glo *.ist *.nav *.run.xml *.snm
view:
	$(PDF_VIEWER) $(MAIN_FILE).pdf
