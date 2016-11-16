all: readme printing

readme: 
	pandoc -f markdown_github -o README.pdf README.md --variable=geometry:"margin=0.75in" --highlight-style=zenburn --variable=colorlinks:true --variable=papersize:"letter" --variable=fontsize:"10pt"

printing:
	pandoc -f markdown_github -o README-printing.pdf README.md --variable=geometry:"margin=0.75in" --highlight-style=zenburn --variable=colorlinks:true --variable=papersize:"letter" --variable=fontsize:"10pt" --variable=links-as-notes:true
