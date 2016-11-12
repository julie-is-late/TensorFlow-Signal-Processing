all: readme

readme: 
	pandoc -f markdown -o README.pdf README.md --variable=geometry:"margin=0.5in" --highlight-style=zenburn --variable=colorlinks:true


