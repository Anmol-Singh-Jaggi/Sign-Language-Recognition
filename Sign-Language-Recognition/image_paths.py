import os

path = "img"
file = open("image_paths.txt","w")
for (dirpath,dirnames,filenames) in os.walk(path):
	for filename in filenames:
		parent = os.path.join(dirpath,filename)
		label = dirpath[len(dirpath)-1]
		line = parent + "\t" + label + '\n'
		file.write(line)
file.close()
