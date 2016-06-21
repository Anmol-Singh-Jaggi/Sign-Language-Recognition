import os

path = "test_image"
file = open("test_paths.txt","w")
for (dirpath,dirnames,filenames) in os.walk(path):
	for filename in filenames:
		parent = os.path.join(dirpath,filename)
		label = filename[0]
		line = parent + "\t" + label + '\n'
		file.write(line)
file.close()
