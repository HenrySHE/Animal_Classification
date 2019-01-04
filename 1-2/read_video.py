from glob import glob


path = input("Plz input the video path:")
print(path)
#path = ''
filenames = []
filenames = [vid for vid in glob(path+"*.MP4")]
#filenames = [vid for vid in glob("*.AVI")]
for vid in glob(path+"*.AVI"):
	filenames.append(vid)
filenames.sort()
if len(filenames)==0:
	print('no videos!')
else:
	print('found '+str(len(filenames))+ ' videos:')
	

for vid_name in filenames:
	print(vid_name)
