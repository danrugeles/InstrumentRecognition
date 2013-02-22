from cv2 import *
from numpy import *

BLOBNOISE=34

#-- Delete Blob From Contour ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def deleteBlobFromContour(imgbin,contour):
	contour.reshape((contour.shape[0],contour.shape[2]))		
	drcorner=amax(contour,axis=0)[0]
	ulcorner=amin(contour,axis=0)[0]	
	for i in range(ulcorner[0],drcorner[0]):
		for j in range(ulcorner[1],drcorner[1]):
			imgbin[j][i]=255

#-- getBlobCov ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def getBlobCov(blobs,key,imgin):
	if len(imgin.shape)==3:
		bluein=imgin[:][:,:][:,:,0]
		greenin=imgin[:][:,:][:,:,1]
		redin=imgin[:][:,:][:,:,2]
		coords=getBlobCoords(blobs,key)
		#print mean(greenin[coords])
		#print var(greenin[coords])
		#return cov([redin[coords], greenin[coords], bluein[coords]])
		return array([[var(redin[coords]),0,0],[0, var(greenin[coords]),0],[0,0,var(bluein[coords])]])

	else:
		assert("GetBlobCov, please pass an image with 3 channels")


#-- getBlobMean ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def getBlobMean(blobs,key,imgin):
	if len(imgin.shape)==3:
		bluein=imgin[:][:,:][:,:,0]
		greenin=imgin[:][:,:][:,:,1]
		redin=imgin[:][:,:][:,:,2]
		coords=getBlobCoords(blobs,key)
		return (mean(redin[coords]),mean(greenin[coords]),mean(bluein[coords]))
	else:
		assert("GetBlobMean, please pass an image with 3 channels")


#-- getBlobCoords ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def getBlobCoords(blobs,key):
	return zip(*blobs[key])

#-- deleteAllBlobs ---------------------------------
#
#Delete all 8-connected blobs with size less than
#
#-------------------------------------------------
def deleteAllBlobs(imgin,minsize):
	im=imgin.copy()
	importantBlob=list()
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if(im[i][j]==255):
				importantBlob+=deleteBlob(im,i,j,minsize)

	while(len(importantBlob)>0):
		idx=importantBlob.pop()
		im[idx]=255
	return im

#-- deleteBlob ---------------------------------
#
#Delete Blob connected to im[i][j] if size less than minsize
#
#-------------------------------------------------
def deleteBlob(im,i,j,minsize):
	stack=[]
	visited=[]
	im[i][j]=100
	stack.append((i,j))

	while(len(stack)>0):
		i,j=stack.pop()			
		visited.append((i,j))
		im[i][j]=0		

		if(im[i][j+1]==255):
			stack.append((i,j+1))
		if(im[i+1][j+1]==255):
			stack.append((i+1,j+1))
		if(im[i+1][j]==255):
			stack.append((i+1,j))
		if(im[i][j-1]==255):
			stack.append((i,j-1))
		if(im[i+1][j-1]==255):
			stack.append((i+1,j-1))
		if(im[i-1][j-1]==255):
			stack.append((i-1,j-1))
		if(im[i-1][j]==255):
			stack.append((i-1,j))
		if(im[i-1][j+1]==255):
			stack.append((i-1,j+1))
		
	if(len(visited)<minsize):
		del visited
		return []
	else:
		return visited

#-- AllBlobs ---------------------------------
#
# Find All Blobs in an Image
#
#-------------------------------------------------
def AllBlobs(imgbin):
	blobs=dict()
	labels=zeros(imgbin.shape)
	label=1
	labels[0][0]=label
	region=imgbin[0][0]
	for i in range(imgbin.shape[0]):
		for j in range(imgbin.shape[1]):

			#Check for change in region
			if(region==imgbin[i][j]):
				labels[i][j]=label
			else:
				region=imgbin[i][j]
				label=label+1
				labels[i][j]=label
			try:
				blobs[labels[i][j]].append((i,j))	
			except KeyError:
				blobs[labels[i][j]]=[(i,j)]				

	#start= time.clock()#-------------------------------------------------------------
	#endtime()#-----------------------------------------------------------------------

	for i in range(imgbin.shape[0]):
		for j in range(imgbin.shape[1]):
			#relabel old labels to new labels if necccesaryas
			if(i<imgbin.shape[0]-1 and labels[i+1][j]!=labels[i][j] and imgbin[i+1][j]==imgbin[i][j]):	

				#1. Update the label matrix 
				obsoletelabel=labels[i+1][j]
				label=labels[i][j]
				for k,l in blobs[obsoletelabel]:
					labels[k][l]=label
				
				#2. Move all the elements in the dictionary from one position to the other.
				blobs[labels[i][j]]+=blobs[obsoletelabel]
				del blobs[obsoletelabel]

	return blobs
							
if __name__=="__main__":
	im=imread("toy.jpg")
	im=cvtColor(im, cv.CV_RGB2GRAY);

	#Make sure the image is binary
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if(im[i][j]<125):
				im[i][j]=0
			else:
				im[i][j]=255

	deleteAllBlobs(im,BLOBNOISE)

	imwrite("outtoy.jpg",im)
			

