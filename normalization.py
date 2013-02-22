from cv2 import *
from numpy import *
import cv2 as cv2


#-- Normalize Rotation ---------------------------------
#
# Given an image with one blob and  the U  matrix as in U,D,V=linalg.svd(cov(cnt.T))
# Or Given an image with one blob and  the principal and minor axis of an object respect to (0,0) corrdinates
# Then transform the blob such that the major axis is the Y-Axis and the minor axis is the X-axis
#
#-------------------------------------------------
def normalizeRotation(imbin, basis,mx,my):

	newbasis=array([[mx,my],[mx-1,my],[mx,my+1]],float32)
	oldbasis=array([[mx,my],[mx+basis[1][0],my+basis[1][1]],[mx+basis[0][0],my+basis[0][1]]],float32)

	t=getAffineTransform(oldbasis,newbasis)
	
	return warpAffine(imbin,t,(imbin.shape[1],imbin.shape[0]))	


#-- Axes ---------------------------------
#
# Given an Image, return the three mirror images
#
#-------------------------------------------------
def findMirrors(im,mx,my):

    #**Find Mirror 1
	newbasis=array([[mx,my],[mx-1,my],[mx,my+1]],float32)
	oldbasis=array([[mx,my],[mx+1,my],[mx,my+1]],float32)
	t=getAffineTransform(oldbasis,newbasis)
	mirr1=warpAffine(im,t,(im.shape[1],im.shape[0]))	

    #**Find Mirror 2
	oldbasis=array([[mx,my],[mx-1,my],[mx,my-1]],float32)
	t=getAffineTransform(oldbasis,newbasis)
	mirr2=warpAffine(im,t,(im.shape[1],im.shape[0]))	

    #**Find Mirror 3
	oldbasis=array([[mx,my],[mx+1,my],[mx,my-1]],float32)
	t=getAffineTransform(oldbasis,newbasis)
	mirr3=warpAffine(im,t,(im.shape[1],im.shape[0]))	


	return mirr1,mirr2,mirr3


#-- Normalize Contour,Image to Position -----------------------------
#
# Place the contour as left and high as possible in an image
#
#-------------------------------------------------------------
def NormContourToPosition(imout,fcnt):
	x,y,w,h = boundingRect(fcnt)
	fcnt=reshape(fcnt,(fcnt.shape[0],2))
	imout=imout[y:y+h][:,x:x+w]	
	fcnt[:][:,0]-=x
	fcnt[:][:,1]-=y
	fcnt=reshape(fcnt,(fcnt.shape[0],1,fcnt.shape[1]))
	return imout,fcnt



#-- Axes ---------------------------------
#
# Given a Contour find the major and minor axis
# and the center where the axes should be located
#
#-------------------------------------------------
def axes(imbin,cnt):

	M = moments(cnt)
	center = (array([M['m10'],M['m01']])/M['m00']).astype(int)
	
	cnt=reshape(cnt,(cnt.shape[0],2))
	u,d,v=linalg.svd(cov(cnt.T))
	return u,center


	
if __name__=="__main__":

	#Read an image and convert to 1D
	imbin=imread("Instrument2.jpg",0)

	#Make sure the image is binary
	for i in range(imbin.shape[0]):
		for j in range(imbin.shape[1]):
			if(imbin[i][j]<125):
				imbin[i][j]=0
			else:
				imbin[i][j]=255

	contours, hierarchy = findContours(imbin.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		#mask = zeros(imbin.shape,uint8)
		assert(len(contours)==1)
		u,center=axes(imbin,cnt)
		imrot=normalizeRotation(imbin,u,center[0],center[1])
		mirr1,mirr2,mirr3=findMirrors(imrot,center[0],center[1])
		
		imwrite("rotated_normalized_image.jpg",imrot)
		imwrite("rotated_normalized_mirror1.jpg",mirr1)
		imwrite("rotated_normalized_mirror2.jpg",mirr2)
		imwrite("rotated_normalized_mirror3.jpg",mirr3)


