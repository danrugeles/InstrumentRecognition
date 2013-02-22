from cv2 import *
from numpy import *
import operator
import itertools
import time
from scipy.spatial.distance import mahalanobis
import sys
from mysys import *
from myblobs import *
from normalization import *
from signature import *


MINSIZECONTOUR=200
SEGMENTATIONPROB=0

MORPHOLOGYCLOSINGS=3
Toy=False
PROCESS=True
DEBUG=False
BLOBNOISE=20
MINBLOBSIZE=100


#-- Normalize ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def normalize(imghls):
	"""a=array([[1,5,5],[5,50,1],[1,1,5]])
	print a
	print a*(a!=5)+(a==5)*10
	#print (a==5)*10+a
	#imwrite("a.jpg",a)"""
	normalization=((imghls[:][:,:][:,:,0]+1.0)**2+(imghls[:][:,:][:,:,1]+1.0)**2+(imghls[:][:,:][:,:,2]+1.0)**2)**0.5
	normalization=array([normalization])
	imghls[:][:,:][:,:,0]=255*(imghls[:][:,:][:,:,0]/normalization)
	imghls[:][:,:][:,:,1]=255*(imghls[:][:,:][:,:,1]/normalization)
	imghls[:][:,:][:,:,2]=255*(imghls[:][:,:][:,:,2]/normalization)

	return imghls

#-- evalMultiGaussian ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def evalMultiGaussian(mean,cova,point):
	dimension = len(point)
	mean=array(mean)
	point=array(point)
	detCov = sqrt(linalg.det(cova))
	frac = (2*pi)**(-dimension/2.0) * (1/detCov)
	fprime = (mean - point)**2
	return frac * exp(-0.5*dot(dot(fprime, linalg.inv(cova)),fprime.T))

#-- Process ---------------------------------
#
# Saves process if flag turned on
#
#-------------------------------------------------
def Process(msg,img):
	if PROCESS:
		print msg
		if img != None:
			img2=img.copy()
			putText(img2,msg,(50,50),FONT_HERSHEY_COMPLEX_SMALL,2,(100,100,100))		
			imwrite(image+"/"+msg+".jpg",img2)



#---------------------------------------Dan-R.           
#              |\/|  /\  | |\|               *
#              |  | /  \ | | \               *
#--------------------------------------------*


if __name__ == '__main__':
	out=executeUnix("ls,images")
	images=out.split("\n")

	#for image in ['5','20']:
	for image in ['21']:
		#image=image.strip(".JPG")


	#start= time.clock()#-------------------------------------------STAGE 1: PREPROCESSING

		Process("Processing Image "+image+".JPG",None)
		img = imread('images/'+image+'.JPG')
		img = medianBlur(img,5)

		impre = bilateralFilter(img,7,7*2,7/2)
		Process("1a.Preprocessed",impre)	

		imgnormal=normalize(impre.copy())
		Process("1b.Normalized",imgnormal)

		imghls=cvtColor(impre, cv.CV_RGB2HLS)
		imghls[:][:,:][:,:,1]=0;
		Process("1c.HLS",imghls)

		imgcanny=Canny(imghls,50,100)	
		#imgclean=deleteAllBlobs(imgcanny,BLOBNOISE)
		element = getStructuringElement(MORPH_RECT,(3,3))	
		imgcanny=morphologyEx(imgcanny,MORPH_DILATE,element,iterations=1)
		Process("1d.Canny",imgcanny)

		imgbin=morphologyEx(imgcanny,MORPH_CLOSE,element,iterations=MORPHOLOGYCLOSINGS)
		#imgbin=morphologyEx(imgcanny,MORPH_ERODE,element2,iterations=MORPHOLOGYCLOSINGS)
		Process("1e.FinalBinary",imgbin)


		#start= time.clock()#-------------------------------------------------------------
		#endtime()#-----------------------------------------------------STAGE 2: FIND BLOBS

		Process('2.FindBlobs',None)		
		blobs=AllBlobs(imgbin)

		#start= time.clock()#-------------------------------------------------------------
		#endtime()#-------------------------------------------------STAGE 3: SEGMENT IMAGES

		#** Find Maximum Blob
		maxlen=0
		for k,v in blobs.iteritems():
			if(len(blobs[k])>maxlen):
				maxlen=len(blobs[k])
				maxlenkey=k

		covm=getBlobCov(blobs,maxlenkey,imgnormal)
		backgroundmean=getBlobMean(blobs,maxlenkey,imgnormal)

		imgseg=255*ones(imgbin.shape)
		imback=255*ones(imgbin.shape)

		#**Generate Segmented image
		for k,v in blobs.iteritems():
			tempmean=getBlobMean(blobs,k,imgnormal)
			prob=evalMultiGaussian(backgroundmean,covm,tempmean)
			if DEBUG: 	print "for key",k,"the distance is:",prob

			if(prob>SEGMENTATIONPROB):
				for i,j in blobs[k]:
					imgseg[i][j]=0

		#**Generate Background image
		for i,j in blobs[maxlenkey]:
			imback[i][j]=0
	
		Process("3a.SegmentedImage",imgseg)
		Process("3b.SegmentedBackground",imback)

		imback=asarray(imback,dtype=uint8)

		#start= time.clock()#-------------------------------------------------------------
		#endtime()#--------------------------------------------------STAGE 4: FIND CONTOURS

		contours, hierarchy2 = findContours(imback.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)

		#start= time.clock()#----------------------------------------------------------------
		#endtime()#------------------------------------------------STAGE 5: EXTRACT FEATURES

		i=0

		for cnt in contours:
			if(len(cnt)>MINBLOBSIZE):	

				#instrumentseg=imgseg[y:y+h][:,x:x+w]
				#imwrite("hell"+str(i)+".jpg",instrumentseg)		

				x,y,w,h = boundingRect(cnt)
		
				#**Draw contours in a larger image so that when you rotate you dont go out of bounds
				instrument=zeros((2*imback.shape[0],2*imback.shape[1]))
				drawContours(instrument,[cnt],0,255,-1)
				t=array([[1,0,instrument.shape[0]*0.1],[0,1,instrument.shape[1]*0.1]])
				instrument=warpAffine(instrument,t,(instrument.shape[1],instrument.shape[0]))

				#**Normalize scale and rotation	
				u,center=axes(instrument,cnt)
				imout=normalizeRotation(instrument,u,center[0]+instrument.shape[0]*0.1,center[1]+instrument.shape[1]*0.1)

				mirr1,mirr2,mirr3=findMirrors(imout,center[0]+instrument.shape[0]*0.1,center[1]+instrument.shape[1]*0.1)

				imout=asarray(imout,dtype=uint8)

				mirr1=asarray(mirr1,dtype=uint8)
				mirr2=asarray(mirr2,dtype=uint8)
				mirr3=asarray(mirr3,dtype=uint8)

				#**Find final contour of instrument 
				fcontours, fhierarchy2 = findContours(imout.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)
				fcontours1, fhierarchy2 = findContours(mirr1.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)
				fcontours2, fhierarchy2 = findContours(mirr2.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)
				fcontours3, fhierarchy2 = findContours(mirr3.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)

				if len(fcontours)>1:
					print "Warning: two contours were found"
					maxi=0
					for i,cnt in enumerate(fcontours):
						if(len(cnt)>maxi):
							maxidx=i
					fcnt=fcontours[i]
					fcnt1=fcontours1[i]
					fcnt2=fcontours2[i]
					fcnt3=fcontours3[i]						
				else:
					fcnt=fcontours[0]
					fcnt1=fcontours1[0]
					fcnt2=fcontours2[0]
					fcnt3=fcontours3[0]
			
				#**Normalize the final contour and image to position and store them as features
				imout,fcnt=NormContourToPosition(imout,fcnt)
				mirr1,fcnt1=NormContourToPosition(mirr1,fcnt1)
				mirr2,fcnt2=NormContourToPosition(mirr2,fcnt2)
				mirr3,fcnt3=NormContourToPosition(mirr3,fcnt3)

				#**Manually label the features
				imshow("Instrument", imout)
				if(waitKey(200)==27):
					break
				label=input("Label the image ")

				#**Original image
				Process("5.Ins #"+str(label),imout)
				imwrite("cnt"+image+"_"+str(label)+".jpg",imout)	
				save("storage/cnt"+image+"_"+str(label),fcnt)
				save("storage/cnt"+image+"_"+str(label+7),fcnt1)
				save("storage/cnt"+image+"_"+str(label+14),fcnt2)
				save("storage/cnt"+image+"_"+str(label+21),fcnt3)
				i=i+1
	
		destroyAllWindows()



