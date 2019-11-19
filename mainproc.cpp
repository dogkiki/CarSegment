#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <imgproc/imgproc.hpp>

using namespace cv;

//去边缘噪点
void Denoise(IplImage* img_src)
{
	IplImage* histogramImage=cvCreateImage(cvGetSize(img_src),8,1);
	int width=img_src->width;
	int height=img_src->height;
	int *colheight =new int[height];    
	int value;
	memset(colheight,0,height*4);  //数组必须赋初值为零，否则出错。无法遍历数组。  
	for(int i=0;i<height;i++)   
	{ 
		for(int j=0;j<width;j++)    
		{    
			//value=cvGet2D(dst,j,i).val[0];  
			value=CV_IMAGE_ELEM(img_src,uchar,i,j);
			if(value==255)    
			{    
				colheight[i]++; //统计每行的白色像素点 
			}    
		}
	}    
	int up=0,down=0,h=0;//控制纵向黑色峰值的左右边界,h作为下面hang[6]的内容
	int hang[2]={0,0};//准备存几个纵向的分割线条
	for(int i=0.5*height;i<height;i++)
	{ 
		if(colheight[i]>0.3*height)//如果纵向的黑色峰值大于行高减20，则，，，，，
		{   
			for(up=i;colheight[up]>0.1*height;up--)
			{
			} 
			for(down=i;colheight[down]>0.1*height;down++)
			{
			} 	
			break;
		}				
	}
	CvRect rc = cvRect(0,0,img_src->width,img_src->height);
	for(int y = rc.y;y<rc.y+rc.height;y++)     
	{     
		for(int x =rc.x;x<rc.x+rc.width;x++)     
		{   
			if(x<0.02*width||x>0.96*width||y<up||y>down||y<0.12*height||y>0.88*height)
			{
					cvSet2D(img_src,y,x,cvScalar(0,0,0));
			}
		}     
	}       
}

//判断是否为黄色背景，若是，则黑白互换
void ChangeImage(IplImage* image)
{
	int value;
	long long count=0;
	int width=image->width;
	int height=image->height;
	for(int i=0;i<width;i++)   
	{ 
		for(int j=0;j<height;j++)    
		{    
			value=CV_IMAGE_ELEM(image,uchar,j,i);
			if(value==255)    
			{    
				count++;
			}    
		}
	}
	if (count>0.45*width*height)
	{
		for(int i=0;i<width;i++)   
		{ 
			for(int j=0;j<height;j++)    
			{    
				value=CV_IMAGE_ELEM(image,uchar,j,i);
				if(value==255)    
				{    
					cvSet2D(image,j,i,cvScalar(0));
				} 
				else
					cvSet2D(image,j,i,cvScalar(255));
			}
		}
	}
}

void ChangeCharColor(IplImage* image)
{
	Mat srcImage=Mat(image,false);
	int i,j;  
	int cPointB,cPointG,cPointR;  
	for(i=1;i<srcImage.rows;i++)  
		for(j=1;j<srcImage.cols;j++)  
		{  
			cPointB=srcImage.at<Vec3b>(i,j)[0];  
			cPointG=srcImage.at<Vec3b>(i,j)[1];  
			cPointR=srcImage.at<Vec3b>(i,j)[2];  
			if(cPointR>150&cPointB<100&cPointG<100)    //提取红色，将该区域设置为白色  
			{  
				srcImage.at<Vec3b>(i,j)[0]=255;  
				srcImage.at<Vec3b>(i,j)[1]=255;  
				srcImage.at<Vec3b>(i,j)[2]=255;  
			}  
			image = &IplImage(srcImage);
		}  
}

int main()
{
	IplImage* src=NULL,*dst=NULL;
	cvNamedWindow("srcImage");
	char*str="C:\\Users\\KC\\Desktop\\设计作业\\CarSegment\\CarSegment\\picture\\0001.jpg";//此处为加载图像路径 
	src=cvLoadImage(str);
	cvShowImage("srcImage",src);//显示原始图像

    //二值化图像
	IplImage*pImgThreshold,*pImgTemp;
	pImgTemp=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,3);
	cvCopyImage(src,pImgTemp);
	ChangeCharColor(pImgTemp);
	pImgThreshold=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);    //创建图像
	cvCvtColor(pImgTemp,pImgThreshold,CV_RGB2GRAY);                             //RGB图转灰度图
	dst=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	//	cvSmooth(pImgThreshold,pImgThreshold,CV_GAUSSIAN,3,3,0,0);//3x3          //3*3高斯滤波
	cvThreshold(pImgThreshold, dst, 0, 255, CV_THRESH_BINARY| CV_THRESH_OTSU);//自适应二值化图像
	ChangeImage(dst);         //判断是否是黄色背景，若是，则黑白互换
	Denoise(dst);                  //去除图像边缘噪声
	cvNamedWindow("ThresholdImage");
	cvShowImage("ThresholdImage",dst);//显示二值化图像
		
	//二值化图像分割
	IplImage* histogramImage=cvCreateImage(cvGetSize(dst),8,1);
	int width=dst->width;
	int height=dst->height;
	int *colheight =new int[width];    
	int value;
	memset(colheight,0,width*4);  //数组必须赋初值为零，否则出错，无法遍历数组  
	for(int i=0;i<width;i++)   
	{ 
		for(int j=0;j<height;j++)    
		{    
			value=CV_IMAGE_ELEM(dst,uchar,j,i);
			if(value==255)    
			{    
				colheight[i]++; //统计每列的白色像素点 
			}    
		}
	}    
	int before=0,after=0,h=0;//控制白色峰值的左右边界
	int hang[20]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};//存放字符边界坐标
	int countt=0;
	for(int i=0;i<width;i++)
	{ 
		if(colheight[i]>0.25*height)               //如果白色峰值大于0.2倍图像高度，则表面此处有字符出现
		{   
			for(before=i;colheight[before]>0.05*height;before--)             //由峰值处向前遍历直到高度小于等于0.05倍图像高度，此处寻找字符左边界
			{
			} 
			for(after=i;colheight[after]>0.05*height||(countt==0&&after<0.12*width);after++)     //此处寻找字符左边界
			{
			} 
			if (after-before<0.02*width)           //去除宽度太小的非字符噪声
			{
				i=after;
			}
			else
			{
				hang[h++]=before;                   //记录字符左边界
				hang[h++]=after;                      //记录字符右边界
				i=after;
				countt++;
			}			
		}				
	}
	int index=0;
	IplImage*pImgSegment;
	pImgSegment  = cvCreateImage(cvGetSize(dst),8,3);
	cvCvtColor(dst,pImgSegment,CV_GRAY2RGB);
	for(int i=0;i<7;i++)
	{
		//利用寻找到的字符边界分割字符
		cvRectangle(pImgSegment, cvPoint(hang[index++], int(0.08*height)), cvPoint(hang[index++],int(0.9*height)), CV_RGB(0, 0, 255),1,4);	
	}
	cvNamedWindow("SegmentImage");
	cvShowImage("SegmentImage",pImgSegment);//显示二值化分割图像
	cvWaitKey(0);	
	cvReleaseImage(&pImgSegment);
	cvReleaseImage(&histogramImage);
	cvReleaseImage(&pImgThreshold);
	cvReleaseImage(&pImgTemp);
	return 0;
}