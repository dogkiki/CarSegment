#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <imgproc/imgproc.hpp>

using namespace cv;

//ȥ��Ե���
void Denoise(IplImage* img_src)
{
	IplImage* histogramImage=cvCreateImage(cvGetSize(img_src),8,1);
	int width=img_src->width;
	int height=img_src->height;
	int *colheight =new int[height];    
	int value;
	memset(colheight,0,height*4);  //������븳��ֵΪ�㣬��������޷��������顣  
	for(int i=0;i<height;i++)   
	{ 
		for(int j=0;j<width;j++)    
		{    
			//value=cvGet2D(dst,j,i).val[0];  
			value=CV_IMAGE_ELEM(img_src,uchar,i,j);
			if(value==255)    
			{    
				colheight[i]++; //ͳ��ÿ�еİ�ɫ���ص� 
			}    
		}
	}    
	int up=0,down=0,h=0;//���������ɫ��ֵ�����ұ߽�,h��Ϊ����hang[6]������
	int hang[2]={0,0};//׼���漸������ķָ�����
	for(int i=0.5*height;i<height;i++)
	{ 
		if(colheight[i]>0.3*height)//�������ĺ�ɫ��ֵ�����и߼�20���򣬣�������
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

//�ж��Ƿ�Ϊ��ɫ���������ǣ���ڰ׻���
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
			if(cPointR>150&cPointB<100&cPointG<100)    //��ȡ��ɫ��������������Ϊ��ɫ  
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
	char*str="C:\\Users\\KC\\Desktop\\�����ҵ\\CarSegment\\CarSegment\\picture\\0001.jpg";//�˴�Ϊ����ͼ��·�� 
	src=cvLoadImage(str);
	cvShowImage("srcImage",src);//��ʾԭʼͼ��

    //��ֵ��ͼ��
	IplImage*pImgThreshold,*pImgTemp;
	pImgTemp=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,3);
	cvCopyImage(src,pImgTemp);
	ChangeCharColor(pImgTemp);
	pImgThreshold=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);    //����ͼ��
	cvCvtColor(pImgTemp,pImgThreshold,CV_RGB2GRAY);                             //RGBͼת�Ҷ�ͼ
	dst=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
	//	cvSmooth(pImgThreshold,pImgThreshold,CV_GAUSSIAN,3,3,0,0);//3x3          //3*3��˹�˲�
	cvThreshold(pImgThreshold, dst, 0, 255, CV_THRESH_BINARY| CV_THRESH_OTSU);//����Ӧ��ֵ��ͼ��
	ChangeImage(dst);         //�ж��Ƿ��ǻ�ɫ���������ǣ���ڰ׻���
	Denoise(dst);                  //ȥ��ͼ���Ե����
	cvNamedWindow("ThresholdImage");
	cvShowImage("ThresholdImage",dst);//��ʾ��ֵ��ͼ��
		
	//��ֵ��ͼ��ָ�
	IplImage* histogramImage=cvCreateImage(cvGetSize(dst),8,1);
	int width=dst->width;
	int height=dst->height;
	int *colheight =new int[width];    
	int value;
	memset(colheight,0,width*4);  //������븳��ֵΪ�㣬��������޷���������  
	for(int i=0;i<width;i++)   
	{ 
		for(int j=0;j<height;j++)    
		{    
			value=CV_IMAGE_ELEM(dst,uchar,j,i);
			if(value==255)    
			{    
				colheight[i]++; //ͳ��ÿ�еİ�ɫ���ص� 
			}    
		}
	}    
	int before=0,after=0,h=0;//���ư�ɫ��ֵ�����ұ߽�
	int hang[20]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};//����ַ��߽�����
	int countt=0;
	for(int i=0;i<width;i++)
	{ 
		if(colheight[i]>0.25*height)               //�����ɫ��ֵ����0.2��ͼ��߶ȣ������˴����ַ�����
		{   
			for(before=i;colheight[before]>0.05*height;before--)             //�ɷ�ֵ����ǰ����ֱ���߶�С�ڵ���0.05��ͼ��߶ȣ��˴�Ѱ���ַ���߽�
			{
			} 
			for(after=i;colheight[after]>0.05*height||(countt==0&&after<0.12*width);after++)     //�˴�Ѱ���ַ���߽�
			{
			} 
			if (after-before<0.02*width)           //ȥ�����̫С�ķ��ַ�����
			{
				i=after;
			}
			else
			{
				hang[h++]=before;                   //��¼�ַ���߽�
				hang[h++]=after;                      //��¼�ַ��ұ߽�
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
		//����Ѱ�ҵ����ַ��߽�ָ��ַ�
		cvRectangle(pImgSegment, cvPoint(hang[index++], int(0.08*height)), cvPoint(hang[index++],int(0.9*height)), CV_RGB(0, 0, 255),1,4);	
	}
	cvNamedWindow("SegmentImage");
	cvShowImage("SegmentImage",pImgSegment);//��ʾ��ֵ���ָ�ͼ��
	cvWaitKey(0);	
	cvReleaseImage(&pImgSegment);
	cvReleaseImage(&histogramImage);
	cvReleaseImage(&pImgThreshold);
	cvReleaseImage(&pImgTemp);
	return 0;
}