//-----------------------------------------------------------------------------------------------------------------
// Victoria Depew
// Embedded Software
// Lab 7
// Winter 2018
//
// Objective: integrating microcontroller code to move paddle
//
// 	
//
//-----------------------------------------------------------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string> 
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <opencv2/cudacodec.hpp>
#include <fcntl.h>  /* File Control Definitions          */
#include <termios.h>/* POSIX Terminal Control Definitions*/
#include <unistd.h> /* UNIX Standard Definitions         */
#include <errno.h>  /* ERROR Number Definitions          */

// Select Video Source
// The MP4 demo uses a ROI for better tracking of the moving object
//#define TEST_LIVE_VIDEO


using namespace cv;
using namespace std;

//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=50;

//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

void on_trackbar( int, void* )
{//This function gets called whenever a
	// trackbar position is changed
}

int theObject[2] = {0,0};
//bounding rectangle of the object, we will use the center of this as its position.
cv::Rect objectBoundingRectangle = cv::Rect(0,0,0,0);

void printArray(float arr[], int size)
{
  int i;
  for(i = 0; i < size; i++){
    cout << arr[i] << " ";
  }
  cout << endl;

}

/*-----------------------------------------------------------------------------------
	Create Trackbars
-------------------------------------------------------------------------------------*/

void createTrackbars(){
	//create window for trackbars


    	namedWindow(trackbarWindowName,0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf( TrackbarName, "H_MIN", H_MIN);
	sprintf( TrackbarName, "H_MAX", H_MAX);
	sprintf( TrackbarName, "S_MIN", S_MIN);
	sprintf( TrackbarName, "S_MAX", S_MAX);
	sprintf( TrackbarName, "V_MIN", V_MIN);
	sprintf( TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
    	createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    	createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    	createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    	createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    	createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    	createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );


}

/*-------------------------------------------------------------------------------------
	MorphOps to dilate and erode
--------------------------------------------------------------------------------------*/
void morphOps(Mat &thresh){

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
    //dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

	erode(thresh,thresh,erodeElement);
	erode(thresh,thresh,erodeElement);


	dilate(thresh,thresh,dilateElement);
	dilate(thresh,thresh,dilateElement);
}

/*-------------------------------------------------------------------------------------
	Check if ball samples are monotonic
--------------------------------------------------------------------------------------*/

bool is_monotonic (float arr1[], int n)
{
	int a1 = 0;
	int a2 = 0; 
	int i = 0;
	
	for(i = 0; i < n-1; i++) // check for increasing array 1
    	{
		if(arr1[i] < arr1[i+1]) a1++;
	}
	
	for(i = 0; i < n-1; i++)  // check for decreasing array 1
   	{
		if(arr1[i] > arr1[i+1]) a2++;
	}


	//cout << endl << " mono = " << a1 << a2 << b1 << b2 << endl;
	
	return ((a1 == n-1) || (a2 == n-1)); // return 1 if monotonic
}

/*-------------------------------------------------------------------------------------
	Check if samples are increasing, decreasing or "mostly" constant
--------------------------------------------------------------------------------------*/
bool isIncreasing(float arr[], int size) 
{
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] >= arr[i+1]) {
            return false;
        }
    }
    return true;
}

bool isDecreasing(float arr[], int size) 
{
	for (int i = 0; i < size - 1; i++) {
        	if (arr[i] <= arr[i+1]) {
            	return false;
        	}
    	}
    return true;
}

bool mostlyConstant(float arr[], int size)
{
	int count = 0;
	for (int i = 0; i < size - 2; i++) {
		if ((arr[i+1] - arr[i]) < 2){
			count++;
		}
	}
	if (count > 4){
		return true;
	}
	else
		return false;
}						

/*-------------------------------------------------------------------------------------
	Shift array to add most recent sample
--------------------------------------------------------------------------------------*/
void leftRotate_add_sample(float arr[], float new_sample, int n)
{
    int i = 0;
    for (i = 0; i < n-1; i++)
        arr[i] = arr[i+1];
    arr[i] = new_sample;
}

/*-------------------------------------------------------------------------------------
	Struct that contains the slope and intercept of least squared fit line
--------------------------------------------------------------------------------------*/
struct lineParam 
{
    float m,b;
};


/*-------------------------------------------------------------------------------------
	Perform least squared fit on samples to get best fit line
--------------------------------------------------------------------------------------*/
lineParam lsf (float x[], float y[], int n)
{
    lineParam l;
    float xsum=0,x2sum=0,ysum=0,xysum=0;                
    float m = 0, b = 0; 
    for (int i=0;i<n;i++)
    {
        xsum=xsum+x[i];                 
        ysum=ysum+y[i];                      
        x2sum=x2sum+pow(x[i],2);
        xysum=xysum+x[i]*y[i];          
    }
    l.m=(n*xysum-xsum*ysum)/(n*x2sum-xsum*xsum);        //calculate slope
    l.b=(x2sum*ysum-xsum*xysum)/(x2sum*n-xsum*xsum);    // calculate y intercept, b     
    return l;   // return m and b of x or y vs t, where t is frame number
		// y = m*t + b, x = m*t + b
}


/*-------------------------------------------------------------------------------------
	Struct that contains the x, y and frame # of predicted impact point of ball
	with game boundaries
--------------------------------------------------------------------------------------*/

struct impactPoint  // impact point consists of x, y, and t (frame )
{
    int x,y,t;
};

/*-------------------------------------------------------------------------------------
	Calculate the impact point of the ball with game boundaries
--------------------------------------------------------------------------------------*/
impactPoint caclulateImpactPoint(lineParam line_x, lineParam line_y, float x_lim, float y_lim)
{
    float t1, t2;

    impactPoint impact;
    
    /*  y = mx + b      // equation of line
        
        in our case, we have:
        line: x = mt + b
        
        solve for t:
        t = (x-b)/m                                                              
    */
    t1 = abs(((x_lim - line_x.b)/line_x.m));
    t2 = abs(((y_lim - line_y.b)/line_y.m));
    
    
    //cout << "tx = " << tx << "  ty = " << ty;
    // if t1 is less, impact will be at time t1
    // x coordinate = x limit 
    // y coordinate = y = m*t +b, where t = t1
    if (t1 < t2) {
        impact.x = (int) x_lim;
        impact.y = (int) (line_y.m * t1 + line_y.b + 0.5f);// add 0.5f to round to nearest int
        impact.t = int(t1 + 0.5f);
        //cout << "x_lim = " << x_lim << endl;
        //cout << "impact.x = " << impact.x << endl;
        //cout << "impact.y = " << impact.y << endl;                          
        //cout << "line_y.m = " << line_y.m << endl;
        //cout << "line_y.b = " << line_y.b << endl;
        }                                          
    // if t2 is less, impact will be at time t2
    // x coordinate = x = m*t + b, where t = t2 
    // y coordinate = y limit 
    else {
        impact.x = (int) (line_x.m * t2 + line_x.b + 0.5f);// add 0.5f to round to nearest int 
        impact.y = y_lim;
        impact.t = int (t2 + 0.5f);
    }
    //cout << "t1 = " << t1 << "  t2 = " << t2 << endl;
    return impact; // return next point of impact
}
//-----------------------------------------------------------------------------------------------------------------
// int to string helper function
//-----------------------------------------------------------------------------------------------------------------
string intToString(float number){
 
    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}
 
//-----------------------------------------------------------------------------------------------------------------
// Search for Moving Object
//-----------------------------------------------------------------------------------------------------------------
Point2i searchForMovement(cv::Mat thresholdImage, cv::Mat &cameraFeed){
    //notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
    //to take the values passed into the function and manipulate them, rather than just working with a copy.
    //eg. we draw to the cameraFeed to be displayed in the main() function.

    bool objectDetected = false;
    int xpos, ypos;

    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    cv::Mat temp;

    thresholdImage.copyTo(temp);

#ifdef TEST_LIVE_VIDEO

    //find contours of filtered image using openCV findContours function
    cv::findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

#else


    int offset_x = 520; 
    int offset_y = 92;
    int width = 307;
    int height = 533;

    cv::Rect roi(offset_x, offset_y, width, height); 

    //cv:Rect roi(300, 150, 600, 400); //original code

    cv::Mat roi_temp = temp(roi); 

    //find contours of filtered image using openCV findContours function
    cv::findContours(roi_temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

#endif

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)
	objectDetected = true;
    else 
	objectDetected = false;
 
    if(objectDetected){

        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));

        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));

        xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;
 
        //update the objects positions by changing the 'theObject' array values
        theObject[0] = xpos , theObject[1] = ypos;
    }



#ifdef TEST_LIVE_VIDEO

    int x = theObject[0];
    int y = theObject[1];

#else

    float x_pos = (float)(theObject[0]+offset_x);
    float y_pos = (float)(theObject[1]+offset_y);

    	

#endif
     

 
    // velocity Vector Code

	static int n = 8;
	static float x[8]={0}, y[8]={0}, t[8] = {0}; // arrays to x,y,t position points    	
    	static int count = 0;
    	bool lsf_flag = 0;
	bool check_for_vertical = 0;
    	float north = 92, south = 623, east = 827, west = 520; // game boundaries
    	float x_bound = 0, y_bound = 0;
    	lineParam lx, ly;
    	impactPoint point_of_impact;
    	int point[2] = {0};
	static Point2i pt =  Point(0,0);
    	bool rebound_flag = false;



	// shift in new samples (x and y coordinate of the ball, and frame)
        leftRotate_add_sample(x,x_pos,n);  
        leftRotate_add_sample(y,y_pos,n); 
	leftRotate_add_sample(t,count,n);   
        count++;
   
	if (count > n-1){
            //printArray(x,n);
            //printArray(y,n);
            //printArray(t,n);
    
            // check for monotonicity
		if ((is_monotonic(x,n)) && (is_monotonic(y,n))){
                //cout<< "Both x and y are monotonic" << endl;
                //cout << "perform lsf" << endl;
                lsf_flag = 1;
            	}
		else if ((mostlyConstant(x,n)) && (is_monotonic(y,n))){ // catches the case of almost vertical ball movement
		//cout << "X is mostly constant, Y is monotonic" << endl;
		check_for_vertical = 1;
		lsf_flag = 1;
		}
            
            	else {
                //cout << "not monotonic, need another sample" << endl;
                lsf_flag = 0;
            	}
            
        }
        
	if (lsf_flag){
            // what direction is ball moving? (what bounds do we use?)

		if (check_for_vertical){
			if (isIncreasing(y,n)){
				x_bound = 10000;
				y_bound = south;
			}
			else{
				x_bound = 10000;
				y_bound = north;
			}
		}
								
            	if (isIncreasing(x,n) && isIncreasing(y,n)){
                x_bound = east;
                y_bound = south;
                //cout << "dir: east, south" << endl;
            	}
           	 else if (isIncreasing(x,n) && isDecreasing(y,n)){
                x_bound = east;
                y_bound = north;
                //cout << "dir: east, north" << endl;
            	}
            	else if (isDecreasing(x,n) && isIncreasing(y,n)){
                x_bound = west;
                y_bound = south;
                //cout << "dir: west, south" << endl;
            	}
            	else if (isDecreasing(x,n) && isDecreasing(y,n)){
                x_bound = west;
                y_bound = north;
                //cout << "dir: west, north" << endl;
                
            	}
		else if (isDecreasing(x,n) && isDecreasing(y,n)){
                x_bound = west;
                y_bound = north;
                //cout << "dir: west, north" << endl;
                
            	}
            	else{
                //cout << "***** lsf_bound error" << endl; 
            	}
           
            
            	// calculate least squares fit for t,x and t,y
            	// lx and ly are structs that return m (slope) and b (y-intercept)
            	lx = lsf(t,x,n);
            		ly = lsf(t,y,n);
            
            	//cout << "m = "<< lx.m << "   b = "<< lx.b << endl; 
            	//cout << "m = "<< ly.m << "   b = "<< ly.b << endl;  
            
           	bool horizontal_wall = false;
            	point[0] = x[n-1];
            	point[1] = y[n-1];

		while (horizontal_wall == false){
                
                // calculate next point of impact with ROI wall
            	point_of_impact = caclulateImpactPoint(lx, ly, x_bound, y_bound);
            	//cout << "Next impact point: " << point_of_impact.x << ", " << point_of_impact.y << endl;
            	//cout << "Time of impact (frame #):    " << point_of_impact.t << endl;
            
                	
                	if (point_of_impact.x == x_bound){    // if ball hitting vertical wall next

                    		// after collision with wall, x-slope will change signs
                    		lx.m = -1 * lx.m;
                    		//cout << "slope change to " << lx.m;

                    		// change intercept
                    		lx.b = point_of_impact.x - (lx.m * point_of_impact.t);

                    		// change direction (in order to change x limit)
                    		if (x_bound == east){
                        		x_bound = west;
                    		}
                   		 else { 
                        		x_bound = east;
                    		}

				if ((point_of_impact.x >= west) && (point_of_impact.x <= east) && (point_of_impact.y >= north) && 					(point_of_impact.y <= south)) {
					pt.x = point_of_impact.x;
					pt.y = point_of_impact.y;
					// draw line from last sample to horizontal impact point
					//arrowedLine(cameraFeed,Point(point[0],point[1]),Point(point_of_impact.x,point_of_impact.y),Scalar(0,0,255),2,2,0.3);

				}                   
				//update point to be point of impact
                    		point[0] = point_of_impact.x;
                    		point[1] = point_of_impact.y;                    
                	}
                
                	// if ball will hit horizontal wall next
			else {  
			
			// draw line			
                    	//cout << "p1 = " << point[0] << ",   " << point[1] << endl;
                    	//cout << "p2 = " << point_of_impact.x << ",  " << point_of_impact.y << endl;
			
                   	horizontal_wall = true;

				if ((point_of_impact.x >= west) && (point_of_impact.x <= east) && (point_of_impact.y >= north) && (point_of_impact.y <= south)) {
					pt.x = point_of_impact.x;
					pt.y = point_of_impact.y;
					//arrowedLine(cameraFeed,Point(point[0],point[1]),Point(point_of_impact.x,point_of_impact.y),Scalar(0,0,255),2,2,0.3);

				}
                	}
                    
                    
                
            	}
            
        }


	

	// draw cirlce around predicted impact point

	circle(cameraFeed, pt, 10, Scalar( 255,0,0 ), 1, 8);
	

	// draw rectangle around game area roi
	rectangle(cameraFeed,roi,cv::Scalar(0,0,255),2);

	return pt;
}

//-----------------------------------------------------------------------------------------------------------------
// Search for Paddles
//-----------------------------------------------------------------------------------------------------------------

Point2i searchForPaddles(cv::Rect roi, cv::Mat thresholdImage, cv::Mat &cameraFeed){
	
    	vector< vector<Point> > contours;
    	vector<Vec4i> hierarchy;
   	cv::Rect Pad = cv::Rect(0,0,0,0);
    	cv::Mat temp;
	Point2i pt =  Point(0,0);

	
	
	

    thresholdImage.copyTo(temp);
    
    cv::Mat roi_temp = temp(roi);
    cv::findContours(roi_temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    if(contours.size()>0){
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));
        Pad =  boundingRect(largestContourVec.at(0));

	pt.x = Pad.x+Pad.width/2;
        pt.y = Pad.y+Pad.height/2;

    }

    // Draw roi around paddle area
    rectangle(cameraFeed,roi,Scalar(255,0,0),2);

    Pad.x = Pad.x + roi.x;
    Pad.y = Pad.y + roi.y;

    // Draw box around paddle
    if(contours.size()>0){
        line(cameraFeed,Point(Pad.x,Pad.y),Point(Pad.x+Pad.width,Pad.y),Scalar(0,0,255),2);
        line(cameraFeed,Point(Pad.x,Pad.y),Point(Pad.x,Pad.y+Pad.height),Scalar(0,0,255),2);
        line(cameraFeed,Point(Pad.x+Pad.width,Pad.y),Point(Pad.x+Pad.width,Pad.y+Pad.height),Scalar(0,0,255),2);
        line(cameraFeed,Point(Pad.x,Pad.y+Pad.height),Point(Pad.x+Pad.width,Pad.y+Pad.height),Scalar(0,0,255),2);
        putText(cameraFeed,"(" + intToString(Pad.x)+","+intToString(Pad.y)+")",Point(Pad.x,Pad.y),1,1,Scalar(0,255,0),1);
    }

	return pt;
}

//-----------------------------------------------------------------------------------------------------------------
// Find Corners
//-----------------------------------------------------------------------------------------------------------------
void findCorners(cv::Mat src_gray, cv::Mat &cameraFeed){
    
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(src_gray.size(), CV_32FC1);

    // harris corner detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    int thresh = 200;
    double k = 0.04;
    
    // use corner detector
    cv::cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    // normalize
    cv::normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // Drawing circlec around corners found
    for(int j = 0; j < dst_norm.rows; j++){
        for(int i = 0; i < dst_norm.cols; i++){
            if((int) dst_norm.at<float>(j,i) > thresh){
		
                cv::circle(cameraFeed,Point(i,j),5,Scalar(0,0,255),2,8,0);
            }
        }
    }
}




//-----------------------------------------------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------------------------------------------
int main() {

	char write_buffer[3];
  	char read_buffer[3];
  	int  bytes_written;  
  	int  bytes_read; 
  	struct termios options;           // Terminal options
  	int fd;                           // File descriptor for the port


	// open communication with arduino
  	fd = open("/dev/ttyUSB0",O_RDWR | O_NOCTTY);   // Open tty device for RD and WR
  	//usleep(2000000);// suspend thread (for acm device, my arduino at home) 

  	if(fd == 1) {
     		printf("\n  Error! in Opening ttyUSB0\n");
  	}
  	else
     		printf("\n  ttyACM0 Opened Successfully\n");

    	tcgetattr(fd, &options);               // Get the current options for the port
    	cfsetispeed(&options, B115200);        // Set the baud rates to 115200          
    	cfsetospeed(&options, B115200);                   
    	options.c_cflag |= (CLOCAL | CREAD);   // Enable the receiver and set local mode           
   	 options.c_cflag &= ~PARENB;            // No parity                 
    	options.c_cflag &= ~CSTOPB;            // 1 stop bit                  
    	options.c_cflag &= ~CSIZE;             // Mask data size         
    	options.c_cflag |= CS8;                // 8 bits
    	options.c_cflag &= ~CRTSCTS;           // Disable hardware flow control    

	// Enable data to be processed as raw input
	options.c_lflag &= ~(ICANON | ECHO | ISIG);
	     
    	tcsetattr(fd, TCSANOW, &options);      // Apply options immediately
    	fcntl(fd, F_SETFL, FNDELAY);    

	
	//create slider bars for HSV filtering
	createTrackbars();
	
	// dilates and erodes image (a way to get rid of "noise" when filtering by color)
	bool useMorphOps = true; 	

    	// OpenCV frame matrices
    	cv::Mat frame0, frame0_warped, frame1, frame1_warped, result, HSV, hsv_threshold, hsv_corners, threshold_top, threshold_bot, threshold_all;

    	cv::cuda::GpuMat gpu_frame0, gpu_frame0_warped, gpu_frame1, gpu_frame1_warped, gpu_grayImage0, gpu_grayImage1, gpu_differenceImage, gpu_thresholdImage;

	// ROI parameters for top and bottom paddles
    	cv::Rect roi_top(520, 66, 307, 26);
    	cv::Rect roi_bot(520, 625, 307, 26);

    	int toggle, frame_count;

	Point2i ball_pt =  Point(0,0);

	Point2i right_paddle_pt =  Point(0,0);
	

#ifdef TEST_LIVE_VIDEO

    	// Camera video pipeline
    	std::string pipeline = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

#else

    	// MP4 file pipeline
    	std::string pipeline = "filesrc location=/home/nvidia/Desktop/EE555/workspace/lab_3/build/pong_video.mp4 ! qtdemux name=demux ! h264parse ! omxh264dec ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

#endif

    	std::cout << "Using pipeline: " << pipeline << std::endl;
 
    	// Create OpenCV capture object, ensure it works.
    	cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    	if (!cap.isOpened()) {
        	std::cout << "Connection failed" << std::endl;
        return -1;
    	}

	// Input/Output Image coordinates of game: starts at upper-left, then clockwise
	Point2f inputQuad[4];
	Point2f outputQuad[4];
	
	// Lambda Matrix
	cv:: Mat lambda( 2, 4, CV_32FC1 );

	inputQuad[0] = Point2f(520, 78);
	inputQuad[1] = Point2f(880, 77);
	inputQuad[2] = Point2f(923, 668);
	inputQuad[3] = Point2f(472, 655);


	outputQuad[0] = Point2f(437, 0);
	outputQuad[1] = Point2f(842, 0);
	outputQuad[2] = Point2f(842, 719);
	outputQuad[3] = Point2f(437, 719);


	// get perspective transform matrix lambda
	lambda = cv::getPerspectiveTransform(inputQuad,outputQuad);
     
    	// Capture the first frame with GStreamer
    	cap >> frame0;

	// upload to GPU memory
    	gpu_frame0.upload(frame0);

	// warp perspective
	cv::cuda::warpPerspective(gpu_frame0,gpu_frame0_warped,lambda,gpu_frame0.size());

	// download to CPU memory 	
	gpu_frame0_warped.download(frame0_warped);

	
	// convert to HSV
	cvtColor(frame0_warped,HSV,COLOR_BGR2HSV);
	
	//filter HSV image between values and store filtered image to
	//threshold matrix
	inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),hsv_threshold);

	//perform morphological operations on thresholded image to eliminate noise
	//and emphasize the filtered object
	if(useMorphOps)
		morphOps(hsv_threshold);


	// Convert the frames to gray scale (monochrome)
    	
    	cv::cuda::cvtColor(gpu_frame0_warped,gpu_grayImage0,cv::COLOR_BGR2GRAY);

    	// Initialize 
    	toggle = 0;
    	frame_count = 0;

    	while (frame_count < 2500) {

        	if (toggle == 0) {
           		// Get a new frame from file
           		cap >> frame1;

			// upload to gpu
	   		gpu_frame1.upload(frame1);

			// Warp Perspective Transformation
			cv::cuda::warpPerspective(gpu_frame1,gpu_frame1_warped,lambda,gpu_frame1.size());
			
			// download warped frame to cpu
			gpu_frame1_warped.download(frame1_warped);	
			
			// convert to hsv colorspace
			//cvtColor(frame1_warped,HSV,COLOR_BGR2HSV);

			//filter HSV image between values and store filtered image to
			//threshold matrix
			//inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),hsv_threshold);

			//perform morphological operations on thresholded image to eliminate noise
			//and emphasize the filtered object(s)
			//if(useMorphOps)
			//morphOps(hsv_threshold);



			 // Convert the frames to gray scale (monochrome)    
			//cv::cvtColor(frame1,grayImage1,cv::COLOR_BGR2GRAY);
           		cv::cuda::cvtColor(gpu_frame1_warped,gpu_grayImage1,cv::COLOR_BGR2GRAY);
           		toggle = 1;
        	} 
        	else {
	   		cap >> frame0;
           		gpu_frame0.upload(frame0);

			// warp perspective
			cv::cuda::warpPerspective(gpu_frame0,gpu_frame0_warped,lambda,gpu_frame0.size());

			gpu_frame0_warped.download(frame0_warped);

			//cvtColor(frame0_warped,HSV,COLOR_BGR2HSV);
			//filter HSV image between values and store filtered image to
			//threshold matrix
			//inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),hsv_threshold);
			//perform morphological operations on thresholded image to eliminate noise
			//and emphasize the filtered object(s)
			//if(useMorphOps)
			//morphOps(hsv_threshold);
     	   
			// convert to grayscale
			//cv::cvtColor(frame0,grayImage0,cv::COLOR_BGR2GRAY);
           		cv::cuda::cvtColor(gpu_frame0_warped,gpu_grayImage0,cv::COLOR_BGR2GRAY);
           		toggle = 0;
		}
 
		// Compute the absolute value of the difference
		//cv::absdiff(grayImage0, grayImage1, differenceImage);
		cv::cuda::absdiff(gpu_grayImage0, gpu_grayImage1, gpu_differenceImage);


		// Threshold the difference image
		//cv::threshold(differenceImage, thresholdImage, 50, 255, cv::THRESH_BINARY);
        	cv::cuda::threshold(gpu_differenceImage, gpu_thresholdImage, 50, 255, cv::THRESH_BINARY);

	       	gpu_thresholdImage.download(result);

		// Find the location of any moving object and show the final frame
		if (toggle == 0) {
			//searchForMovement(thresholdImage,frame0);
                	ball_pt = searchForMovement(result,frame0_warped);
			
			// convert to hsv to find corners on original image
			//cvtColor(frame0, hsv_corners, COLOR_BGR2HSV);

			// filter hsv image for top/bottom paddles and store to threshold images
			// these hsv values were found by experimenting with the color filter trackbars
		    	inRange(HSV,Scalar(83,212,196),Scalar(256,256,256),threshold_top);
		    	inRange(HSV,Scalar(0,0,245),Scalar(256,89,256),threshold_bot);
			inRange(hsv_corners,Scalar(0,0,99),Scalar(256,256,256),threshold_all);
			
			// search for paddles (going to be right side player on wall, which is top on .mp4)
			right_paddle_pt = searchForPaddles(roi_top,threshold_top,frame0_warped);
            		//searchForPaddles(roi_bot,threshold_bot,frame0_warped);

			// find corners
			//findCorners(threshold_all,frame0);

			cv::line(frame0,inputQuad[0],inputQuad[1],Scalar(0,255,0),2);
			cv::line(frame0,inputQuad[1],inputQuad[2],Scalar(0,255,0),2);
			cv::line(frame0,inputQuad[2],inputQuad[3],Scalar(0,255,0),2);
			cv::line(frame0,inputQuad[3],inputQuad[0],Scalar(0,255,0),2);
			
			//show frames 
			imshow("Original", frame0);
		    	imshow(windowName2,hsv_threshold);
		    	imshow(windowName,frame0_warped);
		}

		else {
			//searchForMovement(thresholdImage,frame1);
                	ball_pt = searchForMovement(result,frame1_warped);

			// convert to hsv to find corners on original image
			//cvtColor(frame1, hsv_corners, COLOR_BGR2HSV);

			// filter hsv image for top/bottom paddles and store to threshold images
			// these hsv values were found by experimenting with the color filter trackbars
		    	inRange(HSV,Scalar(77,206,202),Scalar(256,256,256),threshold_top);
		    	inRange(HSV,Scalar(0,0,252),Scalar(256,98,256),threshold_bot);
			inRange(hsv_corners, Scalar(0,0,115), Scalar(256,256,256), threshold_all);
			
			// search for paddles and move them
			right_paddle_pt = searchForPaddles(roi_top,threshold_top,frame1_warped);
            		//searchForPaddles(roi_bot,threshold_bot,frame1_warped, pt);

			// find corners
			//findCorners(threshold_all,frame1);

			cv::line(frame1,inputQuad[0],inputQuad[1],Scalar(0,255,0),2);
			cv::line(frame1,inputQuad[1],inputQuad[2],Scalar(0,255,0),2);
			cv::line(frame1,inputQuad[2],inputQuad[3],Scalar(0,255,0),2);
			cv::line(frame1,inputQuad[3],inputQuad[0],Scalar(0,255,0),2);

	
			// show frames
			imshow("Original", frame1);
	        	//imshow("Frame", frame1_warped);
			imshow(windowName2,hsv_threshold);
		    	imshow(windowName,frame1_warped);
		}

		//cout << "ball x =  " << ball_pt.x << "	ball y =  " << ball_pt.y << endl; // testing

		// I am right player on wall

		bool serial_avail = 0;

		
		if (serial_avail == 1)
			if (ball_pt.y = 92) {// ball predicted to hit right boundary, move paddle to x position

			
				// how far away is the paddle from the predicted impact point?
				int difference = ball_pt.x - right_paddle_pt.x;

				char buffer[10];
				memset(&write_buffer[0], 0, sizeof(write_buffer));	// clear array
			
				if (difference > 0) {
					strcpy(write_buffer, "R ");// move paddle to the right
				}
				else {
					strcpy(write_buffer, "L ");// move paddle to the left
				}

				// decide on duration of pulse, depends on how far paddle is away

				int LOW_MAX = 2, MED_MAX = 5, HIGH_MAX = 10, FAST_MAX = 20;

				if (difference < 10)
					sprintf(buffer, "%d", LOW_MAX);
				else if (difference < 50 )
					sprintf(buffer, "%d", MED_MAX);	
				else if (difference < 100 )
					sprintf(buffer, "%d", HIGH_MAX);
				else if (difference < 50 )
					sprintf(buffer, "%d", FAST_MAX);

				strcat(write_buffer, buffer);	// create string of direction and duration

				bytes_written = write(fd, &write_buffer, strlen(write_buffer));

				serial_avail = 0;	

			}

		else{

			if(read(fd, &read_buffer, 1) != 1) {

				cout << "receive ack " << read_buffer[0] << endl;
				serial_avail = 1;
			}
		}		

			
		

		//show frames
	 	//resizeWindow(windowName2,200, 200);
		//imshow(windowName2,hsv_threshold);
		//resizeWindow(windowName1,200, 200);
		//imshow(windowName1,HSV);

		frame_count++;

        	cv::waitKey(1); //needed to show frame
    }
}





















