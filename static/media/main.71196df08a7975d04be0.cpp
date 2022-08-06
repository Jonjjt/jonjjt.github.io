//version 1.01
// [ TIS SDK HEADERS ]
#include "main.h"
#include "tcamimage.h"
#include "tcamprop.h"

// [ OPENCV HEADERS ]
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp" 
#include <opencv2/calib3d.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>

//[ SERVER HEADER ] (to Odrive python file)
#include "udp_client_server.h"

// [ GENERAL HEADERS ]
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <thread>
#include <ctime>
#include <algorithm>

// [ TBB HEADERS ]
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>

// [ CUDA HEADERS ]
#include <cuda_runtime.h>
#include <cuda.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudawarping.hpp>

//----[ INITIALISE GLOBAL VARIABLES ]-----------------------------------------------
//Coordinate system Calibration variables
std::ifstream calibration_file("calibration.txt");
std::vector<double> calibration_inputs;
bool CALIBRATED = true;
double x_offset;
double y_offset;
double x_diff;
double y_diff;

udp_client_server::udp_client client("127.0.1.1", 59200);

//dk yet
bool HOMOGRAPHY = false;
double framewidth;
double frameheight;

//if we want to record a video
bool recording = false;
std::vector<cv::Mat> threshVideoFeed, trackVideoFeed;
cv::VideoWriter trackVideo;

//previous velocity
float prevDx;
float prevDy;

int fixedHittingLineY = 548;
int count = 0;


//for csv recording
std::vector<std::vector<float>> csvRecord;

//  JON
//  strategy stuff
time_t puckLostTime;
bool puckLost = false;
float timeout = 1;
int prevStatus = 0; //  nothing = 0, if (shootPuck) = 1, stayGoalSide = 2, squareUp = 3, catchPuck = 4, hitPuckAway = 5, readyUp = 6 etc.
cv::Point2f prevStrikerTarget = cv::Point2f(0, 0);
// float strikerAssumedSpeed = 400;
double pi = atan(1)*4;
time_t shotClockStart;
double shotClock = 0;
bool puckInWorkspace = false;
float desiredWindUpDistance = 70;
int prevShotChoice = 3; //  straight shot = 0, right bank = 1, left bank = 2

//  Control Variables
float stepsize;
float defenseline = 450; //  blue line = 360, center line = 280, max striker range ~317, max striker pos ~337

//  Catch Performance Test
std::vector<std::vector<float>> catchInfo;
bool loggingCatch = false;
int loggingExcess = -1;
bool neededToSquareUp = false;
int catchID = 0;

//  JON END

//table dimension in camera code
int xMaxW = 250;
int xMinW = 1;
int yMaxW = 548;
int yMinW = 2;

//Stiker Velocity and Acceleration Logging
float strikerMaxVelocityX;
float strikerMaxVelocityY;
float strikerMaxAccelerationX;
float strikerMaxAccelerationY;

// -------[ INITIALISE DATA STRUCTS ]------------------------------------------


struct CameraProperties
{
  // [ GENERAL CAMERA PROPERTIES ]
  /* [NOTE]
    Any other properties may be added here, such as:
    
      - Exposure (shutter speed, auto exposure)
      - Gain (value, auto gain)
      - Format 

    Please refer to The Imaging Source documentation for getting and setting camera properties:
    https://www.theimagingsource.com/documentation/tiscamera/
  */
  const int rate = 120;                      // Frame Rate (FPS)
  const int width = 1440;                    // Width of image output (pixels)
  const int height = 1080;                   // Height of image output (pixels)
  const std::string serialNum = "25020018";  // Specific serial number for camera

  // [ IMAGE CROP PARAMETERS ]
  const int crop_x = 200;                     // x pixel-coordinate position where ROI rect is pinned (from top-left)
  const int crop_y = 200;                      // y pixel-coordinate position where ROI rect is pinned (from top-left)
  const int frameRate_n = 2500000; // Frame rate numerator
  const int frameRate_d = 10593;   // Frame rate denumerator
};

struct ProcessingChainData
{
  /* [ NOTE ]
      Structure to travel along with the pipeline.
      Contains all the local data
  */
  // [ GENERAL ]
  cv::Mat img = cv::Mat(1080, 1440, CV_8UC4);
  cv::Mat bgrImg = cv::Mat(1080, 1440, CV_8UC3);
  //cv::Mat unImg = cv::Mat(1080, 1440, CV_8UC3);

  cv::Mat cropImg,cimg, pimg, smallImg,cimage, unImg, notImg, smoothImg, biImg,
      hsvImg, threshImg, morph1Img, morph2Img, resultImg, striker_thresh, player_striker_thresh, undistorttest,trajectoryImg;

  // [ GPU CUDA ]
  // cv::cuda::GpuMat g_upImg = cv::cuda::GpuMat(1080, 1440, CV_8UC3); 
  // cv::cuda::GpuMat g_unImg = cv::cuda::GpuMat(1080, 1440, CV_8UC3); 
  cv::cuda::GpuMat g_upImg, g_unImg, src_gpu, warp_gpu;

  // [ TIME KEEP ]
  //double cap_T, get_T, cv_T, up_T, un_T, dn_T, cr_T, re_T, in_T, sm_T, hsv_T, th_T, mo1_T, mo2_T, se_T, vi_T, _kal,hom_T,segpuck_T,seghand_T; 
  //double _kal
  // [ SEGMENT - PUCK ]
  std::vector<std::vector<cv::Point>> contours;
  cv::Point2f center; 
  float objRadius;
  float dx;
  float dy;

  // [ SEGMENT - STRIKER ]
  std::vector<std::vector<cv::Point>> strikercontours;
  cv::Point2f strikercenter; 
  float strikerobjRadius;
  float strikerDx;
  float strikerDy;

  // [ SEGMENT - PLAYER STRIKER ]
  std::vector<std::vector<cv::Point>> playerstrikercontours;
  cv::Point2f playerstrikercenter;
  float playerstrikerobjRadius; 

  //  [Strategy]
  std::vector<cv::Point2f> puckBounces;
};

// -------[ FUNCTIONS ]------------------------------------------
//declare functions
float checkTime(std::vector<cv::Point2f> vertices, float currentSpeed,int iteration);
bool checkBounce(cv::Point2f currentPos, cv::Point2f &currentSpeedOut, cv::Point2f &newPos, int hittingLineY);
void moveStriker(cv::Point2f cord, cv::Point2f strikerCenter);
cv::Point2f detectstriker(cv::Mat &mat);
void startPipeline();
void imageProcessingTBB(TcamImage &cam,
                        tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue, 
                        cv::cuda::GpuMat map1, cv::cuda::GpuMat map2);

//trajectory prediction
//return a vector containing coordinates of bounces
std::vector<cv::Point2f> predictTrajectoryMultiplePoints(cv::Mat &mat, cv::Point2f originPoint, float radius, float dx, float dy)
{   
  //std::cout<<"t0 "<<std::endl;
  int point_count =5;
  std::vector<cv::Point2f> vertices(point_count);
  cv::Point2f currentSpeed = cv::Point2f(dx,dy);
  cv::Point2f newVertice = originPoint;
  bool hasTouchHittingLine = false;
  int iteration = 0;
  vertices[iteration] = originPoint;
  float time = 0;
  //std::cout<<"current speed:  "<<dx << " "<<dy <<std::endl;
  //try to figure out the vectors after (point_count) of bounces
  if(abs(dx)>5&&abs(dy)>5)
  {
    while (true)
    { 
      if (iteration == point_count-1)
      {   

          break;
      }

      //std::cout<<"t1 current speed "<<currentSpeed.x << " "<<currentSpeed.y<<std::endl;
      if(checkBounce(vertices[iteration],currentSpeed, newVertice,fixedHittingLineY))
      {
          iteration++;
          vertices[iteration] = cv::Point2f(newVertice.x, newVertice.y);
          iteration++;
          hasTouchHittingLine = true;
          break;
      }

    
      iteration++;
      vertices[iteration] = cv::Point2f(newVertice.x, newVertice.y);
      
     

    }

    // std::cout<<"t2 "<<std::endl;
    time = checkTime(vertices,sqrt(pow(currentSpeed.x,2)+pow(currentSpeed.y,2)),iteration);

    for(int i=0;i<iteration-1;i++)
    {
      //std::cout<<"t4 ["<<vertices[i].x<<","<<vertices[i].y<<"]+["<<vertices[i+1].x<<","<<vertices[i+1].y<<"]"<<std::endl;
      cv::line(mat, vertices[i], vertices[i+1], cv::Scalar(0,255, 0), 2);
    }
  
  }
  std::vector<cv::Point2f> puckBounces;
  puckBounces.push_back(cv::Point2f(vertices[1]));
  for (int i = 1; i < point_count + 1; i++)
  {
    if (iteration - 1 > i)
    {
      puckBounces.push_back(cv::Point2f(vertices[i+1]));
    }
    else
    {
      return puckBounces;
    }
  }
  return puckBounces;
}

float checkTime(std::vector<cv::Point2f> vertices, float currentSpeed,int iteration)
{
  float TotalDistance = 0;
  for(int i = 0; i < iteration- 1; i++)
  {
      TotalDistance += sqrt(pow((vertices[1 + i] - vertices[i]).x,2)+pow((vertices[1 + i] - vertices[i]).y,2));
  }
  // std::cout<<"t3 "<<std::endl;

  float predictedTime = TotalDistance / currentSpeed;
  //Debug.Log(predictedTime);
  return predictedTime;
}

//return true when hit the hitting line
bool checkBounce(cv::Point2f currentPos, cv::Point2f &currentSpeedOut, cv::Point2f &newPos, int hittingLineY)
{ 
  //1.8 -1.8 4 -4
  // 250, 1, 548, 2 
  float radius = 9.45f;
  float xMaxPuck = 250.0f-radius;       // X limit to the right
  float xMinPuck = 1.0f+radius;       // X limit to the left
  float yMaxPuck = 548.0f-radius;       // Y limit to the bottom 
  float yMinPuck = 2.0f+radius;       // Y limit to the top 
  cv::Point2f currentSpeed = cv::Point2f(currentSpeedOut.x,currentSpeedOut.y);
  //to the right
  if (currentSpeed.x > 0)
  {   
      //up direction
      if(currentSpeed.y < 0)
      {
          if((xMaxPuck-currentPos.x)*currentSpeed.y/currentSpeed.x > currentPos.y-yMinPuck)
          {   
              //currentSpeedOut = cv::Point2f(0, 0);
              currentSpeedOut = currentSpeed;
              //currentSpeedOut = cv::Point2f(0, 0);
              newPos = cv::Point2f((yMinPuck-currentPos.y)/currentSpeed.y*currentSpeed.x+currentPos.x, yMinPuck);
              return true;
          }
          else
          {
              currentSpeedOut = cv::Point2f(-currentSpeed.x, currentSpeed.y);
              newPos = cv::Point2f(xMaxPuck, (xMaxPuck - currentPos.x) / currentSpeed.x * currentSpeed.y + currentPos.y);
              return false;

          }
      }
      //down direction
      else
      {
          if ((xMaxPuck - currentPos.x) * currentSpeed.y / currentSpeed.x > -currentPos.y +hittingLineY)
          {
              //currentSpeedOut = cv::Point2f(0, 0);
              currentSpeedOut = currentSpeed;
              newPos = cv::Point2f(-( currentPos.y-hittingLineY) / currentSpeed.y * currentSpeed.x + currentPos.x, hittingLineY);
              return true;
          }
          else
          { 
              //std::cout<<currentSpeed.y<<std::endl;
              currentSpeedOut = cv::Point2f(-currentSpeed.x, currentSpeed.y);
              newPos = cv::Point2f(xMaxPuck, currentPos.y + (xMaxPuck - currentPos.x) / currentSpeed.x * currentSpeed.y);
              return false;
          }
      }

  }
  //to the left
  else
  {
      //up direction
      if (currentSpeed.y < 0)
      {
          if (( currentPos.x-xMinPuck) * currentSpeed.y / currentSpeed.x > -yMinPuck + currentPos.y)
          {
              //currentSpeedOut = cv::Point2f(0, 0);
              currentSpeedOut = currentSpeed;
              newPos =cv::Point2f((yMinPuck- currentPos.y) / currentSpeed.y * currentSpeed.x + currentPos.x, yMinPuck);
              return true;
          }
          else
          {
              currentSpeedOut = cv::Point2f(-currentSpeed.x, currentSpeed.y);
              newPos = cv::Point2f(xMinPuck, (currentPos.x-xMinPuck) / -currentSpeed.x * currentSpeed.y + currentPos.y);
              return false;

          }
      }
      //down direction
      else
      {
          if (-(currentPos.x-xMinPuck) * currentSpeed.y / currentSpeed.x > -currentPos.y +hittingLineY)
          {
              currentSpeedOut = currentSpeed;
              newPos = cv::Point2f(currentPos.x-(currentPos.y-hittingLineY) / currentSpeed.y * currentSpeed.x , hittingLineY);
              return true;
          }
          else
          { 
              //std::cout<<"pos y "<< currentPos.y <<"speed x "<< currentSpeed.x <<"speed y "<< currentSpeed.y<<"dif x "<< ( currentPos.x-xMinPuck) <<std::endl;
              currentSpeedOut = cv::Point2f(-currentSpeed.x, currentSpeed.y);
              newPos = cv::Point2f(xMinPuck, currentPos.y -( currentPos.x-xMinPuck) / currentSpeed.x * currentSpeed.y);
              return false;
          }
      }
  }
}

//  Jon

//GENERAL
/*  function to move striker in direction of target coordinate
    converts target coordinate into short step size version to mimic velocity control
    update stepsize before calling - else stepsize = distance to travel/2
*/
void moveStriker(cv::Point2f cord, cv::Point2f strikerCenter)
{
  float x = cord.x - strikerCenter.x;
  float y = cord.y - strikerCenter.y;
  float c = sqrt(pow(x,2) + pow(y, 2));
  float x_angle = asin(x/c);
  float y_angle = asin(y/c);
  
  if (stepsize == 0)
  {
    stepsize = c/2;
  }
  if (x == 0 || y == 0)
  {
    return;
  }

  cord.x = strikerCenter.x + sin(x_angle) * stepsize;
  cord.y = strikerCenter.y + sin(y_angle) * stepsize;
  // std::cout<<x<<std::endl;
  // std::cout<<y<<std::endl;
  // std::cout<<c<<std::endl;
  // std::cout<<x_angle<<std::endl;
  // std::cout<<y_angle<<std::endl;
  //std::cout<<"coords stepsized (moveStriker): "<<cord.x<<", "<<cord.y<<std::endl;

  cv::Point2f translated_cord;

  cv::Point2f translated_strikercenter;

  translated_cord.x = (cord.x - x_offset)/x_diff;
  translated_cord.y = (cord.y - y_offset)/y_diff;
  translated_cord.x = std::floor(translated_cord.x * 100)/100;
  translated_cord.y = std::floor(translated_cord.y * 100)/100;


  if (translated_cord.x > 1 || translated_cord.y > 1)
  {
    if (translated_cord.y > 1)
    {
      translated_cord.y = 1.003400001;
      translated_cord.y = std::floor(translated_cord.y * 100)/100;

    }
    if (translated_cord.x > 1)
    {
      translated_cord.x = 1.003400001;
      translated_cord.x = std::floor(translated_cord.x * 100)/100;
    }
  } 

  if (translated_cord.x < 0 || translated_cord.y < 0) 
  {
    if (translated_cord.y < 0)
    {
      translated_cord.y = 0.0034300001;
      translated_cord.y = std::floor(translated_cord.y * 100)/100;
    }
    if (translated_cord.x < 0)
    {
      translated_cord.x = 0.0034300001;
      translated_cord.x = std::floor(translated_cord.x * 100)/100;
    }
  } 

  translated_strikercenter.x = (strikerCenter.x - x_offset)/x_diff;
  translated_strikercenter.y = (strikerCenter.y - y_offset)/y_diff;
  translated_strikercenter.x = std::floor(translated_strikercenter.x * 100)/100;
  translated_strikercenter.y = std::floor(translated_strikercenter.y * 100)/100;


  if (translated_strikercenter.x > 1 || translated_strikercenter.y > 1)
  {
    if (translated_strikercenter.y > 1)
    {
      translated_strikercenter.y = 1.003400001;
      translated_strikercenter.y = std::floor(translated_strikercenter.y * 100)/100;

    }
    if (translated_strikercenter.x > 1)
    {
      translated_strikercenter.x = 1.003400001;
      translated_strikercenter.x = std::floor(translated_strikercenter.x * 100)/100;
    }
  } 

  if (translated_strikercenter.x < 0 || translated_strikercenter.y < 0) 
  {
    if (translated_strikercenter.y < 0)
    {
      translated_strikercenter.y = 0.0034300001;
      translated_strikercenter.y = std::floor(translated_strikercenter.y * 100)/100;
    }
    if (translated_strikercenter.x < 0)
    {
      translated_strikercenter.x = 0.0034300001;
      translated_strikercenter.x = std::floor(translated_strikercenter.x * 100)/100;
    }
  } 

  std::stringstream cords_to_send;
  cords_to_send << "(" << translated_cord.x << "," << translated_cord.y << "," << translated_strikercenter.x << "," << translated_strikercenter.y << ")";

  std::string s = cords_to_send.str();
  int n = s.length();
  char char_array[n+1];
  std::strcpy(char_array, s.c_str());
  // // std::cout<<"hello"<<std::endl;
  //std::cout<<"sent coords (moveStriker): "<<char_array<<std::endl;

  client.send(char_array, s.length());
  stepsize = 0;
}

/*  function to move striker in direction of target coordinate
    checks collision with puck, and remaps to avoid puck
    converts target coordinate into short step size version to mimic velocity control
    update stepsize before calling - else stepsize = distance to travel
*/
void moveStrikerAvoidPuck(cv::Point2f cord, cv::Point2f strikerCenter, float strikerRadius, cv::Point2f puckCenter, float puckRadius, float dx, float dy, cv::Mat &mat)
{
  float x = cord.x - strikerCenter.x;
  float y = cord.y - strikerCenter.y;
  float c = sqrt(pow(x,2) + pow(y, 2));
  // float x_angle = asin(x/c);
  // float y_angle = asin(y/c);
  float x_angle = atan2(x, y);
  float y_angle = atan2(y, x);

  float puckSpeed = sqrt(pow(dx, 2) + pow(dy, 2));

  //  check for collision with puck
  bool collide = false;
  for (int i = 0; i < c; i++) //  for i points on the striker trajectory
  {
    float x_new = strikerCenter.x + sin(x_angle) * i; //  coordinates for potential future striker position
    float y_new = strikerCenter.y + sin(y_angle) * i;
    int distSq = pow(x_new - puckCenter.x, 2) + pow(y_new - puckCenter.y, 2);
    int radSumSq = pow(puckRadius + strikerRadius, 2);
    
    if (distSq < radSumSq) //  touching or intersecting
    {
      collide = true;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }
  
  std::cout<<"avoid1"<<std::endl;
  if (collide)
  {
    x = puckCenter.x - strikerCenter.x;
    y = puckCenter.y - strikerCenter.y;
    c = sqrt(pow(x,2) + pow(y, 2));
    x_angle = atan2(x, y);
    y_angle = atan2(y, x);
    //  remap path
    //  calculate angle change needed to avoid puck
    float minSeparation = puckRadius + strikerRadius + 20;
    float puckStrikerDistance = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y - strikerCenter.y, 2));
    float yAngleChange;
    if (puckStrikerDistance > minSeparation)
    {
      yAngleChange = asin(minSeparation/puckStrikerDistance);

      std::cout<<puckRadius<<std::endl;
      std::cout<<strikerRadius<<std::endl;
      std::cout<<minSeparation<<std::endl;
      std::cout<<yAngleChange<<std::endl;

      //  check positive and negative angle change
      float posNewYAngle = y_angle + yAngleChange;
      float negNewYAngle = y_angle - yAngleChange;
      float posNewXAngle = pi/2 - posNewYAngle;
      float negNewXAngle = pi/2 - negNewYAngle;

      std::cout<<posNewXAngle<<std::endl;
      std::cout<<negNewXAngle<<std::endl;

      //  check which side colliding from??
      if (x_angle > y_angle)  //  if horizontal collision -- x_angle larger than y_angle
      {
        //  use x val of puck
        float posY = puckCenter.x * sin(posNewYAngle)/sin(posNewXAngle);
        if (isnan(posY))
        {
          posY = puckCenter.y;
        }
        float negY = puckCenter.x * sin(negNewYAngle)/sin(negNewXAngle);
        if (isnan(negY))
        {
          negY = puckCenter.y;
        }
        //  use the larger Y val (goalside)
        if (posY > negY)
        {
          x = puckCenter.x - strikerCenter.x;
          y = posY - strikerCenter.y;
          c = sqrt(pow(x, 2) + pow(y, 2));
          x_angle = posNewXAngle;
          y_angle = posNewYAngle;
        }
        else
        {
          x = puckCenter.x - strikerCenter.x;
          y = negY - strikerCenter.y;
          c = sqrt(pow(x, 2) + pow(y, 2));
          x_angle = negNewXAngle;
          y_angle = negNewYAngle;
        }
      }
      else  //  if vertical collision -- y_angle larger than x_angle
      {
        //  use y val of puck
        float posX = puckCenter.y * sin(posNewXAngle)/sin(posNewYAngle);
        if (isnan(posX))
        {
          posX = puckCenter.x;
        }
        float negX = puckCenter.y * sin(negNewXAngle)/sin(negNewYAngle);
        if (isnan(negX))
        {
          negX = puckCenter.x;
        }
        if (abs(dx) < 50)
        {
          //  use the X val closer to 122 (the center/goalside)
          if (abs(122 - posX) < abs(122 - negX))
          {
            x = posX - strikerCenter.x;
            y = puckCenter.y - strikerCenter.y;
            c = sqrt(pow(x, 2) + pow(y, 2));
            x_angle = posNewXAngle;
            y_angle = posNewYAngle;
          }
          else
          {
            x = negX - strikerCenter.x;
            y = puckCenter.y - strikerCenter.y;
            c = sqrt(pow(x, 2) + pow(y, 2));
            x_angle = negNewXAngle;
            y_angle = negNewYAngle;
          }
        }
        else
        {
          //  use the X val opposite direction of dx
          if (puckCenter.x * dx - puckCenter.x > 0 && posX - puckCenter.x < 0)
          {
            x = posX - strikerCenter.x;
            y = puckCenter.y - strikerCenter.y;
            c = sqrt(pow(x, 2) + pow(y, 2));
            x_angle = posNewXAngle;
            y_angle = posNewYAngle;
          }
          else
          {
            x = negX - strikerCenter.x;
            y = puckCenter.y - strikerCenter.y;
            c = sqrt(pow(x, 2) + pow(y, 2));
            x_angle = negNewXAngle;
            y_angle = negNewYAngle;
          }
        }
      }
    }
    else  //  if striker is already inside the minimum distance from puck
    {
      yAngleChange = pi;
      y_angle += yAngleChange;
      x_angle = pi/2 - y_angle;
      x = sin(y_angle) * minSeparation - strikerCenter.x;
      y = sin(x_angle) * minSeparation - strikerCenter.y;
      c = sqrt(pow(x, 2) + pow(y, 2));
      stepsize = minSeparation;

      cv::putText(mat, std::to_string(x_angle), cv::Point2f(60,300), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      cv::putText(mat, std::to_string(y_angle), cv::Point2f(60,320), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      cv::putText(mat, std::to_string(x), cv::Point2f(60,340), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      cv::putText(mat, std::to_string(y), cv::Point2f(60,360), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

      std::cout<<yAngleChange<<std::endl;
      std::cout<<y_angle<<std::endl;
      std::cout<<x_angle<<std::endl;
    }
  }
   
  std::cout<<"avoid2"<<std::endl;

  if (stepsize == 0)
  {
    stepsize = c;
  }
  // if (x == 0 || y == 0)
  // {
  //   // return;
  // }
  
  cord.x = strikerCenter.x + sin(x_angle) * stepsize;
  cord.y = strikerCenter.y + sin(y_angle) * stepsize;

  //  visualise trajectory
  // cv::line(mat, strikerCenter, cv::Point2f(x + strikerCenter.x, y + strikerCenter.y), cv::Scalar(0,255,255), 2);
  // cv::circle(mat, cv::Point2f(x + strikerCenter.x, y + strikerCenter.y), strikerRadius, cv::Scalar(0,255,255), 2);
  cv::line(mat, strikerCenter, cv::Point2f(cord.x, cord.y), cv::Scalar(0,255,255), 2);
  cv::circle(mat, cv::Point2f(cord.x, cord.y), strikerRadius, cv::Scalar(0,255,255), 2);
  // cv::putText(mat, "scuffed", cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
  // cv::putText(mat, std::to_string(cord.x), cv::Point2f(60,110), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
  // cv::putText(mat, std::to_string(cord.y), cv::Point2f(60,130), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

  std::cout<<"coordinates: "<<cord.x<<", "<<cord.y<<std::endl;

  cv::Point2f translated_cord;

  cv::Point2f translated_strikercenter;

  translated_cord.x = (cord.x - x_offset)/x_diff;
  translated_cord.y = (cord.y - y_offset)/y_diff;
  translated_cord.x = std::floor(translated_cord.x * 100)/100;
  translated_cord.y = std::floor(translated_cord.y * 100)/100;


  if (translated_cord.x > 1 || translated_cord.y > 1)
  {
    if (translated_cord.y > 1)
    {
      translated_cord.y = 1.003400001;
      translated_cord.y = std::floor(translated_cord.y * 100)/100;

    }
    if (translated_cord.x > 1)
    {
      translated_cord.x = 1.003400001;
      translated_cord.x = std::floor(translated_cord.x * 100)/100;
    }
  } 

  if (translated_cord.x < 0 || translated_cord.y < 0) 
  {
    if (translated_cord.y < 0)
    {
      translated_cord.y = 0.0034300001;
      translated_cord.y = std::floor(translated_cord.y * 100)/100;
    }
    if (translated_cord.x < 0)
    {
      translated_cord.x = 0.0034300001;
      translated_cord.x = std::floor(translated_cord.x * 100)/100;
    }
  } 

  translated_strikercenter.x = (strikerCenter.x - x_offset)/x_diff;
  translated_strikercenter.y = (strikerCenter.y - y_offset)/y_diff;
  translated_strikercenter.x = std::floor(translated_strikercenter.x * 100)/100;
  translated_strikercenter.y = std::floor(translated_strikercenter.y * 100)/100;


  if (translated_strikercenter.x > 1 || translated_strikercenter.y > 1)
  {
    if (translated_strikercenter.y > 1)
    {
      translated_strikercenter.y = 1.003400001;
      translated_strikercenter.y = std::floor(translated_strikercenter.y * 100)/100;

    }
    if (translated_strikercenter.x > 1)
    {
      translated_strikercenter.x = 1.003400001;
      translated_strikercenter.x = std::floor(translated_strikercenter.x * 100)/100;
    }
  } 

  if (translated_strikercenter.x < 0 || translated_strikercenter.y < 0) 
  {
    if (translated_strikercenter.y < 0)
    {
      translated_strikercenter.y = 0.0034300001;
      translated_strikercenter.y = std::floor(translated_strikercenter.y * 100)/100;
    }
    if (translated_strikercenter.x < 0)
    {
      translated_strikercenter.x = 0.0034300001;
      translated_strikercenter.x = std::floor(translated_strikercenter.x * 100)/100;
    }
  } 

  std::stringstream cords_to_send;
  cords_to_send << "(" << translated_cord.x << "," << translated_cord.y << "," << translated_strikercenter.x << "," << translated_strikercenter.y << ")";

  std::string s = cords_to_send.str();
  int n = s.length();
  char char_array[n+1];
  std::strcpy(char_array, s.c_str());
  // std::cout<<"hello"<<std::endl;
  std::cout<<"sent coords: "<<char_array<<std::endl;

  client.send(char_array, s.length());
  stepsize = 0;
}

/*  function to get strikerAssumedSpeed
*/
float getStrikerAssumedSpeed(cv::Point2f targetPoint, cv::Point2f strikerCenter)
{
  float distance = sqrt(pow(targetPoint.x - strikerCenter.x, 2) + pow(targetPoint.y - strikerCenter.y, 2));
  return 100 * pow(distance/100, 2) + 400;
}

/*  function to get position of puck after t time
*/
cv::Point2f getPuckFutureCenter(cv::Point2f puckCenter, float puckRadius, float dx, float dy, float time)
{
  float puckSpeed = sqrt(pow(dx, 2) + pow(dy, 2));
  //  set puck position bounds
  float xMaxPuck = 250 - puckRadius;
  float xMinPuck = 1 + puckRadius;
  float yMaxPuck = 548 - puckRadius;
  float yMinPuck = 2 + puckRadius;
  //  initialise variables to store puckFutureCenter
  cv::Point2f puckFutureCenter;

  if (puckCenter.x + dx*time <= xMaxPuck && puckCenter.x + dx*time >= xMinPuck) //  if puckFutureCenter.x is in bounds
  {
    puckFutureCenter.x = puckCenter.x + dx*time;
  }
  else  //  if puckFutureCenter.x exceeds bounds
  {
    //  find how much exceeded by and put in opposite direction
    if (dx > 0)
    {
      puckFutureCenter.x = xMaxPuck - abs(puckCenter.x + dx*time - xMaxPuck);
    }
    else
    {
      puckFutureCenter.x = xMinPuck + abs(puckCenter.x + dx*time - xMinPuck);
    }
  }

  if (puckCenter.y + dy*time <= yMaxPuck && puckCenter.y + dy*time >= yMinPuck) //  if puckFutureCenter.y is in bounds
  {
    puckFutureCenter.y = puckCenter.y + dy*time;
  }
  else  //  if puckFutureCenter.y exceeds bounds
  {
    //  find how much exceeded by and put in opposite direction
    if (dy > 0)
    {
      puckFutureCenter.y = yMaxPuck - abs(puckCenter.y + dy*time - yMaxPuck);
    }
    else
    {
      puckFutureCenter.y = yMinPuck + abs(puckCenter.y + dy*time - yMinPuck);
    }
  }

  return puckFutureCenter;  
}

//temp function
void gotoPuck(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, float dx, float dy)
{
  float x = puckCenter.x - strikerCenter.x;
  float y = puckCenter.y - strikerCenter.y;
  float c = sqrt(pow(x,2) + pow(y, 2));
  stepsize = c;
  moveStriker(cv::Point2f(puckCenter.x, puckCenter.y - puckRadius), strikerCenter);
}

//DEFENSE
/*  function to check if puck is heading towards robot goal (x = 91, x = 154, y = 548) --- Camelot's values were (x = 90, x = 165)
    returns true if it is
*/
bool puckHeadingToGoal(cv::Point2f puckCenter, float puckRadius, cv::Point2f puckFutureCenter)
{
  float x = puckCenter.x - puckFutureCenter.x;
  float y = puckCenter.y - puckFutureCenter.y;
  float c = sqrt(pow(x,2) + pow(y, 2));
  float x_angle = asin(x/c);  // Sin law
  float y_angle = asin(y/c);

  float x_new = puckCenter.x - (sin(x_angle) * (puckCenter.y - 548) / sin(y_angle));

  if (x_new + puckRadius > 90 && x_new - puckRadius < 165)
  {
    return true;
  }
  else
  {
    return false;
  }
}

/*  function to check if striker is "squared up" with the puck trajectory (in the path of the puck trajectory)
    DO NOT CALL IF PUCK SPEED IS VERY LOW
    returns true if it is
*/
bool squaredUp(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, cv::Point2f puckFutureCenter)
{    
  float x = puckFutureCenter.x - puckCenter.x;
  float y = puckFutureCenter.y - puckCenter.y;
  float c = sqrt(pow(x,2) + pow(y, 2));
  float x_angle = asin(x/c);  // Sin law
  float y_angle = asin(y/c);

  for (int i = 1; i < c; i++) //  for i points on the trajectory, maybe find the first/second bounce number idk
  {
    float x_new = puckCenter.x + sin(x_angle) * i; //  coordinates for potential future puck position
    float y_new = puckCenter.y + sin(y_angle) * i;
    int distSq = (x_new - strikerCenter.x) * (x_new - strikerCenter.x) + (y_new - strikerCenter.y) * (y_new - strikerCenter.y);
    int radSumSq = (puckRadius + strikerRadius) * (puckRadius + strikerRadius);
    
    if (distSq + puckRadius/3 < radSumSq) //  touching or intersecting (add small buffer of puckRadius/3)
    {
      return true;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }
  return false;
}

/*  function that squares the striker up with puck trajectory
    DO NOT CALL IF PUCK SPEED IS VERY LOW
    return true on success.
*/
bool squareUp(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, std::vector<cv::Point2f> puckBounces, float dx, float dy, cv::Mat &mat)
{
  //  find the first bounce point where the puck is in the robot workspace
  int iYInWorkspace = -1;
  std::cout<<"1"<<std::endl;
  for (int i = 0; i < puckBounces.size(); i++)  //  iterate through bounce points
  {
    if (puckBounces[i].y > 337) //  find first bounce point in robot workspace
    {
      iYInWorkspace = i;
      break;
    }
  }
  if (iYInWorkspace != -1)  //  if a bounce point is in robot workspace
  {
    cv::Point2f initialPoint;
    cv::Point2f finalPoint;
    if (iYInWorkspace == 0)
    {
      initialPoint = puckCenter;
      finalPoint = puckBounces[iYInWorkspace];
    }
    else if (iYInWorkspace == puckBounces.size() - 1)
    {
      initialPoint = puckBounces[iYInWorkspace-1];
      finalPoint = puckBounces[iYInWorkspace];
    }
    else
    {
     initialPoint = puckBounces[iYInWorkspace];
     finalPoint = puckBounces[iYInWorkspace+1]; 
    }
    float x = finalPoint.x - initialPoint.x;
    float y = finalPoint.y - initialPoint.y;
    float c = sqrt(pow(x,2) + pow(y, 2));
    float x_angle = asin(x/c);  // Sin law
    float y_angle = asin(y/c);
    
    std::vector<float> distancesToSquareUp;
    std::vector<cv::Point2f> squareUpPoints;

    float squareUpX;
    float squareUpY;
    //  for n points find closest distance?
    //  dy tells direction to be in trajectory of (y coord (initial/final) to be greater than or less than of)
    std::cout<<"2"<<std::endl;
    std::cout<<"y: "<<y<<", puckRadius: "<<puckRadius<<std::endl;
    for (int i = 0; i < y - (puckRadius + strikerRadius); i++)
    {
      if (dy > 0)
      {
        squareUpY = initialPoint.y + puckRadius + strikerRadius + i;
      }
      else
      {
        squareUpY = initialPoint.y - puckRadius - strikerRadius - i;
      }
      if (squareUpY < 337)
      {
        continue;
      }
      squareUpX = finalPoint.x - (sin(x_angle) * (finalPoint.y - squareUpY) / sin(y_angle));
      float squareUpDistance = sqrt(pow(squareUpX - strikerCenter.x, 2) + pow(squareUpY - strikerCenter.y, 2));
      distancesToSquareUp.push_back(squareUpDistance);
      squareUpPoints.push_back(cv::Point2f(squareUpX, squareUpY));
      std::cout<<"2.2"<<std::endl;
      // std::cout<<initialPoint.x<<std::endl;
      // std::cout<<squareUpX<<", "<<squareUpY<<std::endl;
    }
    std::cout<<"2.1"<<std::endl;
    float shortestSquareUpDistance;

    if (distancesToSquareUp.size() == 0)
    {
      return false;
    }
    if (distancesToSquareUp.size() == 1)
    {
      shortestSquareUpDistance = distancesToSquareUp[0];
    }
    else
    {
      shortestSquareUpDistance = *std::min_element(distancesToSquareUp.begin(), distancesToSquareUp.end());
    }
    std::cout<<"3"<<std::endl;
    std::vector<float>::iterator itIntersect = find(distancesToSquareUp.begin(), distancesToSquareUp.end(), shortestSquareUpDistance);  //  find index of intersect point with shortest distance
    int iIntersect = std::distance(distancesToSquareUp.begin(), itIntersect);
    cv::Point2f intersectPoint = squareUpPoints[iIntersect];  // find the nearest intersect point

    //  check time for puck to reach intersect point against time for striker to reach intersect point
    float puckSpeed = sqrt(pow(dx, 2) + pow(dy, 2));
    float puckTimeToIntersect = sqrt(pow(intersectPoint.x - puckCenter.x, 2) + pow(intersectPoint.y - puckCenter.y, 2))/puckSpeed;  //  t = d/s
    float strikerAssumedSpeed = getStrikerAssumedSpeed(intersectPoint, strikerCenter);
    cv::putText(mat, std::to_string(strikerAssumedSpeed), cv::Point2f(30,500), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    float strikerTimeToIntersect = sqrt(pow(intersectPoint.x - strikerCenter.x, 2) + pow(intersectPoint.y - strikerCenter.y, 2))/strikerAssumedSpeed;
    while (puckTimeToIntersect < strikerTimeToIntersect)
    {
      distancesToSquareUp.erase(itIntersect);
      squareUpPoints.erase(find(squareUpPoints.begin(), squareUpPoints.end(), squareUpPoints[iIntersect]));
      if (distancesToSquareUp.size() == 0)
      {
        return false;
      }
      if (distancesToSquareUp.size() == 1)
      {
        shortestSquareUpDistance = distancesToSquareUp[0];
      }
      else
      {
        shortestSquareUpDistance = *std::min_element(distancesToSquareUp.begin(), distancesToSquareUp.end());
      }
      std::cout<<"3.1"<<std::endl;
      itIntersect = find(distancesToSquareUp.begin(), distancesToSquareUp.end(), shortestSquareUpDistance);  //  find index of intersect point with shortest distance
      iIntersect = std::distance(distancesToSquareUp.begin(), itIntersect);
      intersectPoint = squareUpPoints[iIntersect];  // find the nearest intersect point
      puckTimeToIntersect = sqrt(pow(intersectPoint.x - puckCenter.x, 2) + pow(intersectPoint.y - puckCenter.y, 2))/puckSpeed;
      float strikerAssumedSpeed = getStrikerAssumedSpeed(intersectPoint, strikerCenter);
      cv::putText(mat, std::to_string(strikerAssumedSpeed), cv::Point2f(30,500), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      strikerTimeToIntersect = sqrt(pow(intersectPoint.x - strikerCenter.x, 2) + pow(intersectPoint.y - strikerCenter.y, 2))/strikerAssumedSpeed;
    }


    std::cout<<"4"<<std::endl;
    //  display planned striker trajectory
    cv::line(mat, strikerCenter, intersectPoint, cv::Scalar(0,0,255), 2);
    cv::circle(mat, intersectPoint, strikerRadius, cv::Scalar(0,0,255), 2);
    cv::putText(mat, "squaring up", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

    stepsize = shortestSquareUpDistance;
    moveStrikerAvoidPuck(intersectPoint, strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
    prevStatus = 3;
    return true;
  }
  else  //  if no bounce points are in robot workspace
  {
    std::cout << "cannot square up" << std::endl;
    return false;
  }
}

/*  DEPRECATED DO NOT CALL
    function that finds the nearest point of intersection (perpendicular lines) for striker's current position with puck trajectory and tries to square up
    return true on success.
    returns false if cannot square up using perpendicular points
*/
bool squareUpOld(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, std::vector<cv::Point2f> puckBounces, float dx, float dy, cv::Mat &mat)
{
  std::vector<cv::Point2f> squareUpPoints;
  std::vector<float> distancesToSquareUp;
  std::vector<float> puckTravel;  //  distances between puck bounce points
  int iIntersected = -1;

  for (int i = 0; i < puckBounces.size(); i++)
  {
    float x;
    float y;
    bool intersect;
    if (i == 0)
    {
      x = puckBounces[i].x - puckCenter.x;
      y = puckBounces[i].y - puckCenter.y;
      intersect = squaredUp(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces[i]);
    }
    else
    {
      x = puckBounces[i].x - puckBounces[i-1].x;
      y = puckBounces[i].y - puckBounces[i-1].y;
      intersect = squaredUp(puckBounces[i], puckRadius, strikerCenter, strikerRadius, puckBounces[i-1]);
    }
    if (intersect)
    {
      iIntersected = i;
    }
    float c = sqrt(pow(x,2) + pow(y, 2));
    puckTravel.push_back(c);
    float x_angle = asin(x/c);  // Sin law
    float y_angle = asin(y/c);

    // find trajectory of striker perpendicular to trajectory of puck
    float yInterceptTrajectory = puckBounces[i].y - ((y/x) * puckBounces[i].x);
    float yInterceptPerpendicular = strikerCenter.y - ((-x/y) * strikerCenter.x);
    float perpendicularX = (yInterceptTrajectory - yInterceptPerpendicular) / ((-x/y) - (y/x));
    float perpendicularY = (-x/y) * perpendicularX + yInterceptPerpendicular; 
    cv::Point2f perpendicularPoint = cv::Point2f(perpendicularX, perpendicularY);
    cv::line(mat, strikerCenter, perpendicularPoint, cv::Scalar(255,0,0), 5);
    squareUpPoints.push_back(perpendicularPoint);

    float squareUpDistanceX = strikerCenter.x - perpendicularX;
    float squareUpDistanceY = strikerCenter.y - perpendicularY;
    float squareUpDistance = sqrt(pow(squareUpDistanceX, 2) + pow(squareUpDistanceY, 2));
    // std::cout<<squareUpDistance<<std::endl;
    distancesToSquareUp.push_back(squareUpDistance);
  }

  if (iIntersected != -1)  //  if striker is already square with the puck trajectory at any point
  {
    std::cout<<"already square"<<std::endl;
    float smallestIntersectDistance = distancesToSquareUp[iIntersected];
    cv::Point2f intersectPoint = squareUpPoints[iIntersected];
    stepsize = smallestIntersectDistance/2;
    // stepsize = 10;
    moveStriker(intersectPoint, strikerCenter);
    return true;
  }
  else  //  if striker is not square with any part of puck trajectory
  {
    float smallestIntersectDistance = *std::min_element(distancesToSquareUp.begin(), distancesToSquareUp.end()); //  find the smallest distance for striker to travel to intersect with puck
    int iIntersect = *find(distancesToSquareUp.begin(), distancesToSquareUp.end(), smallestIntersectDistance);  //  find index of intersect point with smallest distance
    cv::Point2f intersectPoint = squareUpPoints[iIntersect];  // find the nearest intersect point

    //  check if intersect point y is ahead of puckCenter.y and if intersect point is in workspace
    if ((((dy > 0 && intersectPoint.y > puckCenter.y) || (dy < 0 && intersectPoint.y < puckCenter.y)) && intersectPoint.y > 337) && (intersectPoint.x < 250 - strikerRadius || intersectPoint.x > 1 + strikerRadius))
    {
      std::cout<<"distance to intersect: "<<smallestIntersectDistance<<std::endl;
      stepsize = smallestIntersectDistance;
      moveStriker(intersectPoint, strikerCenter);
      return true;
    }
    else  
    {
      std::cout<<"cannot square up"<<std::endl;
      return false;
    }
  }
  return false;
}

/*  function to check if striker is goal side of the puck at any given position
    returns true if it is
*/
bool goalSide(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius)
{
  float x = puckCenter.x - 122;
  float y = puckCenter.y - (548 + strikerRadius);
  float c = sqrt(pow(x,2) + pow(y, 2));
  float x_angle = asin(x/c);  // Sin law
  float y_angle = asin(y/c);

  for (int i = 1; i < c; i++) //  for i points on the trajectory, maybe find the first/second bounce number idk
  {
    float x_new = sin(x_angle) * i; //  coordinates for potential future puck position
    float y_new = sin(y_angle) * i;
    int distSq = (x_new - strikerCenter.x) * (x_new - strikerCenter.x) + (y_new - strikerCenter.y) * (y_new - strikerCenter.y);
    int radSumSq = (puckRadius + strikerRadius) * (puckRadius + strikerRadius);
    
    if (distSq + puckRadius/3 < radSumSq) //  touching or intersecting (add small buffer of puckRadius/3)
    {
      return true;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }
  return false;
}

/*  function that tells striker to stay between puck and goal
    Striker position as close to center of goal as possible (122, 528)
    end of table on robot side y = 548
*/
void stayGoalSide(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, float dx, float dy, cv::Mat &mat)
{
  float x = puckCenter.x - 122;
  float y = puckCenter.y - (548 + 2*strikerRadius);
  float c = sqrt(pow(x,2) + pow(y, 2));
  float x_angle = asin(x/c);  // Sin law
  float y_angle = asin(y/c);

  bool intersect = false;

  for (int i = 1; i < c; i++) //  for i points on the trajectory, maybe find the first/second bounce number idk
  {
    float x_new = sin(x_angle) * i; //  coordinates for potential future puck position
    float y_new = sin(y_angle) * i;
    int distSq = (x_new - strikerCenter.x) * (x_new - strikerCenter.x) + (y_new - strikerCenter.y) * (y_new - strikerCenter.y);
    int radSumSq = (puckRadius + strikerRadius) * (puckRadius + strikerRadius);
    
    if (distSq < radSumSq) //  touching or intersecting
    {
      intersect = true;
      break;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }

  float puckSpeed = sqrt(pow(dx, 2) + pow(dy, 2));

  float goalSideY;
  float goalSideX;

  if (puckCenter.y + 2 * puckRadius < strikerCenter.y - strikerRadius)
  {
    // goalSideY = 528 + y/(abs(dy)*0.1+1);
    // goalSideY = 528 + y/abs(dy);
    goalSideY = 528 + y/5;
    if (goalSideY < 337)  //  if out of range
    {
      goalSideY = 337;
    }
    if (isnan(goalSideY))
    {
      goalSideY = 528;
    }
    if (goalSideY < defenseline)
    {
      goalSideY = defenseline;
    }
    if (puckHeadingToGoal(puckCenter, puckRadius, cv::Point2f(puckCenter.x * dx, puckCenter.y * dy)) && puckSpeed > 800)
    {
      goalSideY = 528;
    }
    if (puckSpeed > 1600)
    {
      goalSideY = 528;
    }
  }
  else 
  {
    goalSideY = 528;
  }

  std::cout<<"goalsidey: "<<goalSideY<<std::endl;
  goalSideX = puckCenter.x - (sin(x_angle) * (puckCenter.y - goalSideY) / sin(y_angle));
  cv::Point2f gsPosition = cv::Point2f(goalSideX, goalSideY); // goal side position

  float dtoMove = sqrt(pow(goalSideX - strikerCenter.x,2) + pow(goalSideY - strikerCenter.y, 2));
  cv::line(mat, strikerCenter, gsPosition, cv::Scalar(255,0,0),2);
  cv::circle(mat, gsPosition, strikerRadius, cv::Scalar(255,0,0),2);
  cv::putText(mat, "stay goal side", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

  if (intersect)
  {
    stepsize = dtoMove/2;
    moveStriker(gsPosition, strikerCenter);
  }
  else
  {
    stepsize = dtoMove;
    moveStriker(gsPosition, strikerCenter);
  }
  prevStatus = 2;
}

//NEUTRAL
/*  function to catch puck (reduce speed of puck)
    only run if squaredUp() returns true
*/
void catchPuck(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, cv::Point2f puckFutureCenter, float dx, float dy, cv::Mat &mat)
{
  float puckSpeed = sqrt(pow(dx,2) + pow(dy, 2));

  float xPuckToStriker = puckCenter.x - strikerCenter.x;
  float yPuckToStriker = puckCenter.y - strikerCenter.y;
  float cPuckToStriker = sqrt(pow(xPuckToStriker,2) + pow(yPuckToStriker, 2));
  //  time to reach striker
  float timeToCollision = cPuckToStriker/puckSpeed;

  //  have striker move on puck trajectory away from puck slightly
  //  find puck trajectory
  float x = puckCenter.x - puckFutureCenter.x;
  float y = puckCenter.y - puckFutureCenter.y;
  float c = sqrt(pow(x,2) + pow(y, 2));
  // float x_angle = asin(x/c);  // Sin law
  // float y_angle = asin(y/c);  
  float x_angle = atan2(x, y);
  float y_angle = atan2(y, x);

  // float catchTiming = 0.25;
  // float catchTiming = 6/sqrt(puckSpeed);
  float catchTiming = 0.2;
  if (puckSpeed < 1000)
  {
    catchTiming = 0.15;
    if (puckSpeed < 500)
    {
      catchTiming -= 0.01;
    }
  }
  neededToSquareUp = false;
  if (prevStatus == 3 || dy < 0)  //  if it previously was squaring up, it probably isn't directly behind the puck - start moving sooner
  {
    if (puckSpeed > 700 && dy < 0) //  to prevent own goals
    {
      moveStrikerAvoidPuck(cv::Point2f(122, 360), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
      return;
    }
    if (puckSpeed > 500)
    {
      catchTiming += 0.03;
    }
    else
    {
      catchTiming += 0.02;
    }
    if (prevStatus == 3)
    {
      neededToSquareUp = true;
    }
  }
  cv::putText(mat, std::to_string(catchTiming), cv::Point2f(60,130), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

  if (timeToCollision < catchTiming) //  0.15 old
  {
    // if (prevStatus == 4)
    // {
    //   return;
    // }
    //  Catch
    //  find point to move striker towards  
    float catchX;
    float catchY;

    float catchDistance = 50;
    // float catchDistance = puckSpeed * 0.1;
    // if (puckSpeed < 700)  //  arbitrary number
    // {
    //   catchDistance = puckSpeed * 0.05;
    // }
    // if (puckSpeed > 1000)
    // {

    //   catchDistance = puckSpeed * 0.2;
    // }
    //  move striker catchDistance units in the direction of puck trajectory
    catchY = strikerCenter.y - catchDistance * sin(y_angle);
    if (catchY > 528) //  if catch point outside of workspace - causes striker to move unexpectedly
    {
      catchY = 528;
    }
    catchX = puckCenter.x - (sin(x_angle) * (puckCenter.y - catchY) / sin(y_angle));
    cv::Point2f catchPoint = cv::Point2f(catchX, catchY);
    
    std::cout<<"catch!"<<std::endl;
    float strikerToMove = sqrt(pow(catchPoint.x - strikerCenter.x, 2) + pow(catchPoint.y - strikerCenter.y, 2));  //  distance for striker to move to match catch distance
    stepsize = strikerToMove;
    moveStriker(catchPoint, strikerCenter);
    //  display planned striker trajectory
    cv::line(mat, strikerCenter, catchPoint, cv::Scalar(0,255,0), 2);
    cv::circle(mat, catchPoint, strikerRadius, cv::Scalar(0,255,0), 2);
    cv::putText(mat, "catch", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::putText(mat, std::to_string(catchDistance), cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::putText(mat, std::to_string(puckSpeed), cv::Point2f(60,110), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    prevStatus = 4;
    return;
  }
  else
  {
    prevStatus = 0;
  }
  if (timeToCollision > 2*catchTiming)  
  {
    //  find point to move striker towards  
    float catchX;
    float catchY;

    float catchDistance = -puckRadius;
    // float catchDistance = -puckSpeed * 0.1;
    //  move striker catchDistance units in the direction of puck trajectory
    catchY = strikerCenter.y - catchDistance * sin(y_angle);
    catchX = puckCenter.x - (sin(x_angle) * (puckCenter.y - catchY) / sin(y_angle));
    cv::Point2f catchPoint = cv::Point2f(catchX, catchY);

    std::cout<<"Preparing to catch!"<<std::endl;
    float strikerToMove = sqrt(pow(catchPoint.x - strikerCenter.x, 2) + pow(catchPoint.y - strikerCenter.y, 2));  //  distance for striker to move to match catch distance
    stepsize = strikerToMove;
    moveStriker(catchPoint, strikerCenter);
    return;
  }
  std::cout<<"couldn't catch!"<<timeToCollision<<std::endl;
}

/*  function to stay 50 units behind the puck on Y axis
    and similar X value as puck
    returns true if already within the distance
*/
bool readyUp(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, float readyDistance, float dx, float dy, cv::Mat &mat)
{
  float x = puckCenter.x - strikerCenter.x;
  float y = puckCenter.y - strikerCenter.y;
  float c = sqrt(pow(x, 2) + pow(y, 2));
  // float x_angle = asin(x/c);
  // float y_angle = asin(y/c);
  float x_angle = atan2(x, y);
  float y_angle = atan2(y, x);

  // if (c <= readyDistance)
  // {
  //   prevStatus = 6;
  //   return true;
  // }

  // float x_new = strikerCenter.x + sin(x_angle) * (c - readyDistance);
  // float y_new = strikerCenter.y + sin(y_angle) * (c - readyDistance);

  float y_new = puckCenter.y + readyDistance;
  float x_new = (puckCenter.x - 122)*0.8 + 122; 

  bool ready = false;
  if (abs(strikerCenter.y - y_new < 10) && abs(strikerCenter.x - x_new < 10))
  {
    ready =  true;
  }

  cv::Point2f readyPosition = cv::Point2f(x_new, y_new);

  cv::line(mat, strikerCenter, readyPosition, cv::Scalar(255,255,0), 2);
  cv::circle(mat, readyPosition, strikerRadius, cv::Scalar(255,255,0), 2);

  // stepsize = (c - readyDistance);
  // stepsize = sqrt(pow(readyPosition.x - strikerCenter.x, 2) + pow(readyPosition.y - strikerCenter.y, 2));
  // moveStriker(readyPosition, strikerCenter);
  moveStrikerAvoidPuck(readyPosition, strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
  prevStatus = 6;
  if (ready)
  {
    return true;
  }
  return false;
}

/* function to hit puck if at edges
*/
bool unStuckPuck(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, float dx, float dy, cv::Mat &mat)
{
  if (puckCenter.y > 548 - 2*puckRadius)
  {
    std::cout<<"puck at end wall"<<std::endl;
    cv::putText(mat, "puck at end wall", cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    if (abs(122 - puckCenter.x) - puckRadius > abs(122 - strikerCenter.x) - strikerRadius) //  if striker is closer to center on x axis than puck
    {
      float distance = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y + (puckRadius + strikerRadius - 5) - strikerCenter.y, 2));
      stepsize = distance + 100;
      moveStriker(puckCenter, strikerCenter);
      return true;
    }
    else if (puckCenter.x > 122)
    {
      stepsize = 20;
      moveStriker(cv::Point2f(puckCenter.x - (puckRadius + strikerRadius + 20), puckCenter.y), strikerCenter);
      // moveStrikerAvoidPuck(cv::Point2f(puckCenter.x - (puckRadius + strikerRadius + 20), puckCenter.y), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
      return true;
    }
    else if (puckCenter.x < 122)
    {
      stepsize = 20;
      moveStriker(cv::Point2f(puckCenter.x + (puckRadius + strikerRadius + 20), puckCenter.y), strikerCenter);
      // moveStrikerAvoidPuck(cv::Point2f(puckCenter.x + (puckRadius + strikerRadius + 20), puckCenter.y), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
      return true;
    }
  }
  if (puckCenter.x > 250 - 2*puckRadius || puckCenter.x < 1 + 2*puckRadius)
  {
    std::cout<<"puck stuck at side wall"<<std::endl;
    cv::putText(mat, "puck stuck at side wall", cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    if (puckCenter.y + puckRadius < strikerCenter.y - strikerRadius)  //  if striker behind puck
    {
      float distance = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y + (puckRadius + strikerRadius - 5) - strikerCenter.y, 2));
      stepsize = distance + 100;
      moveStriker(puckCenter, strikerCenter);
      return true;
    }
    else
    {
      stepsize = 20;
      moveStriker(cv::Point2f(puckCenter.x, puckCenter.y + (puckRadius + strikerRadius + 20)), strikerCenter);
      // moveStrikerAvoidPuck(cv::Point2f(puckCenter.x, puckCenter.y + (puckRadius + strikerRadius + 20)), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
      return true;
    }
  }
  return false;
}

/*  function to hit puck back to other side randomly
    returns true if action is performed
*/
bool hitPuckAway(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, float dx, float dy, cv::Mat &mat)
{
  float puckSpeed = sqrt(pow(dx, 2) + pow(dy, 2));
  if (prevStatus == 1)  //  if a shot was attempted previously
  {
    return false;
  }
  if (prevStatus == 5)  //  if hitaway was attempted previously
  {
    //  check if current striker trajectory (strikerCenter and prevStrikerTarget) still hits puck
    float sX = prevStrikerTarget.x - strikerCenter.x;
    float sY = prevStrikerTarget.y - strikerCenter.y;
    float sC = sqrt(pow(sX, 2) + pow(sY, 2));
    // float sXAngle = asin(sX/sC);
    // float sYAngle = asin(sY/sC);
    float sXAngle = atan2(sX, sY);
    float sYAngle = atan2(sY, sX);

    bool strikerOnTarget = false;

    for (int i = 0; i < sC; i++)  //  for each point along striker trajectory
    {
      float iSX = sX + sin(sXAngle) * i; //  coordinates for potential future striker position
      float iSY = sY + sin(sYAngle) * i;
      int distSq = pow((iSX - puckCenter.x), 2) + pow((iSY - puckCenter.y), 2);
      int radSumSq = pow((puckRadius + strikerRadius), 2);
      
      if (distSq < radSumSq) //  touching or intersecting
      {
        strikerOnTarget = true;
      }
      else  //  not touching or intersecting
      {
        continue;
      }
    }
    if (strikerOnTarget)
    {
      return true;
    }
    // else
    // {
    //   prevStatus = 0;
    //   return false;
    // }
  }
  //  check striker pos relative to puck
  if (puckCenter.y + puckRadius < strikerCenter.y - strikerRadius) //  if puck in front of striker
  {
    float x = puckCenter.x;
    float y = puckCenter.y;
    if (puckSpeed > 10)
    {
      float strikerToPuckDistance = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y - strikerCenter.y, 2));
      float strikerAssumedSpeed = getStrikerAssumedSpeed(puckCenter, strikerCenter);
      cv::putText(mat, std::to_string(strikerAssumedSpeed), cv::Point2f(30,500), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      x = puckCenter.x + dx * (strikerToPuckDistance/strikerAssumedSpeed); //  puck position position after time for striker to reach puck
      y = puckCenter.y + dy * (strikerToPuckDistance/strikerAssumedSpeed);
    }

    //  display planned striker trajectory
    cv::line(mat, strikerCenter, cv::Point2f(x, y), cv::Scalar(255,0,255), 2);
    cv::circle(mat, cv::Point2f(x, y), strikerRadius, cv::Scalar(255,0,255), 2);
    cv::putText(mat, "hit away!", cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

    float distance = sqrt(pow(x - strikerCenter.x, 2) + pow(y - strikerCenter.y, 2));
    stepsize = distance + 100;
    moveStriker(cv::Point2f(x, y), strikerCenter);
    prevStrikerTarget = puckCenter;
    prevStatus = 5;
    return true;
  }
  else
  {
    prevStatus = 0;
  }

  if (unStuckPuck(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat))
  {
    return true;
  }
  
  cv::putText(mat, "go behind puck to hit away!", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
  stepsize = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y + puckRadius + strikerRadius + 20 - strikerCenter.y, 2));
  moveStrikerAvoidPuck(cv::Point2f(strikerCenter.x, puckCenter.y + puckRadius + strikerRadius + 20), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
  return false;
}

/*  function to hit puck such that it stays in workspace but also not towards own goal
*/
void adjustPuckPosition(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, float dx, float dy, cv::Mat &mat)
{
  //  check trajectory of puck if striker hits it directly
  float x = puckCenter.x - strikerCenter.x;
  float y = puckCenter.y - strikerCenter.y;
  float c = sqrt(pow(x, 2) + pow(y, 2));
  float x_angle = asin(x/c);
  float y_angle = asin(y/c);
  
  if (y < 0)  //  puck would leave workspace eventually, striker behind puck
  {
    //  check if first bounce would be in workspace
    //  find y val at x = 1, x = 250
    float y_new;
    if (puckCenter.y - 1*sin(y_angle)/sin(x_angle) > puckCenter.y - 250*sin(y_angle)/sin(x_angle))
    {
      y_new = puckCenter.y - 250*sin(y_angle)/sin(x_angle);
    }
    else
    {
      y_new = puckCenter.y - 1*sin(y_angle)/sin(x_angle);
    }
    if (y_new < 320)  //  if first bounce not in workspace
    {
      cv::putText(mat, "adjust!", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      moveStrikerAvoidPuck(cv::Point2f(strikerCenter.x, puckCenter.y), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
      return;
    }
    else
    {
      //  display planned striker trajectory
      cv::line(mat, strikerCenter, puckCenter, cv::Scalar(0,122,200), 2);
      cv::circle(mat, puckCenter, strikerRadius, cv::Scalar(0,122,200), 2);
      cv::putText(mat, "adjust!", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      stepsize = c;
      moveStriker(puckCenter, strikerCenter);
      return;
    }
  }
  else  //  striker ahead of puck
  {
    if (puckHeadingToGoal(puckCenter, puckRadius, cv::Point2f(puckCenter.x*dx, puckCenter.y*dy)))
    {
      stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
      return;
    }
    //  check if puck is between goal and striker
    if (goalSide(strikerCenter, strikerRadius, puckCenter, puckRadius))
    {
      cv::putText(mat, "adjust!", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      moveStrikerAvoidPuck(cv::Point2f(strikerCenter.x, puckCenter.y + 50), strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
      return;
    }
    else
    {
      //  display planned striker trajectory
      cv::line(mat, strikerCenter, puckCenter, cv::Scalar(0,122,200), 2);
      cv::circle(mat, puckCenter, strikerRadius, cv::Scalar(0,122,200), 2);
      cv::putText(mat, "adjust!", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      stepsize = c;
      moveStriker(puckCenter, strikerCenter);
      return;
    }
  }
}

//OFFENSE
/*  function to perform shoot action
    find 3 shot trajectories from puck - straight, left bank, right bank 
    for all trajectories check collision with playerstriker
    rank trajectories based on striker position (distance to line up and relative position to puck) and playerstriker distance to block
    
    decide whether or not to shoot,
      take best trajectory if shoot

    returns true if a shot is taken
    returns false if shot not taken, or lining up shot
*/
bool shootPuck(cv::Point2f puckCenter, float puckRadius, cv::Point2f strikerCenter, float strikerRadius, cv::Point2f puckFutureCenter, float dx, float dy, cv::Point2f pStrikerCenter, float pStrikerRadius, cv::Mat &mat, double shotClock)
{
  float goalX = 122;
  float goalY = 2 - 2*puckRadius;

  cv::line(mat, cv::Point2f(91, 2), cv::Point2f(154, 2), cv::Scalar(120,0,255), 2);
  cv::circle(mat, cv::Point2f(goalX, 2), puckRadius, cv::Scalar(120,0,255), 2);

  // std::cout<<"PUCK STATIONARY --------------------------------------"<<std::endl;
  std::vector<std::vector<float>> shotTrajectories;

  float puckSpeed = sqrt(pow(dx, 2) + pow(dy, 2));

  float iX = puckCenter.x;
  float iY = puckCenter.y;

  if (puckSpeed > 10)
  {
    float strikerToPuckDistance = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y - strikerCenter.y, 2));
    float strikerAssumedSpeed = getStrikerAssumedSpeed(cv::Point2f(iX, iY), strikerCenter);
    cv::putText(mat, std::to_string(strikerAssumedSpeed), cv::Point2f(30,500), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    cv::Point2f iPuckCenter = getPuckFutureCenter(puckCenter, puckRadius, dx, dy, strikerToPuckDistance/strikerAssumedSpeed); //  puck position position after time for striker to reach puck
    iX = iPuckCenter.x;
    iY = iPuckCenter.y;
  }

  if (prevStatus == 1)  //  if a shot was attempted previously
  {
    //  check if current striker trajectory (strikerCenter and prevStrikerTarget) still hits puck
    float sX = prevStrikerTarget.x - strikerCenter.x;
    float sY = prevStrikerTarget.y - strikerCenter.y;
    float sC = sqrt(pow(sX, 2) + pow(sY, 2));
    // float sXAngle = asin(sX/sC);
    // float sYAngle = asin(sY/sC);
    float sXAngle = atan2(sX, sY);
    float sYAngle = atan2(sY, sX);

    bool strikerOnTarget = false;

    for (int i = 0; i < sC; i++)  //  for each point along striker trajectory
    {
      float iSX = sX + sin(sXAngle) * i; //  coordinates for potential future puck position
      float iSY = sY + sin(sYAngle) * i;
      int distSq = pow((iSX - iX), 2) + pow((iSY - iY), 2);
      int radSumSq = pow((puckRadius + strikerRadius), 2);
      
      if (distSq < radSumSq) //  touching or intersecting
      {
        strikerOnTarget = true;
      }
      else  //  not touching or intersecting
      {
        continue;
      }
    }
    if (strikerOnTarget)
    {
      return true;
    }
    else  //  striker will miss puck, adjust trajectory
    {
      //  display planned striker trajectory
      // cv::line(mat, strikerCenter, puckCenter, cv::Scalar(255,0,255), 2);
      // cv::circle(mat, puckCenter, strikerRadius, cv::Scalar(255,0,255), 2);
      // cv::putText(mat, "shooting", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

      // float distance = sqrt(pow(puckCenter.x - strikerCenter.x, 2) + pow(puckCenter.y - strikerCenter.y, 2));
      // stepsize = distance + 100;
      // moveStriker(puckCenter, strikerCenter);
      // return true;
    }
  }
  if (prevStatus == 5)
  {
    cv::putText(mat, "hitting away!", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    hitPuckAway(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
    return false;
  }

  //  straight shot
  float iXToGoal = goalX - iX;
  float iYToGoal = goalY - iY;
  float iCToGoal = sqrt(pow(iXToGoal, 2) + pow(iYToGoal, 2));
  std::cout<<"iX: "<<iX<<std::endl;
  std::cout<<"iY: "<<iY<<std::endl;
  std::cout<<"iXToGoal: "<<iXToGoal<<std::endl;
  std::cout<<"iYToGoal: "<<iYToGoal<<std::endl;
  std::cout<<"iCToGoal: "<<iCToGoal<<std::endl;
  // float iXAngle = asin(iXToGoal/iCToGoal);
  // float iYAngle = asin(iYToGoal/iCToGoal);
  float iXAngle = atan2(iXToGoal, iYToGoal);
  float iYAngle = atan2(iYToGoal, iXToGoal);
  
  bool shotValid = true;
  std::cout<<"1"<<std::endl;
  for (int j = 1; j < iCToGoal; j++) //  for j points on shot trajectory
  {
    float jXToGoal = iX + sin(iXAngle) * j; //  coordinates for potential future puck position
    float jYToGoal = iY + sin(iYAngle) * j;
    int distSq = pow((jXToGoal - pStrikerCenter.x), 2) + pow((jYToGoal - pStrikerCenter.y), 2);
    int radSumSq = pow((puckRadius + pStrikerRadius), 2);
    
    if (distSq < radSumSq) //  touching or intersecting
    {
      shotValid = false;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }
  std::cout<<"1.1"<<std::endl;
  if (shotValid)
  {
    //  add to shotTrajectories
    std::vector<float> shotTrajectory;
    shotTrajectory.push_back(iX);
    shotTrajectory.push_back(iY);
    shotTrajectory.push_back(iXToGoal);
    shotTrajectory.push_back(iYToGoal);
    shotTrajectory.push_back(iCToGoal);
    shotTrajectory.push_back(iXAngle);
    shotTrajectory.push_back(iYAngle);
    
    shotTrajectories.push_back(shotTrajectory);
  }
  //  right bank shot
  iXToGoal = goalX + 249 - iX;  //  249 is table width
  iYToGoal = goalY - iY;
  iCToGoal = sqrt(pow(iXToGoal, 2) + pow(iYToGoal, 2));
  // iXAngle = asin(iXToGoal/iCToGoal);
  // iYAngle = asin(iYToGoal/iCToGoal);
  iXAngle = atan2(iXToGoal, iYToGoal);
  iYAngle = atan2(iYToGoal, iXToGoal);

  iXToGoal = 250 - puckRadius - iX;
  iYToGoal = (iXToGoal * sin(iYAngle) / sin(iXAngle));
  iCToGoal = sqrt(pow(iXToGoal, 2) + pow(iYToGoal, 2));

  shotValid = true;
  std::cout<<"2"<<std::endl;
  for (int j = 1; j < iCToGoal; j++) //  for j points on shot trajectory (before bounce)
  {
    float jXToGoal = iX + sin(iXAngle) * j; //  coordinates for potential future puck position
    float jYToGoal = iY + sin(iYAngle) * j;
    int distSq = pow((jXToGoal - pStrikerCenter.x), 2) + pow((jYToGoal - pStrikerCenter.y), 2);
    int radSumSq = pow((puckRadius + pStrikerRadius), 2);
    
    if (distSq < radSumSq) //  touching or intersecting
    {
      shotValid = false;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }
  if (shotValid)
  {
    float kXToGoal = goalX - iXToGoal;
    float kYToGoal = goalY - iYToGoal;
    float kCToGoal = sqrt(pow(kXToGoal, 2) + pow(kYToGoal, 2));
    // float kXAngle = asin(kXToGoal/kCToGoal);
    // float kYAngle = asin(kYToGoal/kCToGoal);
    float kXAngle = atan2(kXToGoal, kYToGoal);
    float kYAngle = atan2(kYToGoal, kXToGoal);

    std::cout<<"3"<<std::endl;
    for (int j = 1; j < kCToGoal; j++) //  for j points on shot trajectory (before bounce)
    {
      float jXToGoal = iXToGoal + sin(kXAngle) * j; //  coordinates for potential future puck position
      float jYToGoal = iYToGoal + sin(kYAngle) * j;
      int distSq = pow((jXToGoal - pStrikerCenter.x), 2) + pow((jYToGoal - pStrikerCenter.y), 2);
      int radSumSq = pow((puckRadius + pStrikerRadius), 2);
      
      if (distSq < radSumSq) //  touching or intersecting
      {
        shotValid = false;
      }
      else  //  not touching or intersecting
      {
        continue;
      }
    }
    if (shotValid)
    {
      //  add to shotTrajectories
      std::vector<float> shotTrajectory;
      shotTrajectory.push_back(iX);
      shotTrajectory.push_back(iY);
      shotTrajectory.push_back(iXToGoal);
      shotTrajectory.push_back(iYToGoal);
      shotTrajectory.push_back(iCToGoal);
      shotTrajectory.push_back(iXAngle);
      shotTrajectory.push_back(iYAngle);
      
      shotTrajectories.push_back(shotTrajectory);
    }
  }

  //  left bank shot
  iXToGoal = goalX - 249 - iX;  //  249 is table width
  iYToGoal = goalY - iY;
  iCToGoal = sqrt(pow(iXToGoal, 2) + pow(iYToGoal, 2));
  // iXAngle = asin(iXToGoal/iCToGoal);
  // iYAngle = asin(iYToGoal/iCToGoal);
  iXAngle = atan2(iXToGoal, iYToGoal);
  iYAngle = atan2(iYToGoal, iXToGoal);

  iXToGoal = 1 + puckRadius - iX;
  iYToGoal = (iXToGoal * sin(iYAngle) / sin(iXAngle));
  iCToGoal = sqrt(pow(iXToGoal, 2) + pow(iYToGoal, 2));

  shotValid = true;
  std::cout<<"4"<<std::endl;
  for (int j = 1; j < iCToGoal; j++) //  for j points on shot trajectory (before bounce)
  {
    float jXToGoal = iX + sin(iXAngle) * j; //  coordinates for potential future puck position
    float jYToGoal = iY + sin(iYAngle) * j;
    int distSq = pow((jXToGoal - pStrikerCenter.x), 2) + pow((jYToGoal - pStrikerCenter.y), 2);
    int radSumSq = pow((puckRadius + pStrikerRadius), 2);
    
    if (distSq < radSumSq) //  touching or intersecting
    {
      shotValid = false;
    }
    else  //  not touching or intersecting
    {
      continue;
    }
  }
  if (shotValid)
  {
    float kXToGoal = goalX - iXToGoal;
    float kYToGoal = goalY - iYToGoal;
    float kCToGoal = sqrt(pow(kXToGoal, 2) + pow(kYToGoal, 2));
    // float kXAngle = asin(kXToGoal/kCToGoal);
    // float kYAngle = asin(kYToGoal/kCToGoal);
    float kXAngle = atan2(kXToGoal, kYToGoal);
    float kYAngle = atan2(kYToGoal, kXToGoal);

    std::cout<<"5"<<std::endl;
    for (int j = 1; j < kCToGoal; j++) //  for j points on shot trajectory (before bounce)
    {
      float jXToGoal = iXToGoal + sin(kXAngle) * j; //  coordinates for potential future puck position
      float jYToGoal = iYToGoal + sin(kYAngle) * j;
      int distSq = pow((jXToGoal - pStrikerCenter.x), 2) + pow((jYToGoal - pStrikerCenter.y), 2);
      int radSumSq = pow((puckRadius + pStrikerRadius), 2);
      
      if (distSq < radSumSq) //  touching or intersecting
      {
        shotValid = false;
      }
      else  //  not touching or intersecting
      {
        continue;
      }
    }
    if (shotValid)
    {
      //  add to shotTrajectories
      std::vector<float> shotTrajectory;
      shotTrajectory.push_back(iX);
      shotTrajectory.push_back(iY);
      shotTrajectory.push_back(iXToGoal);
      shotTrajectory.push_back(iYToGoal);
      shotTrajectory.push_back(iCToGoal);
      shotTrajectory.push_back(iXAngle);
      shotTrajectory.push_back(iYAngle);
      
      shotTrajectories.push_back(shotTrajectory);
    }
  }

  std::vector<float> shotTrajectoryScores;
  std::vector<std::vector<float>> shotTrajectoryScoreDetails;
  std::vector<cv::Point2f> shotTrajectoryLineUpPoints;
  std::vector<cv::Point2f> shotTrajectoryPuckPos;
  std::vector<cv::Point2f> shotTrajectoryTarget;
  std::vector<int> shotTrajectoryID;

  std::cout<<"6"<<std::endl;
  for (int i = 0; i < shotTrajectories.size(); i++) //  for each valid shot trajectory
  {
    //  check striker distance to line up shot
    //  check player striker distance to squareUp
    //  rank!
    std::vector<float> shotTrajectory = shotTrajectories[i];
    float iX = shotTrajectory[0]; //  puck position to shoot from
    float iY = shotTrajectory[1];
    float iXToGoal = shotTrajectory[2];
    float iYToGoal = shotTrajectory[3];
    float iCToGoal = shotTrajectory[4];
    float iXAngle = shotTrajectory[5];
    float iYAngle = shotTrajectory[6];
    
    //  find shortest striker distance to line up shot (same logic as squareUp())
    std::vector<float> distancesToLineUp;
    std::vector<cv::Point2f> lineUpPoints;
    std::cout<<"7"<<std::endl;
    for (int j = 0; j < 528 - (iY + puckRadius + strikerRadius); j++) //  for each y coordinate opposite of shot trajectory
    {
      float lineUpY = iY + puckRadius + strikerRadius + j;
      float lineUpX = (iXToGoal + iX) - (sin(iXAngle) * ((iYToGoal + iY) - lineUpY) / sin(iYAngle));
      //  make sure lineUpPoint is in workspace
      if (lineUpX > 250 - strikerRadius || lineUpX < 1 + strikerRadius)
      {
        continue;
      }
      float lineUpDistance = sqrt(pow(lineUpX - strikerCenter.x, 2) + pow(lineUpY - strikerCenter.y, 2));
      lineUpPoints.push_back(cv::Point2f(lineUpX, lineUpY));
      distancesToLineUp.push_back(lineUpDistance);
    }
    float shortestLineUpDistance;
    float maxWindUpDistance = strikerRadius + puckRadius + 75;
    float minWindUpDistance = strikerRadius + puckRadius + 20;
    std::vector<float>::iterator itIntersect;
    int iIntersect;
    if (distancesToLineUp.size() == 0)
    {
      continue;  //  cannot line up? go next shot trajectory
    }
    else
    {
      std::cout<<"7.1"<<std::endl;
      while (distancesToLineUp.size() > 1)
      {
        shortestLineUpDistance = *std::min_element(distancesToLineUp.begin(), distancesToLineUp.end());
        itIntersect = find(distancesToLineUp.begin(), distancesToLineUp.end(), shortestLineUpDistance);  //  find index of intersect point with shortest distance
        iIntersect = std::distance(distancesToLineUp.begin(), itIntersect);
        // if (sqrt(pow(puckCenter.x - lineUpPoints[iIntersect].x, 2) + pow(puckCenter.y - lineUpPoints[iIntersect].y, 2)) > maxWindUpDistance) //  if the lineup point is greater than windUpDistance from the puck
        // {
        //   distancesToLineUp.erase(itIntersect);
        //   lineUpPoints.erase(find(lineUpPoints.begin(), lineUpPoints.end(), lineUpPoints[iIntersect]));
        // }
        // else if (sqrt(pow(puckCenter.x - lineUpPoints[iIntersect].x, 2) + pow(puckCenter.y - lineUpPoints[iIntersect].y, 2)) < minWindUpDistance) //  if the lineup point is less than windUpDistance - 20 from the puck
        // {
        //   distancesToLineUp.erase(itIntersect);
        //   lineUpPoints.erase(find(lineUpPoints.begin(), lineUpPoints.end(), lineUpPoints[iIntersect]));
        // }
        // else
        // {
          break;
        // }
      }
      std::cout<<"7.1.1"<<std::endl;
      if (distancesToLineUp.size() == 1)
      {
        //  check the distance the point is behind the puck
        // if (sqrt(pow(puckCenter.x - lineUpPoints[0].x, 2) + pow(puckCenter.y - lineUpPoints[0].y, 2)) > maxWindUpDistance)  //  if the lineup point is greater than windUpDistance from the puck
        // {
        //   continue;
        // }
        // else if (sqrt(pow(puckCenter.x - lineUpPoints[0].x, 2) + pow(puckCenter.y - lineUpPoints[0].y, 2)) < minWindUpDistance) //  if the lineup point is less than windUpDistance - 20 from the puck
        // {
        //   continue;
        // }
        shortestLineUpDistance = distancesToLineUp[0];
        iIntersect = 0;
      std::cout<<"7.3"<<std::endl;
      }
    }
    
      std::cout<<"7.2"<<std::endl;
    cv::Point2f intersectPoint = lineUpPoints[iIntersect];  // find the nearest intersect point

    //  find shortest distance for player striker to move to squareUp() at any point of the trajectory
    std::vector<float> distancesToGetBlocked;
    std::vector<cv::Point2f> blockedPoints;
    float getBlockedX;
    float distanceToGetBlocked;
    std::cout<<"8"<<std::endl;
    for (int j = 0; j < iYToGoal; j++)
    {
      float getBlockedY = iY - (puckRadius + pStrikerRadius) - j;
      if (getBlockedY > 280)
      {
        continue;
      }
      getBlockedX = (iXToGoal + iX) - (sin(iXAngle) * ((iYToGoal + iY) - getBlockedY) / sin(iYAngle));
      distanceToGetBlocked = sqrt(pow(getBlockedX - pStrikerCenter.x, 2) + pow(getBlockedY - pStrikerCenter.y, 2));
      distancesToGetBlocked.push_back(distanceToGetBlocked);
      blockedPoints.push_back(cv::Point2f(getBlockedX, getBlockedY));
    }
    if (iYToGoal != goalY - iY) //  check if it's straight shot or not
    {   
      float jXToGoal = 122 - (iXToGoal + iX);
      float jYToGoal = (548 + 2*puckRadius) - (iYToGoal + iY);
      float jCToGoal = sqrt(pow(jXToGoal, 2) + pow(jYToGoal, 2));
      float jXAngle = asin(jXToGoal/jCToGoal);
      float jYAngle = asin(jYToGoal/jCToGoal);
      std::cout<<"9"<<std::endl;
      for (int j = 0; j < jYToGoal; j++)
      {
        float getBlockedY = iY - (puckRadius + pStrikerRadius) - j;
        if (getBlockedY > 280)
        {
          continue;
        }
        getBlockedX = (iXToGoal + iX) - (sin(iXAngle) * ((iYToGoal + iY) - getBlockedY) / sin(iYAngle));
        distanceToGetBlocked = sqrt(pow(getBlockedX - pStrikerCenter.x, 2) + pow(getBlockedY - pStrikerCenter.y, 2));
        distancesToGetBlocked.push_back(distanceToGetBlocked);
        blockedPoints.push_back(cv::Point2f(getBlockedX, getBlockedY));
      }
    }

    float shortestDistanceToGetBlocked;
    if (distancesToGetBlocked.size() == 0)
    {
      shortestDistanceToGetBlocked = 1000;  //  can't be blocked? number out of range
    }
    else if (distancesToGetBlocked.size() == 1)
    {
      shortestDistanceToGetBlocked = distancesToGetBlocked[0];
    }
    else
    {
      shortestDistanceToGetBlocked = *std::min_element(distancesToGetBlocked.begin(), distancesToGetBlocked.end());
    }
    
    //  find line up point nearest windupdistance from puck
    std::vector<float> windUpDistanceScores;
    std::vector<cv::Point2f> windUpPoints;
    std::cout<<"7"<<std::endl;
    for (int j = 0; j < 528 - (iY + puckRadius + strikerRadius); j++) //  for each y coordinate opposite of shot trajectory
    {
      float windUpY = iY + puckRadius + strikerRadius + j;
      float windUpX = (iXToGoal + iX) - (sin(iXAngle) * ((iYToGoal + iY) - windUpY) / sin(iYAngle));
      //  make sure windUpPoint is in workspace
      if (windUpX > 250 - strikerRadius || windUpX < 1 + strikerRadius)
      {
        continue;
      }

      float windUpDistance = sqrt(pow(windUpX - strikerCenter.x, 2) + pow(windUpY - strikerCenter.y, 2));
      windUpPoints.push_back(cv::Point2f(windUpX, windUpY));
      float windUpDistanceScore = abs(windUpDistance - desiredWindUpDistance);
      windUpDistanceScores.push_back(windUpDistanceScore);
    }
    float bestWindUpScore;
    if (windUpDistanceScores.size() == 0)
    {
      continue;  //  cannot line up? go next trajectory
    }
    else if (windUpDistanceScores.size() == 1)
    {
      bestWindUpScore = windUpDistanceScores[0];
      iIntersect = 0;
    }
    else
    {
      bestWindUpScore = *std::min_element(windUpDistanceScores.begin(), windUpDistanceScores.end());
      itIntersect = find(windUpDistanceScores.begin(), windUpDistanceScores.end(), bestWindUpScore);  //  find index of intersect point with best score
      iIntersect = std::distance(windUpDistanceScores.begin(), itIntersect);
    }
    intersectPoint = windUpPoints[iIntersect];  // find the best intersect point

    float shotPreference = 0;
    if (prevShotChoice == i)
    {
      shotPreference = 20;
    }
    float score = shortestDistanceToGetBlocked - shortestLineUpDistance*0 - bestWindUpScore + shotPreference; //  larger the shortest distance to get blocked is better, smaller line up distance is better, smaller windUpDistanceScore is better 
    shotTrajectoryScores.push_back(score);
    std::vector<float> shotTrajectoryScoreInfo = {shortestDistanceToGetBlocked, shortestLineUpDistance, bestWindUpScore};
    shotTrajectoryScoreDetails.push_back(shotTrajectoryScoreInfo);
    shotTrajectoryLineUpPoints.push_back(intersectPoint);
    shotTrajectoryPuckPos.push_back(cv::Point2f(iX, iY));
    shotTrajectoryTarget.push_back(cv::Point2f(iXToGoal + iX, iYToGoal + iY));
    shotTrajectoryID.push_back(i);
  }
  if (shotTrajectoryScores.size() != 0)
  {
    float maxScore;
    int iMaxScore;
    if (shotTrajectoryScores.size() == 1)
    {
      maxScore = shotTrajectoryScores[0];
      iMaxScore = 0;
    }
    else
    {
      maxScore = *std::max_element(shotTrajectoryScores.begin(), shotTrajectoryScores.end());
      std::cout<<"hello1"<<std::endl;
      std::vector<float>::iterator itMaxScore = find(shotTrajectoryScores.begin(), shotTrajectoryScores.end(), maxScore);  //  find index of line up point with best score
      std::cout<<"hello2"<<std::endl;
      iMaxScore = std::distance(shotTrajectoryScores.begin(), itMaxScore);
    }
    prevShotChoice = shotTrajectoryID[iMaxScore];
    std::cout<<"hello3"<<std::endl;
    cv::Point2f intersectPoint = shotTrajectoryLineUpPoints[iMaxScore];  // find the nearest line up point
    std::cout<<"iMaxScore: "<<iMaxScore<<std::endl;
    std::cout<<"vecotr size: "<<shotTrajectoryLineUpPoints.size()<<std::endl;
    if (shotClock >= 3) //  7 seconds is max according to game rules, set to 3 for more fast pace
    {
      std::cout<<"SHOT CLOCK: "<<shotClock<<std::endl;
      cv::putText(mat, "hit away (shot clock)", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      cv::putText(mat, std::to_string(shotClock), cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      hitPuckAway(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
      return false;
    }
    if (puckCenter.y < 360 && dy < 50) //  puck is almost leaving workspace (past blue line)
    {
      std::cout<<"puck leaving workspace!"<<std::endl;
      cv::line(mat, cv::Point2f(0, 360), cv::Point2f(250, 360), cv::Scalar(255,255,0), 2);
      cv::putText(mat, "hit away (puck leaving)", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      hitPuckAway(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
      return false;
    }
    // if (maxScore < 0)  //  if max score is less than a threshold then control??
    // {
    //   std::cout<<"MAX SCORE TOO LOW: "<<maxScore<<std::endl;
    //   cv::putText(mat, "ready up", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
    //   //  control puck?
    //   // adjustPuckPosition(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
    //   // readyUp(puckCenter, puckRadius, strikerCenter, strikerRadius, 50, mat);
    //   return false;
    // }
    else
    {

      //  display planned puck trajectory
      cv::line(mat, shotTrajectoryPuckPos[iMaxScore], shotTrajectoryTarget[iMaxScore], cv::Scalar(255,0,120), 2);
      cv::circle(mat, shotTrajectoryTarget[iMaxScore], puckRadius, cv::Scalar(255,0,120), 2);
      //  check if lined up
      // if (shotTrajectoryScoreDetails[iMaxScore][1] < puckRadius)  //  if shortestLineUpDistance is less than puckRadius 
      if (sqrt(pow(shotTrajectoryLineUpPoints[iMaxScore].x - strikerCenter.x, 2) + pow(shotTrajectoryLineUpPoints[iMaxScore].y - strikerCenter.y, 2)) < strikerRadius && shotTrajectoryScoreDetails[iMaxScore][1] < puckRadius) //  if striker distance from windUpPoint is less than strikerRadius and shortestLineUpDistance is less than puckRadius
      {
        std::cout<<"SHOOTING"<<std::endl;
        //  display planned striker trajectory
        cv::line(mat, strikerCenter, shotTrajectoryPuckPos[iMaxScore], cv::Scalar(255,0,255), 2);
        cv::circle(mat, shotTrajectoryPuckPos[iMaxScore], strikerRadius, cv::Scalar(255,0,255), 2);
        cv::putText(mat, "shooting", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

        float distance = sqrt(pow(shotTrajectoryPuckPos[iMaxScore].x - strikerCenter.x, 2) + pow(shotTrajectoryPuckPos[iMaxScore].y - strikerCenter.y, 2));
        stepsize = distance + 100;
        if (distance < 40)
        {
          stepsize = distance - (puckRadius + strikerRadius);
        }
        moveStriker(shotTrajectoryPuckPos[iMaxScore], strikerCenter);
        prevStrikerTarget = shotTrajectoryPuckPos[iMaxScore];
        prevStatus = 1;
        return true;
      }
      else  //  if not lined up
      {
        std::cout<<"LINING UP SHOT"<<std::endl;
        //  display planned striker trajectory
        cv::line(mat, strikerCenter, intersectPoint, cv::Scalar(255,0,255), 2);
        cv::circle(mat, intersectPoint, strikerRadius, cv::Scalar(255,0,255), 2);
        cv::putText(mat, "lining up", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

        float distance = sqrt(pow(intersectPoint.x - strikerCenter.x, 2) + pow(intersectPoint.y - strikerCenter.y, 2));
        // cv::putText(mat, std::to_string(distance), cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
        if (puckSpeed > 200)
        {
          // don't line up
          readyUp(puckCenter, puckRadius, strikerCenter, strikerRadius, desiredWindUpDistance, dx, dy, mat);
          return false;
        }
        
        // stepsize = 100;
        // if (distance < 100)
        // {
        //   stepsize = distance;
        // }
        stepsize = distance;
        if (distance < puckRadius + strikerRadius + 20)
        {
          stepsize = 10;
        }
        moveStrikerAvoidPuck(intersectPoint, strikerCenter, strikerRadius, puckCenter, puckRadius, dx, dy, mat);
        prevStatus = 0;
        return false;
      }
    }
  }
  else
  {
    //  no shots available
    std::cout<<"CANNOT SHOOT"<<std::endl;
    prevShotChoice = 3;
    cv::putText(mat, "no shots available", cv::Point2f(60,70), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);

    // unStuckPuck(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);  //  gets called in hit puck away

    if (puckSpeed > 50)
    {
      readyUp(puckCenter, puckRadius, strikerCenter, strikerRadius, desiredWindUpDistance, dx, dy, mat);
      return false;
    }
    else
    {
      cv::putText(mat, "hit away (no shots found)", cv::Point2f(60,90), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
      hitPuckAway(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
      return false;
    }
    // adjustPuckPosition(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
    // gotoPuck(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy);
    return false;
  }
}

/*  brains of the robot
*/
void controlLogic(std::vector<cv::Point2f> puckBounces, cv::Point2f puckCenter, float puckRadius, cv::Point strikerCenter, float strikerRadius, float dx, float dy, cv::Point2f pStrikerCenter, float pStrikerRadius, cv::Mat &mat)
{
  float puckSpeed = sqrt(pow(dx,2) + pow(dy, 2));
  // // Catch Puck Performance Testing
  // if (prevStatus == 4 && !loggingCatch)
  // {
  //   loggingCatch = true;
  //   catchID += 1;
  // }
  // if (loggingCatch)
  // {
  //   std::vector<float> iCatchInfo;
  //   iCatchInfo.push_back(catchID);
  //   iCatchInfo.push_back(dx);
  //   iCatchInfo.push_back(dy);
  //   iCatchInfo.push_back(puckSpeed);
  //   iCatchInfo.push_back(prevStatus);
  //   iCatchInfo.push_back(neededToSquareUp);
  //   catchInfo.push_back(iCatchInfo);
  //   if (prevStatus != 4 && loggingExcess == -1)
  //   {
  //     loggingExcess = count;
  //   }
  //   if (count - loggingExcess > 15 && loggingExcess != -1)
  //   {
  //     loggingExcess = -1;
  //     loggingCatch = false;
  //   }
  // }
  // strikerAssumedSpeed = getStrikerAssumedSpeed(puckCenter, strikerCenter);
  // cv::putText(mat, std::to_string(strikerAssumedSpeed), cv::Point2f(30,500), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
  // std::cout<<"Puck Speed: "<<puckSpeed<<std::endl;
  // std::cout<<"velocity: "<< dx << ", "<< dy<<std::endl;
  // std::cout<<"y velocity: "<< dy<<std::endl;
  // std::cout<<"puck center: "<< floor(puckCenter.x) << ", "<< floor(puckCenter.y)<<std::endl;

  // std::cout<<"striker center: "<< floor(strikerCenter.x) << ", "<< floor(strikerCenter.y)<<std::endl;
  // std::cout<<puckBounces.size()<<std::endl;
  if (puckBounces.size() == 0 || puckCenter.x > 250 || puckCenter.x < 1 || puckCenter.y > 548 || puckCenter.y < 2 || puckRadius < 0 || puckRadius > 250)  //  if no puck detected (no puck info or garbage passed to funciton)
  {
    if (puckLost)
    {
      time_t currentTime = time(NULL);
      if (difftime(currentTime, puckLostTime) > timeout)
      {
        client.send("(0.50,0.80)", 11);
        prevStatus = 0;
        puckLost = false;
      }
    }
    else
    {
      puckLostTime = time(NULL);
      puckLost = true;
    }
    return;
  }
  //  shot clock stuff
  if (!puckInWorkspace && puckCenter.y > 280) //  puck on robot side of table
  {
    shotClockStart = time(NULL);
    puckInWorkspace = true;
  }
  if (puckInWorkspace)
  {
    time_t currentTime = time(NULL);
    shotClock = difftime(currentTime, shotClockStart);
  }
  if (puckCenter.y < 280)
  {
    puckInWorkspace = false;
    shotClock = 0;
  }

  // check puck speed (if slow, trajectory prediction is pretty meaningless)
  //  - squaredUp and squareUp are useless below a certain speed
  if (puckSpeed >= 200) //  if speed is larger than threshold (arbitrary number for now)
  {
    std::cout<<"Puck Speed: "<<puckSpeed<<std::endl;
    //  check if squared up with any part of trajectory (incl all bounces)
    bool squaredUpAnyBounce = false;
    int iSquaredUp = -1;
    for (int i = 0; i < puckBounces.size(); i++)
    {
      if (i == 0)
      {
        if (squaredUp(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces[i]))
        {
          squaredUpAnyBounce = true;
          iSquaredUp = i;
        }
      }
      else
      {
        if (squaredUp(puckBounces[i-1], puckRadius, strikerCenter, strikerRadius, puckBounces[i]))
        {
          squaredUpAnyBounce = true;
          iSquaredUp = i;
        }
      }
    }

    if (squaredUpAnyBounce) //  if squared up at any point in predicted trajectory
    {
      //decide whether to catch, shoot, or do nothing
      std::cout<<"Square!"<<std::endl;
      // return; //  do nothing for now.

      //  check which puck trajectory (current/ future after bounces) the striker is squared up with
      // puckBounces[iSquaredUp - 1];  //  initial
      // puckBounces[iSquaredUp];  //  final

      if (iSquaredUp == 0)
      {
        if (puckSpeed > 1500)
        {
          hitPuckAway(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
          return;
        }
        catchPuck(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces[iSquaredUp], dx, dy, mat);
        return;
      }
      else
      {
        // catchPuck(puckBounces[iSquaredUp - 1], puckRadius, strikerCenter, strikerRadius, puckBounces[iSquaredUp], dx, dy, mat);  //  this doesn't work!!!! 
        return;
      }
    }
    else  //  if not squared up
    {
      std::cout<<"NOT Square!"<<std::endl;
      if (dy > 0)
        {
        //decide whether to square up
        if (puckHeadingToGoal(puckCenter, puckRadius, puckBounces[0])) //  if the puck is currently goal-bound
        {
          std::cout<<"Defend!"<<std::endl;
          if (!squareUp(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces, dx, dy, mat))
          {
            stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
          }
          return;
        }
        else  //  if puck is not goal-bound
        {
          //check goalbound status after all bounces
          int iGoalBound = -1;
          for (int i = 1; i < puckBounces.size(); i++)
          {
            if (puckHeadingToGoal(puckBounces[i-1], puckRadius, puckBounces[i]))
            {
              iGoalBound = i;
            }
          }
          if (iGoalBound != -1)
          {
            if (!squareUp(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces, dx, dy, mat))
            {
              stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
            }
            return;
          }
          else  //  if puck not goalbound after all bounces
          {
            //  check speed?
            // std::cout<<"Nothing!"<<std::endl;
            if (puckCenter.y > 320) //  puck in robot workspace
            {
              std::cout<<"SquareUp!"<<std::endl;
              squareUp(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces, dx, dy, mat);
              return;
            }
            else  //  puck not in robot workspace
            {
              stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
              return;
            }
          }
        }
      }
      else  //  if dy <= 0, if puck is moving away from robot workspace
      {
        if (puckCenter.y > 320) //  is the puck in the robot workspace
        {
          std::cout<<"shootpuck!"<<std::endl;
          shootPuck(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces[0], dx, dy, pStrikerCenter, pStrikerRadius, mat, shotClock);
          return;
        }
        else  //  if puck not in robot workspace
        {
          std::cout<<"Nice!"<<std::endl;
          stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
          return;
        }
      }
    }
  }
  else  //  if puck speed is below threshold (set above)
  {
    if (puckCenter.y > 320) //  is the puck in the robot workspace
    {

      //  need to make decision for attack/control etc.
      //  look at previous action

      // if (puckHeadingToGoal(puckCenter, puckRadius, puckBounces[0])) //  if the puck is currently goal-bound
      // {
      //   if (!goalSide(puckCenter, puckRadius, strikerCenter, strikerRadius))
      //   {
      //     std::cout<<"Defend2!"<<std::endl;
      //     if (prevStatus != 1)  //  if the previous action was shoot, don't staygoalside
      //     {
      //       stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
      //       return;
      //     }
      //   }
      // }

      std::cout<<"ShootPuck()!"<<std::endl;
      shootPuck(puckCenter, puckRadius, strikerCenter, strikerRadius, puckBounces[0], dx, dy, pStrikerCenter, pStrikerRadius, mat, shotClock);
      return;
    }
    else if (puckCenter.y > 280)
    {
      cv::line(mat, cv::Point2f(0, 320), cv::Point2f(250, 320), cv::Scalar(0,120,255), 2);
      readyUp(puckCenter, puckRadius, strikerCenter, strikerRadius, desiredWindUpDistance, dx, dy, mat);
      return;
    }
    else  //  if puck not in robot workspace
    {
      std::cout<<"Nice!"<<std::endl;
      stayGoalSide(puckCenter, puckRadius, strikerCenter, strikerRadius, dx, dy, mat);
      return;
    }
  }
  return;
}

//return the striker center
cv::Point2f detectstriker(cv::Mat &mat)
{
  std::cout<<"Detect Striker Called"<<std::endl;
  std::vector<std::vector<cv::Point>> scontours;
  cv::Point2f scenter; 
  float sobjRadius;
  cv::findContours(mat, scontours , cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  // Only proceed if at least one contour was found
  if (scontours.size() > 0)
  {
    // [PROGRESS CHECK]
    //std::cout << "Contour found!" << std::endl;
    for (int idx = 0; idx < scontours.size(); idx++)
    {
      // Loop through contour array and only use the one within puck dimensions
      double objArea = cv::contourArea(scontours[idx]);

      if (objArea >= 100.0 && objArea < 900.0)
      {
        // [VALUE CHECK]
      //std::cout << "Contour area = " << objArea << std::endl;

        // Calculate the radius and center coordinates for the puck
        cv::minEnclosingCircle(scontours[idx], scenter, sobjRadius);
        // Create circle outline for visualisation
        // cv::circle(pData->smallImg, pData->strikercenter, pData->strikerobjRadius, blackCol, 1);

        
        
      }
    }
  }

  return scenter;

}



//pipeline
void imageProcessingTBB(TcamImage &cam,
                        tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue, 
                        cv::cuda::GpuMat map1, cv::cuda::GpuMat map2)
{
  // [ COLORS for drawing on preview video ]
  cv::Scalar greenCol = cv::Scalar(0, 255, 0);
  cv::Scalar redCol = cv::Scalar(0, 0, 255);
  cv::Scalar blueCol = cv::Scalar(255, 0, 0);
  cv::Scalar blackCol = cv::Scalar(0, 0, 0);


  // snap image sample rate
  int frameHeight = cam.getHeight();
  int frameWidth = cam.getWidth();
  int pixels = cam.getBytesPerPixel();
  int sampleRate = 2*230;//4.24; 
  //int sampleRate = 2*237;
  std::cout << " h:" <<frameHeight<< " w:" <<frameWidth<< " p:" <<pixels<< std::endl;
  // [ CROP ]
  int offset_x = 18.5;
  int offset_y = 15;


  cv::Rect roi;
  roi.x = offset_x;
  roi.y = offset_y;
  roi.width = framewidth - (offset_x*2);
  roi.height = frameheight - (offset_y*2);
     
  // [ RESIZE ]
  const double scale = 1.5;
  const double fx = 1 /scale ;
  cv::Mat frame = cv::Mat(1080, 1440, CV_8UC4);

  // [ THRESHOLDING ]
  // Create binary image using HSV color thresholding on inverted image
  // -> Want to binarise the color cyan in inverted image instead of red
  // Cyan is 90
// [ THRESHOLDING ]
  // Create binary image using HSV color thresholding on inverted image
  // -> Want to binarise the color cyan in inverted image instead of red
  // Cyan is 90

  // GREEN - FOR PUCK
  const cv::Scalar minThresh = cv::Scalar(120, 0, 70); //  120, 0, 180  //  120, 0, 70
  const cv::Scalar maxThresh = cv::Scalar(160, 80, 255);  //  160, 80, 255

  // RED - DEBUG PLAYERSTRIKER

  const cv::Scalar minplayerstrikerThresh = cv::Scalar(90, 20, 160);  //  90, 20, 160
  const cv::Scalar maxplayerstrikerThresh = cv::Scalar(100, 100, 255);  //  100, 130, 255

  // ORANGE - FOR STRIKER

  const cv::Scalar minstrikerThresh = cv::Scalar(90, 60, 0); //  90, 60, 0
  const cv::Scalar maxstrikerThresh = cv::Scalar(110, 255, 255);  //  110, 255, 255

  // [ARUCO HOMOGRAPHY]
  std::vector<cv::Point2f> boxPts(4);
  std::vector<std::vector<cv::Point2f>> markerCorners , rejectedCandiates;
  std::vector<int> ids; //ID FOR EACH MARKERS
  std::vector<cv::Point2f> pts_dst; //distance between each point
  cv::Mat h; //img container for homography

  // [ MORPH ]
  const int morph_size = 3;
  const cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                                              cv::Point(morph_size, morph_size));

  // [ KALMAN FILTER ]
  int stateSize = 6;
  int measSize = 4;
  int contrSize = 0;
  unsigned int imgType = CV_32F;
  cv::Mat state, meas;
  cv::Mat strikerstate, strikermeas;
  cv::Mat playerstrikerstate, playerstrikermeas;

  cv::KalmanFilter kf;
  cv::KalmanFilter strikerkf;
  cv::KalmanFilter playerstrikerkf;

  //for first time initialization
  bool FOUND = false;
  bool STRIKERFOUND = false;
  bool PLAYERSTRIKERFOUND = false;

  int notFoundCount = 0;    

  double ticks = 0;
  double dT = 0;  

  double strikerticks = 0;
  double strikerdT = 0;

  double playerstrikerticks = 0;
  double playerstrikerdT = 0;

  int loop = 0;  

  cv::Point2f tl;
  cv::Point2f tr;
  cv::Point2f br; 
  float xoffset; 
  float yoffset; 
  float xscale;
  float yscale; 
  bool CALIBRATED1 = false;
  bool CALIBRATED2 = false;
  bool CALIBRATED3 = false;
  bool CALIBRATED4 = false;





  unsigned int microsecond = 1000000;                                     
  
  // Initialise Kalman filter (done once)
  std::cout << "Initialising Kalman ..." << std::endl;
  // [ INITIALISE KALMAN FILTERS ]
  kf = cv::KalmanFilter(stateSize, measSize, contrSize, imgType);
  strikerkf = cv::KalmanFilter(stateSize, measSize, contrSize, imgType);
  playerstrikerkf = cv::KalmanFilter(stateSize, measSize, contrSize, imgType);

  // Initialise state matrix
  state = cv::Mat(stateSize, 1, imgType); // [x,y,v_x,v_y,w,h]
  strikerstate = cv::Mat(stateSize, 1, imgType); // [x,y,v_x,v_y,w,h]
  playerstrikerstate = cv::Mat(stateSize, 1, imgType); // [x,y,v_x,v_y,w,h]

  /* [NOTE]
      x,y          centroid position of the object (i.e. puck)
      v_x,v_y      velocity of the object's centroid position (pixels/s)
      w,h          size of the bounding box (i.e radius of puck)
    */

  // Initialise measurement matrix
  meas = cv::Mat(measSize, 1, imgType); // [z_x,z_y,z_w,z_h]
  strikermeas = cv::Mat(measSize, 1, imgType); // [z_x,z_y,z_w,z_h]
  playerstrikermeas = cv::Mat(measSize, 1, imgType); // [z_x,z_y,z_w,z_h]

  /* [NOTE]
    z_x,z_y      measured centroid position of the object (i.e. puck)
    z_w,z_h      measured size of the bounding box (i.e radius of puck)
  */

  // Initialise Transition State Matrix A
  // [Note: set dT at each processing step!]
  cv::setIdentity(kf.transitionMatrix);
  cv::setIdentity(strikerkf.transitionMatrix);
  cv::setIdentity(playerstrikerkf.transitionMatrix);


  // Initialise Measure Matrix H
  kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, imgType);
  kf.measurementMatrix.at<float>(0) = 1.0f;
  kf.measurementMatrix.at<float>(7) = 1.0f;
  kf.measurementMatrix.at<float>(16) = 1.0f;
  kf.measurementMatrix.at<float>(23) = 1.0f;

  strikerkf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, imgType);
  strikerkf.measurementMatrix.at<float>(0) = 1.0f;
  strikerkf.measurementMatrix.at<float>(7) = 1.0f;
  strikerkf.measurementMatrix.at<float>(16) = 1.0f;
  strikerkf.measurementMatrix.at<float>(23) = 1.0f;

  playerstrikerkf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, imgType);
  playerstrikerkf.measurementMatrix.at<float>(0) = 1.0f;
  playerstrikerkf.measurementMatrix.at<float>(7) = 1.0f;
  playerstrikerkf.measurementMatrix.at<float>(16) = 1.0f;
  playerstrikerkf.measurementMatrix.at<float>(23) = 1.0f;

  // Initialize Process Noise Covariance Matrix Q
  kf.processNoiseCov.at<float>(0) = 1e-2;
  kf.processNoiseCov.at<float>(7) = 1e-2;
  // kf.processNoiseCov.at<float>(14) = 5.0f;
  // kf.processNoiseCov.at<float>(21) = 5.0f;
  kf.processNoiseCov.at<float>(14) = 10.0f;
  kf.processNoiseCov.at<float>(21) = 10.0f;
  kf.processNoiseCov.at<float>(28) = 1e-2;
  kf.processNoiseCov.at<float>(35) = 1e-2;

  strikerkf.processNoiseCov.at<float>(0) = 1e-2;
  strikerkf.processNoiseCov.at<float>(7) = 1e-2;
  strikerkf.processNoiseCov.at<float>(14) = 5.0f;
  strikerkf.processNoiseCov.at<float>(21) = 5.0f;
  strikerkf.processNoiseCov.at<float>(28) = 1e-2;
  strikerkf.processNoiseCov.at<float>(35) = 1e-2;

  playerstrikerkf.processNoiseCov.at<float>(0) = 1e-2;
  playerstrikerkf.processNoiseCov.at<float>(7) = 1e-2;
  playerstrikerkf.processNoiseCov.at<float>(14) = 5.0f;
  playerstrikerkf.processNoiseCov.at<float>(21) = 5.0f;
  playerstrikerkf.processNoiseCov.at<float>(28) = 1e-2;
  playerstrikerkf.processNoiseCov.at<float>(35) = 1e-2;

  // Initialize Measure Noise Covariance Matrix R
  // cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));
  cv::setIdentity(strikerkf.measurementNoiseCov, cv::Scalar(1e-1));
  cv::setIdentity(playerstrikerkf.measurementNoiseCov, cv::Scalar(1e-1));

  std::cout << "Kalman Initialised ..." << std::endl;


  // ------[ PARALLEL PIPELINE ] ----------------------------------------
  tbb::parallel_pipeline(13, // TBB NOTE: (recomendation) NumberOfFilters
    // 1st filter [ CAPTURE IMAGE ]
    tbb::make_filter<void, ProcessingChainData*>(tbb::filter::serial_in_order,
    [&](tbb::flow_control& fc)->ProcessingChainData*
    {
      // double t1 = (double)cv::getTickCount();
      // TBB NOTE: this filter feeds the input into the pipeline
      
        auto pData = new ProcessingChainData;
        // On succes do something with the image data. Here we create
        // a cv::Mat and save the image

        //abandon logic here if u uncomment it
        //char c = (char)cv::waitKey(1);

        if (cam.snapImage(sampleRate))
        {
          // t1 = (double)cv::getTickCount();

          memcpy(frame.data, cam.getImageData(), cam.getImageDataSize());

          if (frame.empty())
          {
            // std::cout << "empty frame ..." << std::endl;

            // double t2 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
            // pData->get_T = t2;

            // tell the TBB to stop the timeline
            fc.stop();
            return 0;
          }
          else
          { 
            // std::cout << "good ..." << std::endl;
            pData->img = frame.clone();

            // double t2 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
            // pData->get_T = t2;
          }
        }

        return pData;// On succes do something with the image data. Here we create
      }

      )&
      // 2nd filter [ CONVERT IMAGE TO BGR ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        cv::cvtColor(pData->img, pData->bgrImg, cv::COLOR_RGBA2BGR, 3);

        // pData->cv_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 
        
        return pData;
      }
      )&
      // 3rd filter [ UPLOAD TO GPU ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        pData->g_upImg.upload(pData->bgrImg);

        // double t2 = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 
        // pData->up_T = t2;
        
        return pData;
      }
      )&
      // 3rd filter [ UNDISTORT]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        cv::cuda::remap(pData->g_upImg, pData->g_unImg, map1, map2, cv::INTER_LINEAR);

        // pData->un_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency();
        
        return pData;
      }
      )&
      // 4th filter [ HOMOGRAPHY ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        while (!HOMOGRAPHY){
          //std::cout<<"calculating"<<std::endl;
          // char c = (char)cv::waitKey(1);
          
          pData->g_unImg.download(pData->unImg,cv::cuda::Stream::Null());
          pData->undistorttest = pData->unImg;
          //std::cout<<pData->unImg.empty()<<std::endl;
          // cv::imshow("Small  Image", pData->undistorttest);
          cv::Ptr<cv::aruco::Dictionary> Dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
          
          cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
          cv::aruco::detectMarkers(pData->undistorttest,Dictionary,markerCorners, ids, parameters);
          std::cout<<"ids size: "<<ids.size()<<std::endl;
          if (ids.size()>6){
            std::vector<cv::Point2f> centerlist;
            cv::RotatedRect test;

            for (const auto& corners : markerCorners) {
              cv::Point2f center(0.f,0.f);
              for (const auto& corner : corners){
                center += corner;
              }
              center /= 4.f;
              cv::circle(pData->undistorttest, center, 3, cv::Scalar(255,0,0));
              centerlist.push_back(center);
              
            }
            test = cv::minAreaRect(centerlist);
            test.points(boxPts.data());
            

            cv::Point2f bl;
            cv::Point2f br;
            cv::Point2f tl;
            cv::Point2f tr;
            
            bl = boxPts[0];
            tl = boxPts[1];
            tr = boxPts[2];
            br = boxPts[3];

            double width1 = cv::norm(br.x - bl.x);
            double width2 = cv::norm(tl.x - tr.x);

            double height1 = cv::norm(tl.y - bl.y);
            double height2 = cv::norm(tr.y - br.y);

            if (width1>width2){
              framewidth = width1;
            }
            else{
              framewidth = width2;
            }

            if (height1>height2){
              frameheight = height1;
            }
            else{
              frameheight = height2;
            }

            // Read destination image.
            // Four corners of the book in destination image.
            pts_dst.push_back(cv::Point2f(0, frameheight));
            pts_dst.push_back(cv::Point2f(0, 0));
            pts_dst.push_back(cv::Point2f(framewidth, 0));
            pts_dst.push_back(cv::Point2f(framewidth, frameheight));
            roi.width = framewidth - (offset_x*2);
            roi.height = frameheight - (offset_y*2);

            HOMOGRAPHY = true;
            h = cv::findHomography(boxPts, pts_dst);
          }
        }

        // Calculate Homography
        // double t1 = (double)cv::getTickCount();
        if (HOMOGRAPHY)
        {
          // src_gpu.upload(pData->unImg);
          cv::cuda::warpPerspective(pData->g_unImg, pData->src_gpu, h, cv::Size(framewidth,frameheight));
          pData->src_gpu.download(pData->unImg,cv::cuda::Stream::Null());
          // pData->hom_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency();
          return pData;
        }
      }
      )&
      // 5th filter [ DOWNLOAD TO CPU ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        pData->src_gpu.download(pData->unImg);

        // double t2 = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 
        // pData->dn_T = t2;
        
        return pData;
      }
      )&
      // 6th filter [ CROP , ROTATE AND RESIZE IMAGE ] 
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();
        

        pData->cropImg = pData->unImg(roi);
        cv::resize(pData->cropImg, pData->smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
        cv::rotate(pData->smallImg, pData->smallImg, cv::ROTATE_90_COUNTERCLOCKWISE);
        // pData->re_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 

        return pData;
      }
      )&
      // 7th filter [ INVERT IMAGE ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        cv::bitwise_not(pData->smallImg, pData->notImg);

        // pData->in_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 

        return pData;
      }
      )&
      // 8th filter [ CONVERT TO HSV SPACE ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        // Convert image frame to HSV color space
        cv::cvtColor(pData->notImg, pData->hsvImg, cv::COLOR_BGR2HSV);

        // pData->hsv_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 

        return pData;
      }
      )&
      // 9th filter [ THRESH HSV IMAGE ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        cv::inRange(pData->hsvImg, minThresh, maxThresh, pData->threshImg);
        cv::inRange(pData->hsvImg, minstrikerThresh, maxstrikerThresh, pData->striker_thresh);
        cv::inRange(pData->hsvImg, minplayerstrikerThresh, maxplayerstrikerThresh, pData->player_striker_thresh);


        // pData->th_T  = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 

        return pData;
      }
      )&
      // 10th filter [ MORPH IMAGE - OPEN ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        cv::morphologyEx(pData->threshImg, pData->morph1Img, cv::MORPH_OPEN, element);
        cv::morphologyEx(pData->striker_thresh, pData->striker_thresh, cv::MORPH_OPEN, element);
       
        cv::morphologyEx(pData->player_striker_thresh, pData->player_striker_thresh, cv::MORPH_OPEN, element);
        cv::dilate(pData->player_striker_thresh, pData->player_striker_thresh, element);

        // pData->mo1_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency();

        return pData;
      }
      )&
      // 11th filter [ SEGMENT STRIKER AND UPDATE KALMAN ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        // find contours in the mask image
        cv::findContours(pData->striker_thresh, pData->strikercontours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Only proceed if at least one contour was found
        if (pData->strikercontours.size() > 0)
        {
          // [PROGRESS CHECK]
          //std::cout << "Contour found!" << std::endl;
          for (int idx = 0; idx < pData->strikercontours.size(); idx++)
          {
            // Loop through contour array and only use the one within puck dimensions
            double objArea = cv::contourArea(pData->strikercontours[idx]);

            if (objArea >= 100.0 && objArea < 1200.0)
            {
              // [VALUE CHECK]
             //std::cout << "Contour area = " << objArea << std::endl;

              // Calculate the radius and center coordinates for the puck
              cv::minEnclosingCircle(pData->strikercontours[idx], pData->strikercenter, pData->strikerobjRadius);
              // Create circle outline for visualisation
              cv::circle(pData->smallImg, pData->strikercenter, pData->strikerobjRadius, blackCol, 1);

              strikermeas.at<float>(0) = pData->strikercenter.x;         // Centroid of the object (x)
              strikermeas.at<float>(1) = pData->strikercenter.y;         // Centroid of the object (y)
              strikermeas.at<float>(2) = (float)pData->strikerobjRadius; // Size of the object (x)
              strikermeas.at<float>(3) = (float)pData->strikerobjRadius; // Size of the object (y)

              if (!STRIKERFOUND) // First detection. Initialize Kalman filter
              {
                // >>>> Initialization
                strikerkf.errorCovPre.at<float>(0) = 1; // px
                strikerkf.errorCovPre.at<float>(7) = 1; // px
                strikerkf.errorCovPre.at<float>(14) = 1;
                strikerkf.errorCovPre.at<float>(21) = 1;
                strikerkf.errorCovPre.at<float>(28) = 1; // px
                strikerkf.errorCovPre.at<float>(35) = 1; // px

                strikerstate.at<float>(0) = strikermeas.at<float>(0); // Centroid of the object (x)
                strikerstate.at<float>(1) = strikermeas.at<float>(1); // Centroid of the object (y)
                strikerstate.at<float>(2) = 0;                 // Velocity of the object (x)
                strikerstate.at<float>(3) = 0;                 // Velocity of the object (y)
                strikerstate.at<float>(4) = strikermeas.at<float>(2); // Size of the object (x)
                strikerstate.at<float>(5) = strikermeas.at<float>(3); // Size of the object (y)
                // <<<< Initialization

                strikerkf.statePost = strikerstate;

                STRIKERFOUND = true;
                //pData->_kal = 1;
              }
              else // Update Kalman
              {
                //pData->_kal = 1;
                strikerkf.correct(strikermeas); // Kalman Correction
              }
            }
          }
        }

        // [ KALMAN TRACKING ]
        /* [Note:] 
        [!] Set dT at each processing step!
          - Check if Kalman is already in action
          - If found:
	          - update matrix A
              - predict state
              - predict trajectory
          - Else, apply kalman correction
	      */

        double strikerprecTick = strikerticks;
        strikerticks = (double)cv::getTickCount();
        double strikerdT = (strikerticks - strikerprecTick) / cv::getTickFrequency(); // in seconds

        if (STRIKERFOUND) // If segmentation was successful
        {
          // Update matrix A
          strikerkf.transitionMatrix.at<float>(2) = strikerdT;
          strikerkf.transitionMatrix.at<float>(9) = strikerdT;

          // Update striker state
          strikerstate = strikerkf.predict();

          // Show new circle for representation
          cv::circle(pData->smallImg, cv::Point2f(strikerstate.at<float>(0), strikerstate.at<float>(1)), strikerstate.at<float>(4), blueCol, 2);
          
          if (!isnan(strikerstate.at<float>(2)) && !isnan(strikerstate.at<float>(3)))
          {
            pData->strikerDx = strikerstate.at<float>(2);
            pData->strikerDy = strikerstate.at<float>(3);
          }

          // pData->_kal = 1;

        }
        else // No striker detected through Kalman
        {
          // [PROGRESS CHECK]
          //std::cout << "/// NO PUCK DETECTED ///" << std::endl << std::endl;

          strikerkf.correct(strikermeas); // Kalman Correction
          // pData->_kal = 0;
        }

        // pData->se_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 

        //coordinate system calibration go to each corner
        if (!CALIBRATED){
          if (loop==1000){
            std::cout<<"point 1"<<std::endl;
            client.send("(0.00,0.00)", 11);
            tl = detectstriker(pData->striker_thresh);
            std::cout<<tl<<std::endl;
            std::cout<<tl.x<<std::endl;
            xoffset = tl.x;
            yoffset = tl.y;
            // CALIBRATED1 = true;
            loop++;
          }
          else if (loop == 2000){
            std::cout<<"point 2"<<std::endl;
            client.send("(1.00,0.00)", 11);
            loop ++;
          }
          else if (loop == 3000){
            tr = detectstriker(pData->striker_thresh);
            std::cout<<tr<<std::endl;
            loop++;
          }
          else if (loop == 4000){
    
            std::cout<<"point 3"<<std::endl;
            client.send("(1.00,1.00)", 11);
            loop ++;
          }
          else if (loop == 5000){
            
            br = detectstriker(pData->striker_thresh);
            std::cout<<br<<std::endl;

            xscale = tr.x - tl.x;
            yscale = br.y-tr.y;

            std::vector<float> b;

            b.push_back(tl.x);
            b.push_back(tl.y);
            b.push_back(tr.x-tl.x);
            b.push_back(br.y-tr.y);


            CALIBRATED = true;
            loop = 0;
            std::ofstream calibration_output_file("calibration.txt");
            std::ostream_iterator<float> output_iterator(calibration_output_file, "\n" );
            std::copy ( b.begin( ), b.end( ), output_iterator); 

            std::cout<< "x_offset is "<<tl.x<<std::endl;
            std::cout<< "y_offset is "<<tl.y<<std::endl;
            std::cout<< "x_diff is "<<tr.x-tl.x<<std::endl;
            std::cout<< "y_diff is "<<br.y-tr.y<<std::endl;

            std::cout<<"Calibration Finished"<<std::endl; 
          }
          else {
            loop ++;
          }
        }

        // pData->segpuck_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency();
        return pData;
      }
      )&
      // 12th filter [ SEGMENT PLAYER STRIKER AND UPDATE KALMAN ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        // find contours in the mask image
        cv::findContours(pData->player_striker_thresh, pData->playerstrikercontours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Only proceed if at least one contour was found
        if (pData->playerstrikercontours.size() > 0)
        {
          // [PROGRESS CHECK]
          //std::cout << "Contour found!" << std::endl;
          for (int idx = 0; idx < pData->playerstrikercontours.size(); idx++)
          {
            // Loop through contour array and only use the one within puck dimensions
            double objArea = cv::contourArea(pData->playerstrikercontours[idx]);

            if (objArea >= 200.0 && objArea < 1200.0)
            {
              // [VALUE CHECK]
             //std::cout << "Contour area = " << objArea << std::endl;

              // Calculate the radius and center coordinates for the puck
              cv::minEnclosingCircle(pData->playerstrikercontours[idx], pData->playerstrikercenter, pData->playerstrikerobjRadius);
              // Create circle outline for visualisation
              cv::circle(pData->smallImg, pData->playerstrikercenter, pData->playerstrikerobjRadius, blackCol, 1);

              playerstrikermeas.at<float>(0) = pData->playerstrikercenter.x;         // Centroid of the object (x)
              playerstrikermeas.at<float>(1) = pData->playerstrikercenter.y;         // Centroid of the object (y)
              playerstrikermeas.at<float>(2) = (float)pData->playerstrikerobjRadius; // Size of the object (x)
              playerstrikermeas.at<float>(3) = (float)pData->playerstrikerobjRadius; // Size of the object (y)

              if (!PLAYERSTRIKERFOUND) // First detection. Initialize Kalman filter
              {
                // >>>> Initialization
                playerstrikerkf.errorCovPre.at<float>(0) = 1; // px
                playerstrikerkf.errorCovPre.at<float>(7) = 1; // px
                playerstrikerkf.errorCovPre.at<float>(14) = 1;
                playerstrikerkf.errorCovPre.at<float>(21) = 1;
                playerstrikerkf.errorCovPre.at<float>(28) = 1; // px
                playerstrikerkf.errorCovPre.at<float>(35) = 1; // px

                playerstrikerstate.at<float>(0) = playerstrikermeas.at<float>(0); // Centroid of the object (x)
                playerstrikerstate.at<float>(1) = playerstrikermeas.at<float>(1); // Centroid of the object (y)
                playerstrikerstate.at<float>(2) = 0;                 // Velocity of the object (x)
                playerstrikerstate.at<float>(3) = 0;                 // Velocity of the object (y)
                playerstrikerstate.at<float>(4) = playerstrikermeas.at<float>(2); // Size of the object (x)
                playerstrikerstate.at<float>(5) = playerstrikermeas.at<float>(3); // Size of the object (y)
                // <<<< Initialization

                playerstrikerkf.statePost = playerstrikerstate;

                PLAYERSTRIKERFOUND = true;
                //pData->_kal = 1;
              }
              else // Update Kalman
              {
                //pData->_kal = 1;
                playerstrikerkf.correct(playerstrikermeas); // Kalman Correction
              }
            }
          }
        }
        else
        {
          PLAYERSTRIKERFOUND = false;
        }
        // [ KALMAN TRACKING ]
        /* [Note:] 
        [!] Set dT at each processing step!
          - Check if Kalman is already in action
          - If found:
	          - update matrix A
              - predict state
              - predict trajectory
          - Else, apply kalman correction
	      */

        double playerstrikerprecTick = playerstrikerticks;
        playerstrikerticks = (double)cv::getTickCount();
        double playerstrikerdT = (playerstrikerticks - playerstrikerprecTick) / cv::getTickFrequency(); // in seconds



        if (PLAYERSTRIKERFOUND) // If segmentation was successful
        {
          // Update matrix A
          playerstrikerkf.transitionMatrix.at<float>(2) = playerstrikerdT;
          playerstrikerkf.transitionMatrix.at<float>(9) = playerstrikerdT;

          // Update puck state
          playerstrikerstate = playerstrikerkf.predict();

          // Show new circle for representation
          cv::circle(pData->smallImg, cv::Point2f(playerstrikerstate.at<float>(0), playerstrikerstate.at<float>(1)), playerstrikerstate.at<float>(4), redCol, 2);
          // std::cout<<"printing player striker"<<std::endl;
          // if (!isnan(strikerstate.at<float>(2)) && !isnan(strikerstate.at<float>(3)))
          // {
          //   // visualise complete trajectory without intersection
          //   cv::Point2f targetPos((state.at<float>(0) + state.at<float>(2)), (state.at<float>(1) + state.at<float>(3)));

          //   //std::cout << "*** VISUALISE NO INTERSECTION ***" << std::endl;
          //   //std::cout << "*** ------------------------- ***" << std::endl << std::endl;
          //   // cv::line(pData->smallImg, cv::Point2f(state.at<float>(0), state.at<float>(1)), targetPos, greenCol, 2);
          //   ImageProcessing test;
          //   test.predictTrajectory(pData->smallImg, pData->center, state.at<float>(4), state.at<float>(2), state.at<float>(3));
          // }
          
          // pData->_kal = 1;



        }
        else // No puck detected through Kalman
        {
          // [PROGRESS CHECK]
          //std::cout << "/// NO PUCK DETECTED ///" << std::endl << std::endl;

          playerstrikerkf.correct(playerstrikermeas); // Kalman Correction
          // pData->_kal = 0;
        }

        // pData->seghand_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency();

        return pData;
      }
      )&
      // 13th filter [ SEGMENT PUCK AND UPDATE KALMAN ]
      tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
      {
        // double t1 = (double)cv::getTickCount();

        // find contours in the mask image
        cv::findContours(pData->morph1Img, pData->contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Only proceed if at least one contour was found
        if (pData->contours.size() > 0)
        {
          // [PROGRESS CHECK]
          //std::cout << "Contour found!" << std::endl;
          for (int idx = 0; idx < pData->contours.size(); idx++)
          {
            // Loop through contour array and only use the one within puck dimensions
            double objArea = cv::contourArea(pData->contours[idx]);

            if (objArea >= 30.0 && objArea < 800.0)   //  100, 800
            {
              notFoundCount = 0;
              // [VALUE CHECK]
             //std::cout << "Contour area = " << objArea << std::endl;

              // Calculate the radius and center coordinates for the puck
              cv::minEnclosingCircle(pData->contours[idx], pData->center, pData->objRadius);
              // Create circle outline for visualisation
              cv::circle(pData->smallImg, pData->center, pData->objRadius, greenCol, 1);

              meas.at<float>(0) = pData->center.x;         // Centroid of the object (x)
              meas.at<float>(1) = pData->center.y;         // Centroid of the object (y)
              meas.at<float>(2) = (float)pData->objRadius; // Size of the object (x)
              meas.at<float>(3) = (float)pData->objRadius; // Size of the object (y)

              if (!FOUND) // First detection. Initialize Kalman filter
              {

                // client.send("(0.50,0.20)",11);
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0); // Centroid of the object (x)
                state.at<float>(1) = meas.at<float>(1); // Centroid of the object (y)
                state.at<float>(2) = 0;                 // Velocity of the object (x)
                state.at<float>(3) = 0;                 // Velocity of the object (y)
                state.at<float>(4) = meas.at<float>(2); // Size of the object (x)
                state.at<float>(5) = meas.at<float>(3); // Size of the object (y)
                // <<<< Initialization

                kf.statePost = state;

                FOUND = true;
                //pData->_kal = 1;
              }
              else // Update Kalman
              {
                //pData->_kal = 1;
                kf.correct(meas); // Kalman Correction
              }
            }
          }
        }
        else // Update Kalman filter check
        {
          notFoundCount++;
          //pData->_kal = 0;

          if (notFoundCount >= 10)// Lost sight of puck
          {
            FOUND = false;    
          }
        }

        // [ KALMAN TRACKING ]
        /* [Note:] 
        [!] Set dT at each processing step!
          - Check if Kalman is already in action
          - If found:
	          - update matrix A
              - predict state
              - predict trajectory
          - Else, apply kalman correction
	      */

        double precTick = ticks;
        ticks = (double)cv::getTickCount();
        double dT = (ticks - precTick) / cv::getTickFrequency(); // in seconds
        
        //image to show trajectory system
        pData->trajectoryImg = pData->smallImg;
        //pData->smallImg.copyTo(pData->trajectoryImg);
        
        //trajectory system debugging
        // ImageProcessing test1;
        // test1.predictTrajectoryMultiplePoints(pData->trajectoryImg,cv::Point2f(150,350),9.45f, -283, -165);
        
        if (FOUND) // If segmentation was successful
        {
          // Update matrix A
          kf.transitionMatrix.at<float>(2) = dT;
          kf.transitionMatrix.at<float>(9) = dT;

          // Update puck state
          state = kf.predict();

          // Show new circle for representation
          cv::circle(pData->smallImg, cv::Point2f(state.at<float>(0), state.at<float>(1)), state.at<float>(4), blueCol, 2);

          if (!isnan(state.at<float>(2)) && !isnan(state.at<float>(3)))
          {
            pData->dx = state.at<float>(2);
            pData->dy = state.at<float>(3);

            // test.predictTrajectoryMultiplePoints(pData->trajectoryImg,pData->center,state.at<float>(4), state.at<float>(2), state.at<float>(3));
            // if (isnan(pData->center.x))
            // {
            //   pData->center = cv::Point2f (300, 100);
            // }
            
            if (pData->center == cv::Point2f(0, 0)) //  if puck tracking momentarily lost, use kalman predicted position
            {
              pData->center = cv::Point2f(state.at<float>(0), state.at<float>(1));
            }
            if (pData->objRadius < 1)
            {
              pData->objRadius = state.at<float>(4);
            }

            //predict the trajectory from kalmanfilter result
            pData->puckBounces = predictTrajectoryMultiplePoints(pData->trajectoryImg,pData->center, state.at<float>(4), state.at<float>(2), state.at<float>(3));
            //go on double free error
            //delete &test;
          }
          
          // pData->_kal = 1;
        }
        else // No puck detected through Kalman
        {
          // [PROGRESS CHECK]
          kf.correct(meas); // Kalman Correction
          // pData->_kal = 0;
        }

        cv::Point2i cen(pData->center.x,pData->center.y);
        std::stringstream circleprint;
        circleprint << "(" << cen.x << "," << cen.y << ")";

        cv::Point2i vel(state.at<float>(2),state.at<float>(3));
        std::stringstream velprint;
        velprint << "(" << vel.x << "," << vel.y << ")";

        cv::putText(pData->smallImg,circleprint.str(), cv::Point2f(60,30), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
        cv::putText(pData->smallImg,velprint.str(), cv::Point2f(60,50), cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),1,cv::LINE_AA);
        // convert_send(pData->center);
        // pData->se_T = ((double)cv::getTickCount() - t1)/cv::getTickFrequency(); 
        return pData;
      }
      )&
      // 15th filter
      tbb::make_filter<ProcessingChainData*,void>(tbb::filter::serial_in_order,
                                                 [&](ProcessingChainData *pData)
      {   // TBB NOTE: pipeline end point. dispatch to GUI
        try
        {
          guiQueue.push(pData);
        }
        catch (...)
        {
          std::cout << "Pipeline caught an exception on the queue" << std::endl;
        }
      }
      )
      );
}



int main(int argc, char **argv)
{
  std::cout << "---[ MAIN FUNCTION START ]---" << std::endl;

  std::cout << "---Starting: Main Initialisation---" << std::endl;
  // Initialise gstreamer
  gst_init(&argc, &argv);

  // Initialise camera properties 
  // [NOTE]
  // Video parameters/camera properties are initialised here for ease of access when changing and visibility
  // for reference. These will be passed on to the ImageProcessing object to initialise
  // the paramters so you only need to change them in the struct.
  //CameraProperties cp;
  //cv::Rect cropROI(cp.crop_x, cp.crop_y, (cp.width - cp.crop_x), (cp.height - cp.crop_y));

  // Initialise the TcamCamera object "cam" from The Imaging Source.
  // [NOTE]
  // This must be done with the serial number of the camera which is:
  // > 25020018 < 
  // [!] The camera feed is not possible to directly access with opencv functions
  //TcamImage cam(cp.serialNum);

  // Set a color video format, resolution and frame rate
  //cam.set_capture_format("BGRx", gsttcam::FrameSize{cp.width, cp.height}, gsttcam::FrameRate{cp.rate, 1});
  // Initialise the ImageProcessing object "camera" and initialise data parameters
  // [NOTE]
  // Optimising code for OpenCV CUDA modules requires that return arrays are pre-allocated
  // since memory allocation is a large cause for delay in running the OpenCV GPU modules
  std::cout << "---Complete: Main Initialisation---" << std::endl;
  std::string isRecording;
  std::cout<<"Would you like to run recording (y/n)"<<std::endl;
  std::cin >> isRecording;
  if(isRecording=="y")
  {
    recording = true;
    std::cout<<"Record start"<<std::endl;
   
    
    int frameMax = 10000000;   //Mumber of frames (original is 10000)
    trackVideoFeed.reserve(frameMax);

  }

  //reserve the memory space for csv record
  csvRecord.reserve(200000);

  std::string calq;
  std::cout<<"Would you like to run Calibration? (y/n)"<<std::endl;
  std::cin >> calq;
  if  (calq == "y"){
    CALIBRATED = false;
    std::cout<<"Running Calibration"<<std::endl;
  }
  else {

    std::cout.flush();


    double tempVar;

    while ( calibration_file >> tempVar)
    {
      calibration_inputs.push_back(tempVar);
    }


    x_offset = calibration_inputs[0];
    y_offset = calibration_inputs[1];
    x_diff = calibration_inputs[2];
    y_diff = calibration_inputs[3];

    std::cout<<"Calibration not selected, using saved variables"<<std::endl;

    std::cout<<"x_offset is "<<x_offset<<std::endl;
    std::cout<<"y_offset is "<<y_offset<<std::endl;
    std::cout<<"x_diff is "<<x_diff<<std::endl;
    std::cout<<"y_diff is "<<y_diff<<std::endl;
    client.send("(0.50,0.80)",11);

  }

  startPipeline();
  return 0;
}

 void startPipeline()
  {   
    /* [ tbb pipeline initialization ]
      -initialize the camera
      -start the pipeline

      Pipeline:
      - Capture image
      - Convert to BGR (3)
      - undistort / crop img
      - smooth (Gauss)
      - Invert image and thresh
      - segment (contours)
      - Kalman filter (tracking)
      - Visualise image
    */
    // [ CAMERA ]
    // Initialise the TcamCamera object "cam" from The Imaging Source.
    // [NOTE]
    // This must be done with the serial number of the camera which is:
    // > 25020018 < 
    // [!] The camera feed is not possible to directly access with opencv functions
    TcamImage cam("25020018"); 


    CameraProperties cp;
    cam.set_capture_format("RGBx", gsttcam::FrameSize{cp.width, cp.height}, gsttcam::FrameRate{cp.frameRate_n, cp.frameRate_d});

    //Image information (not used)
    int frameHeight = cam.getHeight();
    std::cout<<"frame height is  "<<frameHeight<<" ,cp height is  "<<cp.height<<std::endl;
    int frameWidth = cam.getWidth();
    int pixls = cam.getBytesPerPixel();
    int frameRate = (cp.frameRate_n / cp.frameRate_d);
    int sampleRate = 2 * frameRate;

    // [ VIDEO writer ]
    // cv::Size frameSize(685, 410);
    // double videoFrameRate = 150;
    // string videoName = "trackVideo_150.avi";
    
    // cv::VideoWriter trackVideo(videoName, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), videoFrameRate, frameSize, true);
    // //cv::VideoWriter threshVideo("threshVideo_150.avi", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), videoFrameRate, frameSize, true);
    // std::vector<cv::Mat> threshVideoFeed, trackVideoFeed;
    
    // int frameMax = 10000000;   //Mumber of frames (original is 10000)
    // trackVideoFeed.reserve(frameMax);
    // threshVideoFeed.reserve(frameMax);

    //put it in the pipeline to push video
    // trackVideoFeed.push_back(pData->smallImg);

    // [ UNDISTORT img parameter ]
    int imgWidth = 1440;
    int imgHeight = 1080;
    cv::Size imgSize = cv::Size(imgWidth, imgHeight);
    cv::Mat cameraMatrix, distCoeffs, map1, map2;
    
    // Read calibration file 
    cv::FileStorage fs("default.xml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();
    // Get optimal camera matrix
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0);
    // Create undistort maps for full image
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                newCameraMatrix, imgSize,
                                CV_32FC1, map1, map2);

    cv::cuda::GpuMat map_x;
    cv::cuda::GpuMat map_y;
    map_x = cv::cuda::GpuMat(map1);
    map_y = cv::cuda::GpuMat(map2);

    //dont forget to start camera
    cam.start();

    //start the pipeline 
    tbb::concurrent_bounded_queue<ProcessingChainData *> guiQueue;
    guiQueue.set_capacity(15); // TBB NOTE: flow control so the pipeline won't fill too much RAM
    auto pipelineRunner = std::thread(imageProcessingTBB, std::ref(cam),
                                      std::ref(guiQueue), map_x, map_y);

    std::cout << "Pipeline Initialised ..." << std::endl;
    // TBB NOTE: GUI is executed in main thread
    ProcessingChainData *pData = 0;

    std::cout << "strategy loop start..." << std::endl;
    // std::cout << guiQueue.try_pop(pData) << std::endl;

    //  CALCULATE STRIKER ACCELERATION
    time_t prevTime = time(NULL);
    float prevStrikerDx = 0;
    float prevStrikerDy = 0;
    std::vector<float> maxStrikerAccVel;
    maxStrikerAccVel.push_back(0);
    maxStrikerAccVel.push_back(0);
    maxStrikerAccVel.push_back(0);
    maxStrikerAccVel.push_back(0);

    //  READ MAX STRIKER ACCELERATION AND VELOCITY
    std::ifstream strikerData;
    strikerData.open("strikerAccVel.csv");  //  d2x, d2y, dx, dy
    if (!strikerData.is_open())
    {
      // std::cout<<"FILENOTOPEN"<<std::endl;
    }
    //  READ CURRENT TOP ACCEL & VEL
    std::vector<std::string> fileStrVector;
    std::string fileString, strStrikerData;
    std::getline(strikerData, fileString);

    std::stringstream s(fileString);
    while (std::getline(s, strStrikerData, ','))
    {
      fileStrVector.push_back(strStrikerData);
    }

    std::vector<float> fileVector;
    for (int i = 0; i < fileStrVector.size(); i++)
    {
      fileVector.push_back(stof(fileStrVector[i]));
    }

    if (fileVector.size() == 4)
    {  
      // std::cout << fileVector[0] << ", " << fileVector[1] << ", " << fileVector[2] << ", " << fileVector[3] << std::endl;

      strikerMaxAccelerationX = fileVector[0];
      strikerMaxAccelerationY = fileVector[1];
      strikerMaxVelocityX = fileVector[2];
      strikerMaxVelocityY = fileVector[3];
    }
    strikerData.close();


    while(true){
      // std::cout << "while loop start..." << std::endl;
      if (guiQueue.try_pop(pData))
      {   
        char c = (char)cv::waitKey(1);
        if(c==27)
        { 
          std::cout << "closing the program" << std::endl;
          client.send("(0.50,0.80)", 11);
           // [ VIDEO writer ]
          if(recording)
          {
            double videoFrameRate = 60;
            std::string videoName = "trackVideo_60.avi";
            trackVideo.open(videoName, cv::VideoWriter::fourcc('a','v', 'c', '1'), videoFrameRate, cv::Size(trackVideoFeed[0].size()), true);
            //cv::VideoWriter threshVideo("threshVideo_150.avi", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), videoFrameRate, frameSize, true);
           
            for (int i =0;i<trackVideoFeed.size();i++) 
            {
              trackVideo.write(trackVideoFeed[i]);
            }

            trackVideo.release();
            std::cout << "saved done, we have  " <<trackVideoFeed.size()<<" frames."<< std::endl;
          }
          //  UPDATE STRIKER TOP ACCEL & VEL
          std::ofstream strikerDataNew;
          strikerDataNew.open("strikerAccVelNew.csv");  //  d2x, d2y, dx, dy
          if (!strikerDataNew.is_open())
          {
            // std::cout<<"FILE2NOTOPEN"<<std::endl;
          }
          if (strikerDataNew.is_open())
          {
            strikerDataNew << std::to_string(maxStrikerAccVel[0]) << ",";
            strikerDataNew << std::to_string(maxStrikerAccVel[1]) << ",";
            strikerDataNew << std::to_string(maxStrikerAccVel[2]) << ",";
            strikerDataNew << std::to_string(maxStrikerAccVel[3]) << "\n";
            strikerDataNew.close();
            remove("strikerAccVel.csv");
            rename("strikerAccVelNew.csv", "strikerAccVel.csv");
          }

          // //  Catch Performance CSV Output
          // std::ofstream catchDataOutput;
          // catchDataOutput.open("catchPerformanceData.csv");
          // if (!catchDataOutput.is_open())
          // {

          // }
          // else
          // {
          //   catchDataOutput<<"catchID,dx,dy,puckSpeed,prevStatus,squareup\n";
          //   for(int i=0;i<catchInfo.size();i++)
          //   {
          //     for(int j=0;j<6;j++)
          //     { 
          //       if(j==5)
          //       {
          //         catchDataOutput << std::to_string(catchInfo[i][j]) << "\n";
          //       }
          //       else
          //       {
          //         catchDataOutput << std::to_string(catchInfo[i][j]) << ",";
          //       }
          //     }
          //   }
          //   catchDataOutput.close();
          // }

          break;
        }

        // double t1 = (double)cv::getTickCount();
        //to match the number of loop with video
        count++;
        std::cout<<count<<"th loop start ----------------------------------------- "<<std::endl;
        
        // cv::imshow("Circles  Image", pData->cimage);

    

        // cv::imshow("Striker Thresh Image", pData->striker_thresh);


        // cv::imshow("Invert Image", pData->notImg);
        // cv::imshow("Thresh Image", pData->threshImg);
        // std::cout << "why?" << std::endl;
        

        cv::imshow("TrajectoryImg", pData->trajectoryImg);
       
       
        //strategy
        //  std::cout << "while loop start..." << std::endl;

        //  CALCULATE STRIKER ACCELERATION
        float strikerD2x;
        float strikerD2y;
        time_t currTime = time(NULL);
        double timeElapsed = difftime(currTime, prevTime);
        if (timeElapsed == 0)
        {
          strikerD2x = 0;
          strikerD2y = 0;
        }
        else
        {
          strikerD2x = (pData->strikerDx - prevStrikerDx)/timeElapsed;
          strikerD2y = (pData->strikerDy - prevStrikerDy)/timeElapsed;
          prevStrikerDx = pData->strikerDx;
          prevStrikerDy = pData->strikerDy;
          prevTime = currTime;
        }

        if (maxStrikerAccVel[0] < abs(strikerD2x))
        {
          maxStrikerAccVel[0] = abs(strikerD2x);
        }
        if (maxStrikerAccVel[1] < abs(strikerD2y))
        {
          maxStrikerAccVel[1] = abs(strikerD2y);
        }
        if (maxStrikerAccVel[2] < abs(pData->strikerDx))
        {
          maxStrikerAccVel[2] = abs(pData->strikerDx);
        }
        if (maxStrikerAccVel[3] < abs(pData->strikerDy))
        {
          maxStrikerAccVel[3] = abs(pData->strikerDy);
        }

        //  CONTROL LOGIC
        controlLogic(pData->puckBounces, pData->center, pData->objRadius, pData->strikercenter, pData->strikerobjRadius, pData->dx, pData->dy, pData->playerstrikercenter, pData->playerstrikerobjRadius, pData->smallImg);
                
        cv::imshow("Small  Image", pData->smallImg);
        trackVideoFeed.push_back(pData->smallImg);
        //for checking bounces
        prevDx = pData->dx;
        prevDy = pData->dy;
          
       
        // double t2 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
        // pData->vi_T = t2;
        delete pData;
        pData = 0;
      }
    }

  }

