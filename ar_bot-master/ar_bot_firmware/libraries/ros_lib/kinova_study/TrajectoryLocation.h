#ifndef _ROS_kinova_study_TrajectoryLocation_h
#define _ROS_kinova_study_TrajectoryLocation_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace kinova_study
{

  class TrajectoryLocation : public ros::Msg
  {
    public:
      typedef const char* _trajectory_one_type;
      _trajectory_one_type trajectory_one;
      typedef const char* _trajectory_two_type;
      _trajectory_two_type trajectory_two;

    TrajectoryLocation():
      trajectory_one(""),
      trajectory_two("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      uint32_t length_trajectory_one = strlen(this->trajectory_one);
      varToArr(outbuffer + offset, length_trajectory_one);
      offset += 4;
      memcpy(outbuffer + offset, this->trajectory_one, length_trajectory_one);
      offset += length_trajectory_one;
      uint32_t length_trajectory_two = strlen(this->trajectory_two);
      varToArr(outbuffer + offset, length_trajectory_two);
      offset += 4;
      memcpy(outbuffer + offset, this->trajectory_two, length_trajectory_two);
      offset += length_trajectory_two;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      uint32_t length_trajectory_one;
      arrToVar(length_trajectory_one, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_trajectory_one; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_trajectory_one-1]=0;
      this->trajectory_one = (char *)(inbuffer + offset-1);
      offset += length_trajectory_one;
      uint32_t length_trajectory_two;
      arrToVar(length_trajectory_two, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_trajectory_two; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_trajectory_two-1]=0;
      this->trajectory_two = (char *)(inbuffer + offset-1);
      offset += length_trajectory_two;
     return offset;
    }

    virtual const char * getType() override { return "kinova_study/TrajectoryLocation"; };
    virtual const char * getMD5() override { return "1528c9e70d96e1274837680ec7d852eb"; };

  };

}
#endif
