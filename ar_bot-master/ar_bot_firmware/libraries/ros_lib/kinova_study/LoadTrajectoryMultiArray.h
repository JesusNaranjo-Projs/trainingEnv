#ifndef _ROS_kinova_study_LoadTrajectoryMultiArray_h
#define _ROS_kinova_study_LoadTrajectoryMultiArray_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "kinova_study/LoadTrajectory.h"

namespace kinova_study
{

  class LoadTrajectoryMultiArray : public ros::Msg
  {
    public:
      uint32_t trajectories_length;
      typedef kinova_study::LoadTrajectory _trajectories_type;
      _trajectories_type st_trajectories;
      _trajectories_type * trajectories;

    LoadTrajectoryMultiArray():
      trajectories_length(0), st_trajectories(), trajectories(nullptr)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->trajectories_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->trajectories_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->trajectories_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->trajectories_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->trajectories_length);
      for( uint32_t i = 0; i < trajectories_length; i++){
      offset += this->trajectories[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      uint32_t trajectories_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      trajectories_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      trajectories_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      trajectories_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->trajectories_length);
      if(trajectories_lengthT > trajectories_length)
        this->trajectories = (kinova_study::LoadTrajectory*)realloc(this->trajectories, trajectories_lengthT * sizeof(kinova_study::LoadTrajectory));
      trajectories_length = trajectories_lengthT;
      for( uint32_t i = 0; i < trajectories_length; i++){
      offset += this->st_trajectories.deserialize(inbuffer + offset);
        memcpy( &(this->trajectories[i]), &(this->st_trajectories), sizeof(kinova_study::LoadTrajectory));
      }
     return offset;
    }

    virtual const char * getType() override { return "kinova_study/LoadTrajectoryMultiArray"; };
    virtual const char * getMD5() override { return "b72d868fac337953cbdf832c795b2bf1"; };

  };

}
#endif
