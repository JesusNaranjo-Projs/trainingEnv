#ifndef _ROS_kinova_study_TrajectoryInfo_h
#define _ROS_kinova_study_TrajectoryInfo_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "kinova_study/LoadTrajectory.h"

namespace kinova_study
{

  class TrajectoryInfo : public ros::Msg
  {
    public:
      uint32_t trajectories_length;
      typedef kinova_study::LoadTrajectory _trajectories_type;
      _trajectories_type st_trajectories;
      _trajectories_type * trajectories;
      uint32_t grasp_region_length;
      typedef int8_t _grasp_region_type;
      _grasp_region_type st_grasp_region;
      _grasp_region_type * grasp_region;

    TrajectoryInfo():
      trajectories_length(0), st_trajectories(), trajectories(nullptr),
      grasp_region_length(0), st_grasp_region(), grasp_region(nullptr)
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
      *(outbuffer + offset + 0) = (this->grasp_region_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->grasp_region_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->grasp_region_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->grasp_region_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->grasp_region_length);
      for( uint32_t i = 0; i < grasp_region_length; i++){
      union {
        int8_t real;
        uint8_t base;
      } u_grasp_regioni;
      u_grasp_regioni.real = this->grasp_region[i];
      *(outbuffer + offset + 0) = (u_grasp_regioni.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->grasp_region[i]);
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
      uint32_t grasp_region_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      grasp_region_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      grasp_region_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      grasp_region_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->grasp_region_length);
      if(grasp_region_lengthT > grasp_region_length)
        this->grasp_region = (int8_t*)realloc(this->grasp_region, grasp_region_lengthT * sizeof(int8_t));
      grasp_region_length = grasp_region_lengthT;
      for( uint32_t i = 0; i < grasp_region_length; i++){
      union {
        int8_t real;
        uint8_t base;
      } u_st_grasp_region;
      u_st_grasp_region.base = 0;
      u_st_grasp_region.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->st_grasp_region = u_st_grasp_region.real;
      offset += sizeof(this->st_grasp_region);
        memcpy( &(this->grasp_region[i]), &(this->st_grasp_region), sizeof(int8_t));
      }
     return offset;
    }

    virtual const char * getType() override { return "kinova_study/TrajectoryInfo"; };
    virtual const char * getMD5() override { return "2db82b942157e1b0f14aa8b90cfdcd51"; };

  };

}
#endif
