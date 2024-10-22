#ifndef _ROS_kinova_study_LoadTrajectory_h
#define _ROS_kinova_study_LoadTrajectory_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "moveit_msgs/DisplayTrajectory.h"

namespace kinova_study
{

  class LoadTrajectory : public ros::Msg
  {
    public:
      typedef moveit_msgs::DisplayTrajectory _approach_type;
      _approach_type approach;
      typedef moveit_msgs::DisplayTrajectory _grasp_type;
      _grasp_type grasp;
      typedef moveit_msgs::DisplayTrajectory _handover_type;
      _handover_type handover;

    LoadTrajectory():
      approach(),
      grasp(),
      handover()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      offset += this->approach.serialize(outbuffer + offset);
      offset += this->grasp.serialize(outbuffer + offset);
      offset += this->handover.serialize(outbuffer + offset);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      offset += this->approach.deserialize(inbuffer + offset);
      offset += this->grasp.deserialize(inbuffer + offset);
      offset += this->handover.deserialize(inbuffer + offset);
     return offset;
    }

    virtual const char * getType() override { return "kinova_study/LoadTrajectory"; };
    virtual const char * getMD5() override { return "fe17c7fde0c19ddc5ab25b24b5eedebf"; };

  };

}
#endif
