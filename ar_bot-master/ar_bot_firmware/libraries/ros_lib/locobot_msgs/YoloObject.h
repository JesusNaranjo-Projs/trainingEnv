#ifndef _ROS_SERVICE_YoloObject_h
#define _ROS_SERVICE_YoloObject_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "geometry_msgs/Point.h"

namespace locobot_msgs
{

static const char YOLOOBJECT[] = "locobot_msgs/YoloObject";

  class YoloObjectRequest : public ros::Msg
  {
    public:
      uint32_t objects_length;
      typedef char* _objects_type;
      _objects_type st_objects;
      _objects_type * objects;

    YoloObjectRequest():
      objects_length(0), st_objects(), objects(nullptr)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->objects_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->objects_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->objects_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->objects_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->objects_length);
      for( uint32_t i = 0; i < objects_length; i++){
      uint32_t length_objectsi = strlen(this->objects[i]);
      varToArr(outbuffer + offset, length_objectsi);
      offset += 4;
      memcpy(outbuffer + offset, this->objects[i], length_objectsi);
      offset += length_objectsi;
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      uint32_t objects_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      objects_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      objects_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      objects_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->objects_length);
      if(objects_lengthT > objects_length)
        this->objects = (char**)realloc(this->objects, objects_lengthT * sizeof(char*));
      objects_length = objects_lengthT;
      for( uint32_t i = 0; i < objects_length; i++){
      uint32_t length_st_objects;
      arrToVar(length_st_objects, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_st_objects; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_st_objects-1]=0;
      this->st_objects = (char *)(inbuffer + offset-1);
      offset += length_st_objects;
        memcpy( &(this->objects[i]), &(this->st_objects), sizeof(char*));
      }
     return offset;
    }

    virtual const char * getType() override { return YOLOOBJECT; };
    virtual const char * getMD5() override { return "39d6292ea712a13252ebdb5470ba0086"; };

  };

  class YoloObjectResponse : public ros::Msg
  {
    public:
      typedef geometry_msgs::Point _points_type;
      _points_type points;

    YoloObjectResponse():
      points()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      offset += this->points.serialize(outbuffer + offset);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      offset += this->points.deserialize(inbuffer + offset);
     return offset;
    }

    virtual const char * getType() override { return YOLOOBJECT; };
    virtual const char * getMD5() override { return "3fb3f9dacc279b964c4c8341122c34df"; };

  };

  class YoloObject {
    public:
    typedef YoloObjectRequest Request;
    typedef YoloObjectResponse Response;
  };

}
#endif
