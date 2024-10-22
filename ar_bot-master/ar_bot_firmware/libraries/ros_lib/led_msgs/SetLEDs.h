#ifndef _ROS_SERVICE_SetLEDs_h
#define _ROS_SERVICE_SetLEDs_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "led_msgs/LEDState.h"

namespace led_msgs
{

static const char SETLEDS[] = "led_msgs/SetLEDs";

  class SetLEDsRequest : public ros::Msg
  {
    public:
      uint32_t leds_length;
      typedef led_msgs::LEDState _leds_type;
      _leds_type st_leds;
      _leds_type * leds;

    SetLEDsRequest():
      leds_length(0), st_leds(), leds(nullptr)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->leds_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->leds_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->leds_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->leds_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->leds_length);
      for( uint32_t i = 0; i < leds_length; i++){
      offset += this->leds[i].serialize(outbuffer + offset);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      uint32_t leds_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      leds_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      leds_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      leds_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->leds_length);
      if(leds_lengthT > leds_length)
        this->leds = (led_msgs::LEDState*)realloc(this->leds, leds_lengthT * sizeof(led_msgs::LEDState));
      leds_length = leds_lengthT;
      for( uint32_t i = 0; i < leds_length; i++){
      offset += this->st_leds.deserialize(inbuffer + offset);
        memcpy( &(this->leds[i]), &(this->st_leds), sizeof(led_msgs::LEDState));
      }
     return offset;
    }

    virtual const char * getType() override { return SETLEDS; };
    virtual const char * getMD5() override { return "c05c69b609770827fe456a05c8523fa4"; };

  };

  class SetLEDsResponse : public ros::Msg
  {
    public:
      typedef bool _success_type;
      _success_type success;
      typedef const char* _message_type;
      _message_type message;

    SetLEDsResponse():
      success(0),
      message("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.real = this->success;
      *(outbuffer + offset + 0) = (u_success.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->success);
      uint32_t length_message = strlen(this->message);
      varToArr(outbuffer + offset, length_message);
      offset += 4;
      memcpy(outbuffer + offset, this->message, length_message);
      offset += length_message;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.base = 0;
      u_success.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->success = u_success.real;
      offset += sizeof(this->success);
      uint32_t length_message;
      arrToVar(length_message, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_message; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_message-1]=0;
      this->message = (char *)(inbuffer + offset-1);
      offset += length_message;
     return offset;
    }

    virtual const char * getType() override { return SETLEDS; };
    virtual const char * getMD5() override { return "937c9679a518e3a18d831e57125ea522"; };

  };

  class SetLEDs {
    public:
    typedef SetLEDsRequest Request;
    typedef SetLEDsResponse Response;
  };

}
#endif
