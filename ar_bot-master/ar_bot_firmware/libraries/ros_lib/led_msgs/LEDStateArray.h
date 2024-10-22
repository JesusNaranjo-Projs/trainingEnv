#ifndef _ROS_led_msgs_LEDStateArray_h
#define _ROS_led_msgs_LEDStateArray_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "led_msgs/LEDState.h"

namespace led_msgs
{

  class LEDStateArray : public ros::Msg
  {
    public:
      uint32_t leds_length;
      typedef led_msgs::LEDState _leds_type;
      _leds_type st_leds;
      _leds_type * leds;

    LEDStateArray():
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

    virtual const char * getType() override { return "led_msgs/LEDStateArray"; };
    virtual const char * getMD5() override { return "c05c69b609770827fe456a05c8523fa4"; };

  };

}
#endif
