#include <common.h>

__kernel void argmax(OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM3
                     __read_only image2d_t input,
                     __private const int channels,
                     __private const int remain_channels,
                     __write_only image2d_t output) {

  const int w = get_global_id(1);
  const int h = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim1 || h >= global_size_dim2) {
    return;
  }
#endif

  const int channel_blocks = global_size_dim0 - 1;
  const int width = global_size_dim1;

  int idx = 0;
  float max = -1;

  for (int i = 0; i < channel_blocks; ++i) {
    const int pos = mad24(i, width, w);
    DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, h));

    const int i4 = (i * 4);
    if (in.x > max) {
      max = in.x;
      idx = i4;
    }
    if (in.y > max) {
      max = in.y;
      idx = i4 + 1;
    }
    if (in.z > max) {
      max = in.z;
      idx = i4 + 2;
    }
    if (in.w > max) {
      max = in.w;
      idx = i4 + 3;
    }
  }

  const int pos = mad24(channel_blocks - 1, width, w);
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, h));

  const int i4 = (channel_blocks * 4);

  switch(remain_channels) {
    case 0:
      if (in.w > max) {
        max = in.w;
        idx = i4 + 3;
      }
    case 1:
      if (in.z > max) {
        max = in.z;
        idx = i4 + 2;
      }
    case 2:
      if (in.y > max) {
        max = in.y;
        idx = i4 + 1;
      }
    case 3:
      if (in.x > max) {
        max = in.x;
        idx = i4;
      }
  }

  if (in.x > max) {
    max = in.x;
    idx = i4;
  }
  if (in.y > max) {
    max = in.y;
    idx = i4 + 1;
  }
  if (in.z > max) {
    max = in.z;
    idx = i4 + 2;
  }
  if (in.w > max) {
    max = in.w;
    idx = i4 + 3;
  }

  WRITE_IMAGET(output, (int2)(w, h), (float4)(idx, 0, 0, 0));
}
