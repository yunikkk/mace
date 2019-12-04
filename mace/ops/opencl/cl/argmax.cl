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
  float max = -FLT_MAX;

  for (int i = 0; i < channel_blocks; ++i) {
    const int pos = mad24(i, width, w);
    DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, h));

    const int i4 = (i * 4);
    idx = select(idx, i4, in.x > max);
    max = select(max, in.x, in.x > max);

    idx = select(idx, i4 + 1, in.y > max);
    max = select(max, in.y, in.y > max);

    idx = select(idx, i4 + 2, in.z > max);
    max = select(max, in.z, in.z > max);

    idx = select(idx, i4 + 3, in.w > max);
    max = select(max, in.w, in.w > max);
  }

  const int pos = mad24(channel_blocks, width, w);
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, h));

  const int i4 = (channel_blocks * 4);

  switch (remain_channels) {
    case 1:
      idx = select(idx, i4 + 2, in.z > max);
      max = select(max, in.z, in.z > max);
    case 2:
      idx = select(idx, i4 + 1, in.y > max);
      max = select(max, in.y, in.y > max);
    case 3:
      idx = select(idx, i4, in.x > max);
  }

  WRITE_IMAGET(output, (int2)(w, h), (float4)(idx, 0, 0, 0));
}
