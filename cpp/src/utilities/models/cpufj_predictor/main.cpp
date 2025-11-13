/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "header.h"

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif
#define N_TARGET    1
#define MAX_N_CLASS 1

const unsigned char is_categorical[] = {
  0,
  0,
  0,
  0,
};
static const int32_t num_class[] = {
  1,
};

int32_t cpufj_predictor::get_num_target(void) { return N_TARGET; }
void cpufj_predictor::get_num_class(int32_t* out)
{
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t cpufj_predictor::get_num_feature(void) { return 4; }
const char* cpufj_predictor::get_threshold_type(void) { return "float64"; }
const char* cpufj_predictor::get_leaf_output_type(void) { return "float64"; }

void cpufj_predictor::predict(union Entry* data, int pred_margin, double* result)
{
  // Quantize data
  for (int i = 0; i < 4; ++i) {
    if (data[i].missing != -1 && !is_categorical[i]) {
      data[i].qvalue = quantize(data[i].fvalue, i);
    }
  }

  unsigned int tmp;
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 20))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += 156.82697390964927;
        } else {
          result[0] += 160.51072839782955;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += 167.07152558194775;
        } else {
          result[0] += 173.16044577743776;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 197.67049116907555;
        } else {
          result[0] += 193.73228835720744;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 204.19401403625514;
        } else {
          result[0] += 233.21262399057034;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      result[0] += 271.7565681764123;
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 384.56067848347016;
      } else {
        result[0] += 490.5143183873164;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 50))) {
    if (LIKELY(false || (data[2].qvalue <= 14))) {
      if (LIKELY(false || (data[3].qvalue <= 114))) {
        if (LIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -13.943317103763608;
        } else {
          result[0] += -10.411451244915156;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -1.6607479643348968;
        } else {
          result[0] += 1.873995960655583;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 13.317445150475752;
        } else {
          result[0] += -3.2833804644580917;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 42))) {
          result[0] += 28.787436474976477;
        } else {
          result[0] += 22.808869505615306;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (UNLIKELY(false || (data[3].qvalue <= 188))) {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += 3.5889216009186664;
        } else {
          result[0] += 5.910052863589392;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 210))) {
          result[0] += 89.99770637090484;
        } else {
          result[0] += 127.79448643063864;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 191.2644805975275;
      } else {
        result[0] += 286.4874098858173;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 50))) {
    if (LIKELY(false || (data[2].qvalue <= 14))) {
      if (LIKELY(false || (data[3].qvalue <= 114))) {
        if (LIKELY(false || (data[3].qvalue <= 76))) {
          result[0] += -12.539408056743746;
        } else {
          result[0] += -9.311441710639189;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -1.4947050956018029;
        } else {
          result[0] += 1.6866257811476963;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 11.98745272164608;
        } else {
          result[0] += -2.9554856143381034;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 22.718528792836054;
        } else {
          result[0] += 52.594838808257;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (UNLIKELY(false || (data[3].qvalue <= 188))) {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += 3.2306536812875586;
        } else {
          result[0] += 5.321595177064682;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 79.5197036158342;
        } else {
          result[0] += 112.02363049281628;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 172.34821010044644;
      } else {
        result[0] += 257.92682935697115;
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 20))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -11.281112483305671;
        } else {
          result[0] += -8.601904513350851;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += -4.457196969778106;
        } else {
          result[0] += 0.6775720991155234;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 18.28399618068051;
        } else {
          result[0] += 15.34024990698243;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 23.56902428310345;
        } else {
          result[0] += 47.371626254377695;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      result[0] += 72.50774203412541;
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 155.30278921274038;
      } else {
        result[0] += 232.21351482872598;
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[3].qvalue <= 114))) {
        if (LIKELY(false || (data[3].qvalue <= 78))) {
          result[0] += -10.157360048782069;
        } else {
          result[0] += -7.469901314898597;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -1.2836068490987675;
        } else {
          result[0] += 1.4928852675361874;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 148))) {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 11.862787216911165;
        } else {
          result[0] += -3.825179380397125;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 36))) {
          result[0] += 16.467439176084763;
        } else {
          result[0] += 22.870842463159736;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 240))) {
      if (UNLIKELY(false || (data[3].qvalue <= 188))) {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += -4.329979207208883;
        } else {
          result[0] += -2.4277052596041226;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 62.899179793202485;
        } else {
          result[0] += 89.86758030525095;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 144.9081294228374;
      } else {
        result[0] += 209.06360369591349;
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[3].qvalue <= 118))) {
        if (LIKELY(false || (data[3].qvalue <= 80))) {
          result[0] += -9.111244309609868;
        } else {
          result[0] += -6.450300041406144;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 146))) {
          result[0] += -0.908464880592677;
        } else {
          result[0] += 1.3456266411166913;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 144))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 9.598829002806253;
        } else {
          result[0] += -3.458980194428218;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 36))) {
          result[0] += 14.685884008167873;
        } else {
          result[0] += 19.70403341262041;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[3].qvalue <= 188))) {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += -3.8977343521833427;
        } else {
          result[0] += -2.185981180004069;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 202))) {
          result[0] += 53.636361895007624;
        } else {
          result[0] += 73.84635747136831;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 54))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 109.30218868014822;
        } else {
          result[0] += 130.5223189467278;
        }
      } else {
        result[0] += 188.22157308443514;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 50))) {
    if (LIKELY(false || (data[2].qvalue <= 14))) {
      if (LIKELY(false || (data[3].qvalue <= 114))) {
        if (LIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -8.242985704258855;
        } else {
          result[0] += -6.103964758757681;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 122))) {
          result[0] += -3.0003459256849077;
        } else {
          result[0] += 0.5938833671659278;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 148))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 9.054639032941743;
        } else {
          result[0] += -3.3592089510831986;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 18.040143668653474;
        } else {
          result[0] += 13.175784904763427;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 236))) {
      if (UNLIKELY(false || (data[3].qvalue <= 188))) {
        if (LIKELY(false || (data[3].qvalue <= 118))) {
          result[0] += -3.214705374718561;
        } else {
          result[0] += 0.31848526393665993;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 202))) {
          result[0] += 48.27447091272866;
        } else {
          result[0] += 66.18473238316001;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 76))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 97.70898775652987;
        } else {
          result[0] += 117.56467429340749;
        }
      } else {
        result[0] += 169.45732965745194;
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 20))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -7.402034186828849;
        } else {
          result[0] += -5.63837490580214;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += -3.2673346602099738;
        } else {
          result[0] += 0.4720904756386771;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 11.693457400016031;
        } else {
          result[0] += 10.03852332044065;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 15.437715046858271;
        } else {
          result[0] += 36.647436411890496;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      result[0] += 47.53118689085973;
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 106.35664032988495;
      } else {
        result[0] += 152.56374281850964;
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 20))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -6.661840917407535;
        } else {
          result[0] += -5.074574846660258;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += -2.94088446825355;
        } else {
          result[0] += 0.4248859491848843;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 10.52521173349777;
        } else {
          result[0] += 9.034789845952849;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 13.894155617598422;
        } else {
          result[0] += 33.00796459750472;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (LIKELY(false || (data[0].qvalue <= 72))) {
        result[0] += 42.823961164678224;
      } else {
        result[0] += 36.98781896591187;
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 95.8378507576408;
      } else {
        result[0] += 137.35430146859977;
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 20))) {
        if (LIKELY(false || (data[1].qvalue <= 12))) {
          result[0] += -5.922535414763472;
        } else {
          result[0] += -3.9535302977307456;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += -2.647051125523947;
        } else {
          result[0] += 0.38240139128726525;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 9.473680911813272;
        } else {
          result[0] += 8.131417758482929;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 42))) {
          result[0] += 12.474218356244899;
        } else {
          result[0] += 28.32144924784816;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (LIKELY(false || (data[0].qvalue <= 72))) {
        result[0] += 38.542156344073405;
      } else {
        result[0] += 33.34683068752289;
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 86.359383424193;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 130.56753145370016;
        } else {
          result[0] += 115.99602850603911;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 20))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -5.403420533141926;
        } else {
          result[0] += -4.075259880702465;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += -2.3825756020703164;
        } else {
          result[0] += 0.34416491637505503;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 62))) {
          result[0] += 6.690473585247339;
        } else {
          result[0] += 7.675205616748645;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 11.251804458709806;
        } else {
          result[0] += 26.917319955760036;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (LIKELY(false || (data[0].qvalue <= 72))) {
        result[0] += 34.68847201632427;
      } else {
        result[0] += 30.064251933097843;
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 77.81834311684409;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 117.59088283047355;
        } else {
          result[0] += 104.46758514965971;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[3].qvalue <= 96))) {
        if (UNLIKELY(false || (data[3].qvalue <= 22))) {
          result[0] += -5.645610033293023;
        } else {
          result[0] += -4.712904234092419;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += -1.7467118855770272;
        } else {
          result[0] += -6.7806729780163355;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -3.3298901489085715;
        } else {
          result[0] += 0.9879303535654625;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 156))) {
          result[0] += 4.765656733422152;
        } else {
          result[0] += 10.436091866890687;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 188))) {
      if (LIKELY(false || (data[3].qvalue <= 118))) {
        result[0] += -19.231066348531115;
      } else {
        result[0] += -15.589280898150276;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (UNLIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 25.747728264474212;
        } else {
          result[0] += 40.04473103921778;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 64.38265010597043;
        } else {
          result[0] += 95.58272802311622;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[3].qvalue <= 98))) {
        if (LIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -4.409030071223973;
        } else {
          result[0] += -3.5647114194984613;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += -1.3542190966799321;
        } else {
          result[0] += -6.103673214709666;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.999404705305745;
        } else {
          result[0] += 0.8891527883879043;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 156))) {
          result[0] += 4.289215943564471;
        } else {
          result[0] += 9.392568858155158;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 188))) {
      if (LIKELY(false || (data[3].qvalue <= 116))) {
        result[0] += -17.598092511782436;
      } else {
        result[0] += -16.123547517184555;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 216))) {
        if (UNLIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 23.17394211472633;
        } else {
          result[0] += 35.171000449187375;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 55.995760587885144;
        } else {
          result[0] += 86.04514530858954;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 22))) {
        if (LIKELY(false || (data[1].qvalue <= 12))) {
          result[0] += -3.8888486100132043;
        } else {
          result[0] += -2.5693198969248536;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.301134908182464;
        } else {
          result[0] += 4.7321266664777495;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 5.997381523357489;
        } else {
          result[0] += 5.0375213284435185;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 8.559482811493893;
        } else {
          result[0] += 22.27496138368804;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (LIKELY(false || (data[0].qvalue <= 72))) {
        result[0] += 25.27353965981968;
      } else {
        result[0] += 21.13871028184891;
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 55.52418384593922;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 87.85257342192293;
        } else {
          result[0] += 76.03356082097153;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 50))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[3].qvalue <= 90))) {
        if (LIKELY(false || (data[0].qvalue <= 14))) {
          result[0] += -3.5064268267703387;
        } else {
          result[0] += -6.065722823494794;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += -1.4039415524212295;
        } else {
          result[0] += -5.997225846159178;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.9652186181689757;
        } else {
          result[0] += 0.7283126984643262;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 154))) {
          result[0] += 2.789907904600108;
        } else {
          result[0] += 7.718533347299168;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 188))) {
      result[0] += -18.00435797338746;
    } else {
      if (LIKELY(false || (data[3].qvalue <= 214))) {
        if (UNLIKELY(false || (data[3].qvalue <= 198))) {
          result[0] += 17.330223423965826;
        } else {
          result[0] += 27.46055877083792;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 46.12967429173014;
        } else {
          result[0] += 70.70123466855004;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[3].qvalue <= 102))) {
        if (UNLIKELY(false || (data[3].qvalue <= 22))) {
          result[0] += -3.888827166320002;
        } else {
          result[0] += -3.04723191570044;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += -0.7424769082893876;
        } else {
          result[0] += -5.398447803084307;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.670943097291571;
        } else {
          result[0] += 0.6554928174570391;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 158))) {
          result[0] += 2.9919951855699423;
        } else {
          result[0] += 7.128361704676317;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 188))) {
      if (LIKELY(false || (data[3].qvalue <= 118))) {
        result[0] += -16.31666198848907;
      } else {
        result[0] += -13.216809526331284;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 220))) {
        if (LIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 18.698365978047963;
        } else {
          result[0] += 30.323819222880072;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 54))) {
          result[0] += 48.19287170494928;
        } else {
          result[0] += 66.97776871619591;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 72))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[0].qvalue <= 22))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -2.8871386766483713;
        } else {
          result[0] += -2.194022438199927;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 24))) {
          result[0] += -1.2643512510325772;
        } else {
          result[0] += 0.2946143237550731;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 42))) {
        if (UNLIKELY(false || (data[0].qvalue <= 28))) {
          result[0] += 6.18334904875928;
        } else {
          result[0] += 3.4917401374748973;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 42))) {
          result[0] += 6.637851347005048;
        } else {
          result[0] += 17.628961355963423;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (LIKELY(false || (data[0].qvalue <= 72))) {
        result[0] += 18.427701194716725;
      } else {
        result[0] += 15.068101975917816;
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 76))) {
        result[0] += 39.624109077872816;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 65.43777984806364;
        } else {
          result[0] += 54.793414778446135;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[2].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 86))) {
          result[0] += -2.5929923519240776;
        } else {
          result[0] += -1.2308422313969718;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += -16.777735206670346;
        } else {
          result[0] += -15.496213790630472;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 166))) {
          result[0] += 3.153837529703604;
        } else {
          result[0] += 10.653343513120348;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 1.2707685237496547;
        } else {
          result[0] += -7.577582551412722;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[2].qvalue <= 50))) {
      if (LIKELY(false || (data[0].qvalue <= 70))) {
        if (UNLIKELY(false || (data[3].qvalue <= 216))) {
          result[0] += 5.086264511434565;
        } else {
          result[0] += 4.004129196876291;
        }
      } else {
        result[0] += 18.011869379087937;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 214))) {
        if (UNLIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 12.716813581606937;
        } else {
          result[0] += 21.19923662562386;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 33.0928592549443;
        } else {
          result[0] += 48.6196284504888;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[3].qvalue <= 104))) {
          result[0] += -2.2842871895780763;
        } else {
          result[0] += -0.6940580648135516;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 118))) {
          result[0] += -14.884714915484075;
        } else {
          result[0] += -12.21852856355555;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 1.1320691013866366;
        } else {
          result[0] += -6.824281661647207;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 168))) {
          result[0] += 3.133632898542928;
        } else {
          result[0] += 10.26787174244473;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[3].qvalue <= 216))) {
        if (UNLIKELY(false || (data[0].qvalue <= 66))) {
          result[0] += 0.3519373175810123;
        } else {
          result[0] += 4.883879225540344;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 234))) {
          result[0] += 3.4475335994592777;
        } else {
          result[0] += 4.289605817529637;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 214))) {
        if (UNLIKELY(false || (data[3].qvalue <= 198))) {
          result[0] += 10.944053551737898;
        } else {
          result[0] += 18.520104025075135;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 31.9020659327635;
        } else {
          result[0] += 47.722870097304835;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[3].qvalue <= 84))) {
          result[0] += -2.1126641262849875;
        } else {
          result[0] += -1.0034167734194253;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -14.666533134028597;
        } else {
          result[0] += -13.091642320227521;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.67357385169376;
        } else {
          result[0] += 0.317539563282027;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 172))) {
          result[0] += 2.9924052201762423;
        } else {
          result[0] += 10.018166909712392;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[3].qvalue <= 216))) {
        if (UNLIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.3173503774211839;
        } else {
          result[0] += 4.396111790242462;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 234))) {
          result[0] += 3.1028551632609607;
        } else {
          result[0] += 3.8610566164274576;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += 9.079411213991389;
        } else {
          result[0] += 15.631981353661537;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 230))) {
          result[0] += 24.953231107660457;
        } else {
          result[0] += 38.732870695055425;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[3].qvalue <= 106))) {
          result[0] += -1.8507802734637893;
        } else {
          result[0] += -0.4780342275095875;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += -12.281350727645211;
        } else {
          result[0] += -11.20561954498291;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.4082419443490855;
        } else {
          result[0] += 0.2857906276722831;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 164))) {
          result[0] += 1.8193510979340806;
        } else {
          result[0] += 7.12452591373636;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[3].qvalue <= 216))) {
        if (UNLIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.28616249149215633;
        } else {
          result[0] += 3.957059242184614;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 236))) {
          result[0] += 2.8268370028874497;
        } else {
          result[0] += 3.686134112733879;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (LIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 9.321451071849161;
        } else {
          result[0] += 15.02085895338778;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 24.772053316683984;
        } else {
          result[0] += 39.09600893416962;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 150))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[3].qvalue <= 124))) {
          result[0] += -1.6348218331553022;
        } else {
          result[0] += 0.16446761997202702;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -11.997166693345555;
        } else {
          result[0] += -10.593410837279578;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 176))) {
          result[0] += -0.26077036500572887;
        } else {
          result[0] += 0.26721227295908545;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 174))) {
          result[0] += 3.50927093938965;
        } else {
          result[0] += 9.399961201456877;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[3].qvalue <= 218))) {
        if (UNLIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.25803962657163887;
        } else {
          result[0] += 3.413014759107607;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 230))) {
          result[0] += 2.36220596981744;
        } else {
          result[0] += 2.9185929060991107;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (LIKELY(false || (data[3].qvalue <= 204))) {
          result[0] += 9.258923590762315;
        } else {
          result[0] += 15.922847318580807;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 25.53935408466994;
        } else {
          result[0] += 35.194870544491394;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 140))) {
      if (LIKELY(false || (data[2].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 82))) {
          result[0] += -1.5559766022197978;
        } else {
          result[0] += -0.6821089851736815;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 118))) {
          result[0] += -9.793275858100605;
        } else {
          result[0] += -7.646164855957032;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.185564344146035;
        } else {
          result[0] += 0.23136153315509006;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 164))) {
          result[0] += 1.4281790213369323;
        } else {
          result[0] += 5.862740580154604;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[2].qvalue <= 52))) {
      if (LIKELY(false || (data[0].qvalue <= 70))) {
        if (UNLIKELY(false || (data[3].qvalue <= 214))) {
          result[0] += 3.1620136193354766;
        } else {
          result[0] += 2.374787311783622;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 12.394186796439712;
        } else {
          result[0] += 7.221490024859086;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 222))) {
        if (LIKELY(false || (data[3].qvalue <= 202))) {
          result[0] += 7.724518171783787;
        } else {
          result[0] += 14.107049853741623;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 25.09318574808453;
        } else {
          result[0] += 33.43533238246624;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 150))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[3].qvalue <= 110))) {
          result[0] += -1.348448669905975;
        } else {
          result[0] += 0.3221871974704163;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 126))) {
          result[0] += -8.519520394077722;
        } else {
          result[0] += -4.128400709863002;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 16))) {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += 0.20971876754977306;
        } else {
          result[0] += 2.000764232879495;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 176))) {
          result[0] += 3.2250018408327294;
        } else {
          result[0] += 9.091428443343196;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[2].qvalue <= 46))) {
        result[0] += -0.07772423557166397;
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 7.465886690298717;
        } else {
          result[0] += 2.199134462061703;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += 5.510852046313868;
        } else {
          result[0] += 10.349949332911033;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 17.106146295672758;
        } else {
          result[0] += 26.416688466239478;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 152))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[3].qvalue <= 120))) {
          result[0] += -1.1991422347150693;
        } else {
          result[0] += 0.46237283922421873;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 126))) {
          result[0] += -7.6684254163131875;
        } else {
          result[0] += -3.7949608540857165;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 16))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += 0.18039368614868972;
        } else {
          result[0] += 1.85527472375075;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 172))) {
          result[0] += 2.72143013117147;
        } else {
          result[0] += 6.582942954251957;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[2].qvalue <= 52))) {
      if (LIKELY(false || (data[0].qvalue <= 70))) {
        if (UNLIKELY(false || (data[3].qvalue <= 214))) {
          result[0] += 2.622990881969633;
        } else {
          result[0] += 1.9174728788131057;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 10.617562740828639;
        } else {
          result[0] += 5.75643968641758;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 214))) {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += 4.573403401881785;
        } else {
          result[0] += 9.61596988143479;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 17.169701079810455;
        } else {
          result[0] += 25.958129802402993;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 152))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[3].qvalue <= 92))) {
          result[0] += -1.130718809808625;
        } else {
          result[0] += -0.2762627382286883;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -8.235156694808095;
        } else {
          result[0] += -6.951272825766222;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 32))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += 0.16235716809342288;
        } else {
          result[0] += 1.707373695884848;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 176))) {
          result[0] += 2.8296562924600828;
        } else {
          result[0] += 7.488739806649071;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[0].qvalue <= 66))) {
        result[0] += -0.32786249211636087;
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 6.490628203531107;
        } else {
          result[0] += 1.7818277591800613;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += 3.807200463755219;
        } else {
          result[0] += 8.111758332078471;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 226))) {
          result[0] += 13.144779116022248;
        } else {
          result[0] += 20.863007074513657;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 70))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 22))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -1.0274672289090832;
        } else {
          result[0] += -0.6823801552731604;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 52))) {
          result[0] += 1.0283586431978333;
        } else {
          result[0] += -0.23713978185064843;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[0].qvalue <= 62))) {
          result[0] += 2.1905285297360657;
        } else {
          result[0] += 6.504341335474002;
        }
      } else {
        result[0] += 0.9435717207184171;
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (UNLIKELY(false || (data[1].qvalue <= 74))) {
        if (UNLIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 3.5741883312100953;
        } else {
          result[0] += 4.626694408655167;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 74))) {
          result[0] += 6.38776066924514;
        } else {
          result[0] += 7.9433524366525505;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 27.524095430198624;
      } else {
        result[0] += 17.93763900587164;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 154))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[3].qvalue <= 108))) {
          result[0] += -0.8928303502649112;
        } else {
          result[0] += 0.1926267660463724;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 122))) {
          result[0] += -7.0074780337366285;
        } else {
          result[0] += -3.8229517707148526;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 16))) {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += 0.17887769899557107;
        } else {
          result[0] += 1.2261906364605684;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 178))) {
          result[0] += 2.7207480610255765;
        } else {
          result[0] += 8.3997445237607;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (LIKELY(false || (data[3].qvalue <= 236))) {
        if (UNLIKELY(false || (data[3].qvalue <= 218))) {
          result[0] += 1.928126817594764;
        } else {
          result[0] += 1.3411342581129906;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 1.9525385201789491;
        } else {
          result[0] += 2.4821401437493256;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (UNLIKELY(false || (data[3].qvalue <= 198))) {
          result[0] += 3.713535595855519;
        } else {
          result[0] += 7.180656615177431;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 226))) {
          result[0] += 11.193661980597666;
        } else {
          result[0] += 17.555807665127727;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 188))) {
    if (LIKELY(false || (data[3].qvalue <= 154))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += -0.7003375835348792;
        } else {
          result[0] += -3.7302757340894694;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += -6.599387818444294;
        } else {
          result[0] += -5.72489429399885;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 32))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += 0.15262275808585868;
        } else {
          result[0] += 1.1954652973697173;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 178))) {
          result[0] += 2.4487311442319593;
        } else {
          result[0] += 7.490679951544426;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 56))) {
      if (UNLIKELY(false || (data[3].qvalue <= 214))) {
        if (UNLIKELY(false || (data[0].qvalue <= 66))) {
          result[0] += -0.4618247323919986;
        } else {
          result[0] += 2.1528753353821717;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 234))) {
          result[0] += 1.2097848819662675;
        } else {
          result[0] += 1.752838457626999;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += 2.420871486643817;
        } else {
          result[0] += 6.029272785166614;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 230))) {
          result[0] += 10.227185956634905;
        } else {
          result[0] += 15.974606997518203;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 192))) {
    if (LIKELY(false || (data[3].qvalue <= 124))) {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += -2.865659975119095;
        } else {
          result[0] += -0.6896181889893488;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 116))) {
          result[0] += -5.944063382797502;
        } else {
          result[0] += -5.094536373408761;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 160))) {
          result[0] += 0.6397310298301389;
        } else {
          result[0] += 2.9220460179454832;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 138))) {
          result[0] += -2.666624663971065;
        } else {
          result[0] += 0.4270094967652458;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 72))) {
      if (LIKELY(false || (data[3].qvalue <= 236))) {
        if (UNLIKELY(false || (data[3].qvalue <= 218))) {
          result[0] += 1.521282712441143;
        } else {
          result[0] += 1.0810978861849587;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 1.582856585015188;
        } else {
          result[0] += 2.0611044482921446;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 216))) {
        if (LIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 4.439645479292941;
        } else {
          result[0] += 7.705468000634004;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 11.427783097865925;
        } else {
          result[0] += 17.19202195652448;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 192))) {
    if (LIKELY(false || (data[3].qvalue <= 124))) {
      if (LIKELY(false || (data[2].qvalue <= 28))) {
        if (UNLIKELY(false || (data[3].qvalue <= 24))) {
          result[0] += -1.2367334931972405;
        } else {
          result[0] += -0.568679959736126;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 112))) {
          result[0] += -8.32192050869182;
        } else {
          result[0] += -5.006960492312537;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 42))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.4550818668115695;
        } else {
          result[0] += -3.0743532402463507;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 160))) {
          result[0] += 0.13097623312980142;
        } else {
          result[0] += 2.786328266103951;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[1].qvalue <= 72))) {
          result[0] += 1.0925176197221544;
        } else {
          result[0] += 7.582882419296458;
        }
      } else {
        result[0] += -3.9018135684874005;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 224))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 4.15326719153954;
        } else {
          result[0] += 7.678964746257113;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 78))) {
          result[0] += 19.747446627865543;
        } else {
          result[0] += 12.500771902929218;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 192))) {
    if (LIKELY(false || (data[3].qvalue <= 124))) {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (UNLIKELY(false || (data[3].qvalue <= 22))) {
          result[0] += -1.167979990569673;
        } else {
          result[0] += -0.5133742356132379;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -5.603649494204901;
        } else {
          result[0] += -4.409295198837661;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 160))) {
          result[0] += 0.5480854924955038;
        } else {
          result[0] += 2.363109929409574;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 138))) {
          result[0] += -2.302846027254356;
        } else {
          result[0] += 0.34306161983695516;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 0.9832828581009277;
        } else {
          result[0] += 6.834192669120016;
        }
      } else {
        result[0] += -3.521149036128347;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 224))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 3.738054988080166;
        } else {
          result[0] += 6.911557381148448;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 78))) {
          result[0] += 17.78481683344929;
        } else {
          result[0] += 11.252482971264655;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 56))) {
    if (LIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 0))) {
        result[0] += -0.5726678321736987;
      } else {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += -0.14924717663033055;
        } else {
          result[0] += 6.235019242422922;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 1.162497192248416;
        } else {
          result[0] += 2.0120424899788536;
        }
      } else {
        result[0] += 0.47169244387428955;
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (UNLIKELY(false || (data[2].qvalue <= 42))) {
        if (UNLIKELY(false || (data[2].qvalue <= 34))) {
          result[0] += -0.357844108120637;
        } else {
          result[0] += 1.7605654400603017;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 70))) {
          result[0] += 6.52940587259161;
        } else {
          result[0] += 3.3080480937743113;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 16.01724653501452;
      } else {
        result[0] += 8.753001223254058;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 192))) {
    if (LIKELY(false || (data[3].qvalue <= 156))) {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += -0.4137669463081151;
        } else {
          result[0] += -2.6409307308959122;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -5.376507786444898;
        } else {
          result[0] += -4.389560249429967;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 32))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += 0.025008362302337356;
        } else {
          result[0] += 1.019241657971192;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 72))) {
          result[0] += 2.1411528979907484;
        } else {
          result[0] += -0.47484663873831007;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 0.8378079743141229;
        } else {
          result[0] += 5.832806958458091;
        }
      } else {
        result[0] += -3.500358439422236;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 222))) {
        if (LIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 2.851221265160206;
        } else {
          result[0] += 5.566558597933049;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 78))) {
          result[0] += 14.425347984875641;
        } else {
          result[0] += 9.578878534245574;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 192))) {
    if (LIKELY(false || (data[3].qvalue <= 124))) {
      if (LIKELY(false || (data[2].qvalue <= 28))) {
        if (UNLIKELY(false || (data[3].qvalue <= 22))) {
          result[0] += -0.9195523527649098;
        } else {
          result[0] += -0.3719639557405976;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -7.786786672274272;
        } else {
          result[0] += -3.9362138288079898;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 0.6022720589428014;
        } else {
          result[0] += 7.092300986713834;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 190))) {
          result[0] += -2.1043147125075716;
        } else {
          result[0] += 0.12674302444068922;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 50))) {
      if (LIKELY(false || (data[3].qvalue <= 232))) {
        if (UNLIKELY(false || (data[3].qvalue <= 218))) {
          result[0] += 1.031090335758177;
        } else {
          result[0] += 0.5392873356886984;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 0.9291700147586669;
        } else {
          result[0] += 1.5196304502959777;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 216))) {
        if (LIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 2.5829880193644055;
        } else {
          result[0] += 4.763908216949989;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 6.935099835666378;
        } else {
          result[0] += 10.036216690922197;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 164))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[3].qvalue <= 72))) {
        if (LIKELY(false || (data[1].qvalue <= 6))) {
          result[0] += -0.43590572608869044;
        } else {
          result[0] += -2.4316455018950878;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 42))) {
          result[0] += -0.04380141068835597;
        } else {
          result[0] += 1.7110383905553694;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 144))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += -3.530168794490836;
        } else {
          result[0] += -1.0214699319685518;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 148))) {
          result[0] += -6.775844377790179;
        } else {
          result[0] += -10.790508870091934;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 42))) {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (LIKELY(false || (data[3].qvalue <= 188))) {
          result[0] += -0.02275360025237618;
        } else {
          result[0] += -0.5838206132835355;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 1.7417272007672004;
        } else {
          result[0] += 0.660660463610099;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 208))) {
        if (LIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 1.9156509211867474;
        } else {
          result[0] += 3.135028625615758;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 4.960511180617512;
        } else {
          result[0] += 8.359091821368834;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 194))) {
    if (LIKELY(false || (data[3].qvalue <= 160))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[3].qvalue <= 70))) {
          result[0] += -0.42133291745401114;
        } else {
          result[0] += -0.010851476443648929;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -2.4223382968632965;
        } else {
          result[0] += -6.899222971006882;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 16))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += -0.03413012978096565;
        } else {
          result[0] += 0.5317297186286644;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 2.211563570567631;
        } else {
          result[0] += -0.16486923946003673;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 50))) {
      if (LIKELY(false || (data[3].qvalue <= 234))) {
        if (UNLIKELY(false || (data[3].qvalue <= 218))) {
          result[0] += 0.8620758848535985;
        } else {
          result[0] += 0.47621186848670294;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 238))) {
          result[0] += 0.8507697571985345;
        } else {
          result[0] += 1.3028689973708651;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 216))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 2.41099243454025;
        } else {
          result[0] += 4.090173047276771;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 5.632316419302413;
        } else {
          result[0] += 8.200667309946828;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 166))) {
    if (LIKELY(false || (data[1].qvalue <= 80))) {
      if (LIKELY(false || (data[3].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 14))) {
          result[0] += -0.38327232438457565;
        } else {
          result[0] += -2.1415671585396967;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += -0.048803059306061464;
        } else {
          result[0] += -1.7470427286358465;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 118))) {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -3.8643859740923037;
        } else {
          result[0] += -3.030297649090965;
        }
      } else {
        result[0] += -1.569212761065539;
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 42))) {
      if (UNLIKELY(false || (data[0].qvalue <= 54))) {
        if (UNLIKELY(false || (data[0].qvalue <= 52))) {
          result[0] += 1.1504605549094933;
        } else {
          result[0] += -0.02918860846727582;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 2.077977738321157;
        } else {
          result[0] += 0.5359649431721014;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (LIKELY(false || (data[3].qvalue <= 202))) {
          result[0] += 1.652849064266966;
        } else {
          result[0] += 2.954255485765737;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.5669657233750427;
        } else {
          result[0] += 5.676198660468206;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 168))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[2].qvalue <= 38))) {
        if (UNLIKELY(false || (data[3].qvalue <= 40))) {
          result[0] += -0.5244874333393534;
        } else {
          result[0] += -0.18256259456848412;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -5.51355459890058;
        } else {
          result[0] += 1.411684886661277;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 144))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += -2.6606175581636338;
        } else {
          result[0] += -0.47356034191920265;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 152))) {
          result[0] += -5.942664098495093;
        } else {
          result[0] += -10.448295749482655;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 38))) {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (LIKELY(false || (data[3].qvalue <= 188))) {
          result[0] += -0.011800565116617453;
        } else {
          result[0] += -0.5202224220078566;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 1.7775453851705354;
        } else {
          result[0] += 0.49081054656235107;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 222))) {
        if (LIKELY(false || (data[3].qvalue <= 204))) {
          result[0] += 1.614115694236453;
        } else {
          result[0] += 3.0704803932483227;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.5108923716568848;
        } else {
          result[0] += 6.589366885775977;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 42))) {
    if (LIKELY(false || (data[0].qvalue <= 30))) {
      if (LIKELY(false || (data[1].qvalue <= 0))) {
        result[0] += -0.2892059588012491;
      } else {
        if (LIKELY(false || (data[0].qvalue <= 14))) {
          result[0] += -0.08035600794549216;
        } else {
          result[0] += -0.4720559073325542;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.8967780238095124;
        } else {
          result[0] += 6.595205778394427;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.3520288227121683;
        } else {
          result[0] += 0.38974464605775183;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (UNLIKELY(false || (data[2].qvalue <= 38))) {
        if (LIKELY(false || (data[1].qvalue <= 64))) {
          result[0] += 0.7589099157010143;
        } else {
          result[0] += -1.366375525846731;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 66))) {
          result[0] += 4.22967240474964;
        } else {
          result[0] += 1.4943104257462279;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 9.129350705249179;
      } else {
        result[0] += 3.068747078527702;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 168))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[2].qvalue <= 42))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.18545114512070737;
        } else {
          result[0] += -1.7692822476814012;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -4.801651736039382;
        } else {
          result[0] += 1.3646239105274842;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += -2.5169209574656417;
        } else {
          result[0] += -0.18660289084267964;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 148))) {
          result[0] += -4.541809815605482;
        } else {
          result[0] += -8.652151740789414;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 1.3663672464257006;
        } else {
          result[0] += 3.04170718867983;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 0.2559705229975156;
        } else {
          result[0] += 2.9168273909132068;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 4.97017162342233;
        } else {
          result[0] += 11.503874847499691;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 0.4665647305038452;
        } else {
          result[0] += 5.509817603230935;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 162))) {
    if (LIKELY(false || (data[1].qvalue <= 80))) {
      if (UNLIKELY(false || (data[3].qvalue <= 2))) {
        if (LIKELY(false || (data[0].qvalue <= 58))) {
          result[0] += -1.3283728127725134;
        } else {
          result[0] += -6.130559274355571;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 62))) {
          result[0] += -0.2896825481817099;
        } else {
          result[0] += -0.07162487032383137;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 118))) {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -3.1167449558905838;
        } else {
          result[0] += -2.3599501347059975;
        }
      } else {
        result[0] += -1.059417066398789;
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 42))) {
      if (UNLIKELY(false || (data[0].qvalue <= 54))) {
        if (UNLIKELY(false || (data[0].qvalue <= 52))) {
          result[0] += 0.834224641552816;
        } else {
          result[0] += -0.015370587395285587;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 0.9254950158058017;
        } else {
          result[0] += 0.36421699256211726;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (LIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 1.0215455914435474;
        } else {
          result[0] += 1.9619948363613522;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.6542972311475775;
        } else {
          result[0] += 3.9645617071617623;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 42))) {
    if (LIKELY(false || (data[0].qvalue <= 22))) {
      if (LIKELY(false || (data[1].qvalue <= 14))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -0.22003805724566253;
        } else {
          result[0] += -0.05306048893366094;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 16))) {
          result[0] += -1.3451441339924273;
        } else {
          result[0] += -1.5488482760393083;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 28))) {
        if (UNLIKELY(false || (data[2].qvalue <= 10))) {
          result[0] += 0.21580833438485533;
        } else {
          result[0] += 1.177993229883637;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 56))) {
          result[0] += -0.2994307611075257;
        } else {
          result[0] += 0.305894106833898;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (UNLIKELY(false || (data[2].qvalue <= 34))) {
        if (LIKELY(false || (data[2].qvalue <= 32))) {
          result[0] += 0.6300423600302785;
        } else {
          result[0] += -0.4677024667015657;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 60))) {
          result[0] += 2.92097058914907;
        } else {
          result[0] += 0.9749046067206072;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 7.280391073314689;
      } else {
        result[0] += 1.8221294908582069;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 170))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[2].qvalue <= 42))) {
        if (UNLIKELY(false || (data[3].qvalue <= 20))) {
          result[0] += -0.5336229335357556;
        } else {
          result[0] += -0.12326033481423136;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 120))) {
          result[0] += -3.7782573749400954;
        } else {
          result[0] += 1.2031945074524026;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[3].qvalue <= 136))) {
          result[0] += -2.0138202324018293;
        } else {
          result[0] += 0.4287253040269907;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 150))) {
          result[0] += -4.26031586774496;
        } else {
          result[0] += -8.189678469057437;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 36))) {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (LIKELY(false || (data[3].qvalue <= 188))) {
          result[0] += 0.03067279449262314;
        } else {
          result[0] += -0.4287195475348111;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 1.8796244512104765;
        } else {
          result[0] += 0.3056819085450995;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 226))) {
        if (LIKELY(false || (data[3].qvalue <= 202))) {
          result[0] += 0.9636725612848687;
        } else {
          result[0] += 1.932067083622997;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.6860055204538198;
        } else {
          result[0] += 4.570361189431599;
        }
      }
    }
  }
  if (LIKELY(false || (data[1].qvalue <= 42))) {
    if (LIKELY(false || (data[0].qvalue <= 22))) {
      if (LIKELY(false || (data[1].qvalue <= 14))) {
        if (LIKELY(false || (data[1].qvalue <= 0))) {
          result[0] += -0.1814554734769651;
        } else {
          result[0] += -0.03493688255810839;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 16))) {
          result[0] += -1.1580987454621139;
        } else {
          result[0] += -1.3413217625073497;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.6065784994119716;
        } else {
          result[0] += 6.64156315122332;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 56))) {
          result[0] += -0.2710747772797691;
        } else {
          result[0] += 0.24590059195978556;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (UNLIKELY(false || (data[2].qvalue <= 38))) {
        if (LIKELY(false || (data[1].qvalue <= 64))) {
          result[0] += 0.52839377206309;
        } else {
          result[0] += -1.3097734584676666;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 66))) {
          result[0] += 3.0327157128432702;
        } else {
          result[0] += 0.8301613199220086;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 6.102585921419179;
      } else {
        result[0] += 1.1868023468270625;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 170))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[2].qvalue <= 42))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.1099508760937471;
        } else {
          result[0] += -1.4799074079672025;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 74))) {
          result[0] += -3.9533722562056326;
        } else {
          result[0] += 1.0527497525643543;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[3].qvalue <= 136))) {
          result[0] += -1.874856650108779;
        } else {
          result[0] += 0.36151445149430145;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 152))) {
          result[0] += -4.013347158834968;
        } else {
          result[0] += -7.940386810302735;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 206))) {
          result[0] += 0.8194996574746792;
        } else {
          result[0] += 1.9073466437063578;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 0.1914437036803978;
        } else {
          result[0] += 1.9462096802861646;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 3.445255372400778;
        } else {
          result[0] += 9.341778161720354;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 0.3243846697838024;
        } else {
          result[0] += 3.292591407943689;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 160))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (UNLIKELY(false || (data[3].qvalue <= 2))) {
        if (LIKELY(false || (data[0].qvalue <= 58))) {
          result[0] += -0.9774807565883121;
        } else {
          result[0] += -5.423777811527253;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += -0.12073383773519127;
        } else {
          result[0] += 0.9911425235059078;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += -1.8435700376685866;
        } else {
          result[0] += -0.10737506746612407;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 148))) {
          result[0] += -3.3288382873336477;
        } else {
          result[0] += -5.702344606187609;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 202))) {
          result[0] += 0.6121670947641649;
        } else {
          result[0] += 1.5478825951517399;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 0.15159187149918565;
        } else {
          result[0] += 1.7525244521134302;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 3.1021024062818072;
        } else {
          result[0] += 8.417132546913868;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 0.29207001641087005;
        } else {
          result[0] += 2.9643452592336215;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 20))) {
    if (LIKELY(false || (data[1].qvalue <= 0))) {
      result[0] += -0.14003554272994065;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 46))) {
          result[0] += -0.015514364763912126;
        } else {
          result[0] += 1.0510222973719732;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.27117390740511166;
        } else {
          result[0] += 0.14354185812943476;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 62))) {
          result[0] += 0.5018898936445311;
        } else {
          result[0] += 2.5336810498326523;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 2.3642130662022995;
        } else {
          result[0] += 0.07374194009502717;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 4.874216369284443;
      } else {
        result[0] += 0.44699526391786304;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 170))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (UNLIKELY(false || (data[3].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 58))) {
          result[0] += -0.35060793036231724;
        } else {
          result[0] += -5.047665471633276;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 38))) {
          result[0] += -0.07028608501406199;
        } else {
          result[0] += 0.7109949202510606;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 144))) {
        if (LIKELY(false || (data[3].qvalue <= 122))) {
          result[0] += -1.8132749471478289;
        } else {
          result[0] += -0.4798894299710876;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 152))) {
          result[0] += -3.6392164039153325;
        } else {
          result[0] += -6.78748151870001;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[0].qvalue <= 32))) {
          result[0] += 0.6818792886595983;
        } else {
          result[0] += 2.0346700524088903;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 70))) {
          result[0] += 0.1650370627648028;
        } else {
          result[0] += 1.4122047675458287;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 2.743139011175984;
        } else {
          result[0] += 7.534331532576863;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 0.25562844944051427;
        } else {
          result[0] += 2.4035809405106767;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 20))) {
    if (LIKELY(false || (data[1].qvalue <= 0))) {
      result[0] += -0.1149636651708558;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 46))) {
          result[0] += -0.005786597801570078;
        } else {
          result[0] += 0.9536320919638706;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.25774071878779176;
        } else {
          result[0] += 0.12494917135148927;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 0.416251413237658;
        } else {
          result[0] += 4.65468264502448;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 1.9891757155656817;
        } else {
          result[0] += 0.06062405904969436;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 4.150901784204815;
      } else {
        result[0] += 0.1636860781066988;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 20))) {
    if (LIKELY(false || (data[1].qvalue <= 0))) {
      result[0] += -0.10346745600044038;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 46))) {
          result[0] += -0.005207970482558494;
        } else {
          result[0] += 0.8583114141430566;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.2319702347159344;
        } else {
          result[0] += 0.11246252282811312;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 62))) {
          result[0] += 0.36237019282195543;
        } else {
          result[0] += 2.185361301747475;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 1.7916299525787094;
        } else {
          result[0] += 0.05456246786985419;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 3.7383578847668653;
      } else {
        result[0] += 0.1474179036193099;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 20))) {
    if (LIKELY(false || (data[0].qvalue <= 22))) {
      if (LIKELY(false || (data[1].qvalue <= 14))) {
        if (LIKELY(false || (data[0].qvalue <= 16))) {
          result[0] += -0.07824623744678852;
        } else {
          result[0] += 2.4147477472795025;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 20))) {
          result[0] += -0.9602189832977626;
        } else {
          result[0] += -1.2336897440823642;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[2].qvalue <= 10))) {
          result[0] += 0.17808383312156362;
        } else {
          result[0] += 0.7764366538076664;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.20877644063379633;
        } else {
          result[0] += 0.10122370049456518;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 0.3366817124844217;
        } else {
          result[0] += 3.9891647478052095;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 1.613702605033743;
        } else {
          result[0] += 0.04910696472777197;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 3.3668153467865807;
      } else {
        result[0] += 0.13276691204199761;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 170))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (UNLIKELY(false || (data[3].qvalue <= 2))) {
        if (LIKELY(false || (data[2].qvalue <= 28))) {
          result[0] += -0.8003296552631252;
        } else {
          result[0] += -5.168554219404857;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 28))) {
          result[0] += -0.04191612245860841;
        } else {
          result[0] += -0.31322033502373364;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 142))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += -1.6260716001993314;
        } else {
          result[0] += -0.07253028186242319;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 152))) {
          result[0] += -3.059761641596405;
        } else {
          result[0] += -6.156701689220611;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[3].qvalue <= 176))) {
          result[0] += 0.8382331706298907;
        } else {
          result[0] += 0.1470265483651418;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 68))) {
          result[0] += 1.4136662728343927;
        } else {
          result[0] += 0.42639604831684264;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (UNLIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += 0.2137965369719358;
        } else {
          result[0] += 1.5807652149640596;
        }
      } else {
        result[0] += 3.582011053264141;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 20))) {
    if (LIKELY(false || (data[1].qvalue <= 0))) {
      result[0] += -0.0809220032676647;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 46))) {
          result[0] += 0.009295531913092564;
        } else {
          result[0] += 0.7015704698474159;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          result[0] += -0.18839625520645842;
        } else {
          result[0] += 0.13815172995417008;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 0.2833961104028503;
        } else {
          result[0] += 3.9494598141232053;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 1.3130535747634955;
        } else {
          result[0] += 0.04139692407062909;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 2.875092684213369;
      } else {
        result[0] += -0.03741247869342383;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 198))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[3].qvalue <= 170))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += -0.9280781442850904;
        } else {
          result[0] += -0.05519963885812408;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 18))) {
          result[0] += 0.13575882302728218;
        } else {
          result[0] += 1.3530812716367793;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 192))) {
        if (LIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -1.0690103616905782;
        } else {
          result[0] += -2.214087039734459;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.6739287079004651;
        } else {
          result[0] += 0.08192197004299273;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[1].qvalue <= 72))) {
          result[0] += 0.16779935065337956;
        } else {
          result[0] += 2.7936280922346484;
        }
      } else {
        result[0] += -6.01828142584824;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 216))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.6464965410008258;
        } else {
          result[0] += 1.2127821355938402;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 56))) {
          result[0] += 2.304393523269968;
        } else {
          result[0] += -0.03369406329707866;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 14))) {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[0].qvalue <= 44))) {
        if (LIKELY(false || (data[2].qvalue <= 6))) {
          result[0] += -0.053129047160861614;
        } else {
          result[0] += -0.6797773812157768;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += 0.4863998644119598;
        } else {
          result[0] += 3.517865025900506;
        }
      }
    } else {
      result[0] += -0.18996639918014335;
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (UNLIKELY(false || (data[0].qvalue <= 36))) {
        if (LIKELY(false || (data[0].qvalue <= 34))) {
          result[0] += 0.21126964469127507;
        } else {
          result[0] += 0.9292041399773837;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.38898047466443736;
        } else {
          result[0] += 0.0343316630215836;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 78))) {
        result[0] += 2.3603220532788822;
      } else {
        result[0] += -0.030345338012543195;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 198))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[3].qvalue <= 170))) {
        if (LIKELY(false || (data[2].qvalue <= 12))) {
          result[0] += -0.03196203652937306;
        } else {
          result[0] += -0.32291491116843224;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 14))) {
          result[0] += 0.1059633693894297;
        } else {
          result[0] += 1.1109217701835943;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 192))) {
        if (LIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -0.9750124564919093;
        } else {
          result[0] += -2.011326445104133;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.6277616333681295;
        } else {
          result[0] += 0.052653140828700706;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[1].qvalue <= 72))) {
          result[0] += 0.14758938512729353;
        } else {
          result[0] += 2.5144118215047015;
        }
      } else {
        result[0] += -5.434481533329662;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.5607584407428231;
        } else {
          result[0] += 1.097345812808785;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 56))) {
          result[0] += 2.188365555677249;
        } else {
          result[0] += -0.027329945962853226;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 196))) {
    if (LIKELY(false || (data[1].qvalue <= 72))) {
      if (LIKELY(false || (data[3].qvalue <= 174))) {
        if (UNLIKELY(false || (data[3].qvalue <= 16))) {
          result[0] += -0.339980565686603;
        } else {
          result[0] += -0.03185759232203992;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 42))) {
          result[0] += 0.1301433477915698;
        } else {
          result[0] += 1.580000992364498;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 192))) {
        if (LIKELY(false || (data[3].qvalue <= 190))) {
          result[0] += -1.685654765519544;
        } else {
          result[0] += -1.0927866830676793;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.5651196538004228;
        } else {
          result[0] += -0.10798605171852133;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[0].qvalue <= 74))) {
          result[0] += 0.13680401406260853;
        } else {
          result[0] += 2.8238424551720716;
        }
      } else {
        result[0] += -4.904288046301866;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 226))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.45463644431627187;
        } else {
          result[0] += 1.0130458030200111;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 2.236542576950049;
        } else {
          result[0] += -0.024613159223933895;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 198))) {
    if (LIKELY(false || (data[1].qvalue <= 72))) {
      if (LIKELY(false || (data[3].qvalue <= 172))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.032568909202472296;
        } else {
          result[0] += -0.4469851314428541;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += 0.11294361883379918;
        } else {
          result[0] += 1.0056842141087612;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 192))) {
        if (LIKELY(false || (data[3].qvalue <= 190))) {
          result[0] += -1.517244761721867;
        } else {
          result[0] += -0.9837414394446418;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.508728386740654;
        } else {
          result[0] += 0.03608265112965852;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 0.11915447057376925;
        } else {
          result[0] += 2.0825805265428143;
        }
      } else {
        result[0] += -4.425821080207825;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 226))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.4592645630622676;
        } else {
          result[0] += 0.9118048684410376;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 2.0132186895684288;
        } else {
          result[0] += -0.022167118571104448;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 198))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[3].qvalue <= 160))) {
        if (LIKELY(false || (data[2].qvalue <= 12))) {
          result[0] += -0.020052189264742216;
        } else {
          result[0] += -0.4390079826397957;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 18))) {
          result[0] += 0.06723042230135214;
        } else {
          result[0] += 0.6321455651905148;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 192))) {
        if (LIKELY(false || (data[3].qvalue <= 142))) {
          result[0] += -0.6849792752988858;
        } else {
          result[0] += -1.5843785915044357;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.45796421690851963;
        } else {
          result[0] += 0.026405314510040745;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 50))) {
      if (LIKELY(false || (data[3].qvalue <= 230))) {
        if (UNLIKELY(false || (data[3].qvalue <= 218))) {
          result[0] += 0.3385478130008445;
        } else {
          result[0] += -0.10764744508292204;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 236))) {
          result[0] += 0.20488484351232758;
        } else {
          result[0] += 0.47811499378647015;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 212))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.4209993070324332;
        } else {
          result[0] += 0.7287778763739126;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 1.187064839378982;
        } else {
          result[0] += -0.01996407508027334;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 176))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[0].qvalue <= 60))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.027774129306039258;
        } else {
          result[0] += -0.8385662967485399;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 120))) {
          result[0] += -3.519087969462077;
        } else {
          result[0] += 0.7293787751512967;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 148))) {
        if (LIKELY(false || (data[3].qvalue <= 118))) {
          result[0] += -1.1172824090308644;
        } else {
          result[0] += -0.20902441584933884;
        }
      } else {
        result[0] += -5.111647744634573;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[3].qvalue <= 186))) {
        if (UNLIKELY(false || (data[0].qvalue <= 52))) {
          result[0] += 1.8609278771357982;
        } else {
          result[0] += 0.12480108283901199;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += -0.08227578925901015;
        } else {
          result[0] += 0.25625433402161485;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.9637861349021059;
        } else {
          result[0] += 5.291859932724311;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.9927995753498028;
        } else {
          result[0] += 0.14113094589752082;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 160))) {
    if (LIKELY(false || (data[2].qvalue <= 12))) {
      if (LIKELY(false || (data[3].qvalue <= 130))) {
        if (UNLIKELY(false || (data[3].qvalue <= 46))) {
          result[0] += -0.13887715472257542;
        } else {
          result[0] += 0.01749027468530986;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += 0.8747182175833713;
        } else {
          result[0] += 3.3873552828056868;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (LIKELY(false || (data[3].qvalue <= 140))) {
          result[0] += -2.0719160557169487;
        } else {
          result[0] += -0.9527323590531762;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 112))) {
          result[0] += -4.278540461588714;
        } else {
          result[0] += -0.22532110219787005;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[0].qvalue <= 52))) {
        if (LIKELY(false || (data[3].qvalue <= 204))) {
          result[0] += 0.13994283376545083;
        } else {
          result[0] += 0.628766931454757;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 168))) {
          result[0] += -0.2077112285433431;
        } else {
          result[0] += 0.0971080345193495;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.8677912951323141;
        } else {
          result[0] += 4.768073993215755;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.8939668307876266;
        } else {
          result[0] += 0.1270565048333717;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 160))) {
    if (LIKELY(false || (data[2].qvalue <= 12))) {
      if (LIKELY(false || (data[3].qvalue <= 130))) {
        if (UNLIKELY(false || (data[3].qvalue <= 36))) {
          result[0] += -0.166060933557046;
        } else {
          result[0] += 0.004294648853588787;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += 0.7873344634071392;
        } else {
          result[0] += 3.0512455477455793;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        if (UNLIKELY(false || (data[1].qvalue <= 16))) {
          result[0] += 3.321445830033885;
        } else {
          result[0] += -1.8410162796558962;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.5648229020291158;
        } else {
          result[0] += -0.40508307941792243;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (LIKELY(false || (data[1].qvalue <= 38))) {
        if (LIKELY(false || (data[3].qvalue <= 218))) {
          result[0] += 0.116824997205243;
        } else {
          result[0] += 0.0017106091707403482;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 200))) {
          result[0] += 0.09749333921964695;
        } else {
          result[0] += 0.512061380766423;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (UNLIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.8049728208270159;
        } else {
          result[0] += 0.11438560255422983;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.7813580416089511;
        } else {
          result[0] += 4.296131911715682;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 160))) {
    if (LIKELY(false || (data[2].qvalue <= 12))) {
      if (LIKELY(false || (data[3].qvalue <= 132))) {
        if (UNLIKELY(false || (data[3].qvalue <= 50))) {
          result[0] += -0.10165509214748221;
        } else {
          result[0] += 0.027573987237460126;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += 0.8965004316498251;
        } else {
          result[0] += 2.7484863860847417;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (LIKELY(false || (data[3].qvalue <= 140))) {
          result[0] += -1.6997008955624044;
        } else {
          result[0] += -0.678268726322754;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 118))) {
          result[0] += -1.114816829956962;
        } else {
          result[0] += -0.07990083268080249;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (UNLIKELY(false || (data[3].qvalue <= 186))) {
        if (LIKELY(false || (data[3].qvalue <= 178))) {
          result[0] += 0.08501821194709441;
        } else {
          result[0] += 0.7073547907959752;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += -0.09598099681424313;
        } else {
          result[0] += 0.1882562171309855;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.7035334691233133;
        } else {
          result[0] += 3.8709025872240264;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += -0.03477044912415934;
        } else {
          result[0] += 0.6388646830943916;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 4))) {
    if (LIKELY(false || (data[0].qvalue <= 14))) {
      if (LIKELY(false || (data[1].qvalue <= 0))) {
        result[0] += -0.034231165787913284;
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 0))) {
          result[0] += -0.008538500447320395;
        } else {
          result[0] += 0.030661489227495748;
        }
      }
    } else {
      result[0] += -0.6560679714832831;
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[2].qvalue <= 46))) {
        if (LIKELY(false || (data[0].qvalue <= 46))) {
          result[0] += 0.05367898474501495;
        } else {
          result[0] += 0.44735028827783085;
        }
      } else {
        result[0] += 6.229511664254325;
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        result[0] += -0.14752365398422523;
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.38398941999281766;
        } else {
          result[0] += 0.04404838897933778;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 176))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (UNLIKELY(false || (data[3].qvalue <= 0))) {
        if (LIKELY(false || (data[2].qvalue <= 28))) {
          result[0] += -0.7549177495924217;
        } else {
          result[0] += -4.158142198324204;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 42))) {
          result[0] += -0.02530038694314765;
        } else {
          result[0] += 0.5546607631795957;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 148))) {
        if (LIKELY(false || (data[3].qvalue <= 136))) {
          result[0] += -0.6813185032064286;
        } else {
          result[0] += 0.29119811621696806;
        }
      } else {
        result[0] += -4.5671472818360614;
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (LIKELY(false || (data[1].qvalue <= 40))) {
        if (LIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.06089688550626267;
        } else {
          result[0] += -0.9885366824653841;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 180))) {
          result[0] += 0.8258845435182254;
        } else {
          result[0] += 5.761587441940307;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 238))) {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += -0.07740265934103074;
        } else {
          result[0] += 0.16502792587020448;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 80))) {
          result[0] += 0.3005847261842887;
        } else {
          result[0] += 1.4347149255319134;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 152))) {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[3].qvalue <= 134))) {
        if (LIKELY(false || (data[0].qvalue <= 58))) {
          result[0] += -0.010806768205029583;
        } else {
          result[0] += -3.3904279910193553;
        }
      } else {
        result[0] += 3.6360500988061872;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 56))) {
        if (UNLIKELY(false || (data[1].qvalue <= 48))) {
          result[0] += -1.3970841696833907;
        } else {
          result[0] += -0.5310959589164045;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 136))) {
          result[0] += -0.43780724551513983;
        } else {
          result[0] += 0.38160829548283537;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 238))) {
      if (LIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[0].qvalue <= 58))) {
          result[0] += 0.1260601616574519;
        } else {
          result[0] += 2.3280856465403716;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 188))) {
          result[0] += -6.748794538567707;
        } else {
          result[0] += 0.03388208818968886;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.4861732172767124;
        } else {
          result[0] += 3.34139486286105;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.6939914549665678;
        } else {
          result[0] += 0.012280274946395664;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 176))) {
    if (LIKELY(false || (data[1].qvalue <= 54))) {
      if (UNLIKELY(false || (data[3].qvalue <= 2))) {
        if (LIKELY(false || (data[1].qvalue <= 20))) {
          result[0] += -0.464813733628163;
        } else {
          result[0] += -3.2180759784840705;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 48))) {
          result[0] += -0.07920781023396078;
        } else {
          result[0] += 0.019886892967067187;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 146))) {
        if (UNLIKELY(false || (data[1].qvalue <= 60))) {
          result[0] += -1.3944523714625319;
        } else {
          result[0] += -0.6633490458105974;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 58))) {
          result[0] += -0.4361867431137107;
        } else {
          result[0] += 0.09866640502908253;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (LIKELY(false || (data[1].qvalue <= 40))) {
        if (LIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.04221310244085489;
        } else {
          result[0] += -0.8607172810069976;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 180))) {
          result[0] += 0.7308448858222341;
        } else {
          result[0] += 5.17753285443306;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 198))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.12604388146867565;
        } else {
          result[0] += -0.5198739781442024;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += 0.0001483644827354292;
        } else {
          result[0] += 0.4382489570445441;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 12))) {
    if (LIKELY(false || (data[0].qvalue <= 58))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.2715686357969588;
        } else {
          result[0] += -0.17382652834003876;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.5295756487690885;
        } else {
          result[0] += -1.247911300102006;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 64))) {
        result[0] += -4.132038760807204;
      } else {
        result[0] += -2.048248645525712;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 236))) {
      if (LIKELY(false || (data[3].qvalue <= 152))) {
        if (LIKELY(false || (data[2].qvalue <= 12))) {
          result[0] += 0.009045345990511855;
        } else {
          result[0] += -0.36819514234182105;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.15071853324608264;
        } else {
          result[0] += 0.0023504781977204342;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.4615429381822286;
        } else {
          result[0] += 2.967287079375618;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.39428092203572945;
        } else {
          result[0] += -0.027793250841637182;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 4))) {
    if (LIKELY(false || (data[0].qvalue <= 14))) {
      if (LIKELY(false || (data[1].qvalue <= 2))) {
        if (UNLIKELY(false || (data[0].qvalue <= 0))) {
          result[0] += 0.02095625550210984;
        } else {
          result[0] += -0.02470681499709372;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 0))) {
          result[0] += -0.006936135990179306;
        } else {
          result[0] += 0.028313332943848205;
        }
      }
    } else {
      result[0] += -0.45814704971029124;
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[2].qvalue <= 46))) {
        if (LIKELY(false || (data[0].qvalue <= 46))) {
          result[0] += 0.059813599712526144;
        } else {
          result[0] += 0.4052626695215156;
        }
      } else {
        result[0] += 6.220781436647688;
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        result[0] += -0.14287609636708531;
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.28909888374082054;
        } else {
          result[0] += 0.0223884762447394;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 178))) {
    if (LIKELY(false || (data[1].qvalue <= 52))) {
      if (UNLIKELY(false || (data[3].qvalue <= 10))) {
        if (LIKELY(false || (data[1].qvalue <= 20))) {
          result[0] += -0.20717493847839827;
        } else {
          result[0] += -2.852040849850889;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 30))) {
          result[0] += -0.008160190303522096;
        } else {
          result[0] += 0.27599937314740675;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 146))) {
        if (UNLIKELY(false || (data[1].qvalue <= 60))) {
          result[0] += -1.1881260380224656;
        } else {
          result[0] += -0.5625438965153544;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 58))) {
          result[0] += -0.403338839639409;
        } else {
          result[0] += 0.10268439573411683;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 184))) {
      if (LIKELY(false || (data[1].qvalue <= 40))) {
        if (LIKELY(false || (data[3].qvalue <= 182))) {
          result[0] += -0.013368157221764328;
        } else {
          result[0] += 0.11043304821096991;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 180))) {
          result[0] += 1.5817916361431956;
        } else {
          result[0] += 4.191132561870824;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 236))) {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += -0.05418531103810077;
        } else {
          result[0] += 0.11582636097487908;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 80))) {
          result[0] += 0.21679116682674598;
        } else {
          result[0] += 0.9937182303717;
        }
      }
    }
  }
  if (LIKELY(false || (data[0].qvalue <= 76))) {
    if (LIKELY(false || (data[0].qvalue <= 74))) {
      if (LIKELY(false || (data[1].qvalue <= 0))) {
        result[0] += -0.020289773669981;
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 28))) {
          result[0] += 0.07888539366069491;
        } else {
          result[0] += -0.0043942118674004035;
        }
      }
    } else {
      result[0] += -0.8819353631826549;
    }
  } else {
    if (UNLIKELY(false || (data[0].qvalue <= 78))) {
      result[0] += 1.166342015953883;
    } else {
      result[0] += -0.21953414862872633;
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 178))) {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[3].qvalue <= 134))) {
        if (LIKELY(false || (data[3].qvalue <= 128))) {
          result[0] += -0.014985389955920162;
        } else {
          result[0] += 0.5030502722795092;
        }
      } else {
        result[0] += 3.209527899147808;
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        if (LIKELY(false || (data[3].qvalue <= 146))) {
          result[0] += -1.359497708335802;
        } else {
          result[0] += -0.048409546375607654;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.2462483312344813;
        } else {
          result[0] += -0.15689708895777443;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (LIKELY(false || (data[1].qvalue <= 40))) {
        if (UNLIKELY(false || (data[3].qvalue <= 182))) {
          result[0] += -0.011608086477160455;
        } else {
          result[0] += 0.054071549124002416;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 180))) {
          result[0] += 1.4247202960054501;
        } else {
          result[0] += 4.292557497723102;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 196))) {
        if (LIKELY(false || (data[1].qvalue <= 72))) {
          result[0] += 0.11744769877683264;
        } else {
          result[0] += -0.6228392828110428;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.01810292778067102;
        } else {
          result[0] += 0.33713691230011644;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 32))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.271129972670566;
        } else {
          result[0] += -0.0968952898723599;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.4419965803483365;
        } else {
          result[0] += -1.1427374036977256;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 64))) {
        result[0] += -2.964990935176611;
      } else {
        result[0] += -2.0024242342435397;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 236))) {
      if (LIKELY(false || (data[3].qvalue <= 152))) {
        if (LIKELY(false || (data[1].qvalue <= 28))) {
          result[0] += 0.019913351953155762;
        } else {
          result[0] += -0.3451557406567103;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 0.09431825657210167;
        } else {
          result[0] += -0.040080507458776934;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.2811388814670214;
        } else {
          result[0] += 2.54007776661795;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.33998703221387494;
        } else {
          result[0] += -0.11096001409545336;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 54))) {
    if (LIKELY(false || (data[0].qvalue <= 46))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 18))) {
          result[0] += -0.10381124910582794;
        } else {
          result[0] += -0.036053796621203345;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 18))) {
          result[0] += -0.4795079805215589;
        } else {
          result[0] += -0.3343803406439497;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 58))) {
        if (UNLIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -1.3594113743305207;
        } else {
          result[0] += -0.8394250588417054;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += -3.2252649987262227;
        } else {
          result[0] += -1.8098834428420432;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 234))) {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += 0.03700457930168985;
        } else {
          result[0] += 17.47663833459218;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 138))) {
          result[0] += -0.5601782767579584;
        } else {
          result[0] += 0.017180070918654807;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.2387817133710211;
        } else {
          result[0] += -4.100601076265661;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.843944808030713;
        } else {
          result[0] += -0.2201947167753442;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 200))) {
    if (LIKELY(false || (data[2].qvalue <= 48))) {
      if (LIKELY(false || (data[3].qvalue <= 188))) {
        if (LIKELY(false || (data[0].qvalue <= 66))) {
          result[0] += -0.0005495605667856106;
        } else {
          result[0] += -3.58726428370322;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 66))) {
          result[0] += -0.3697917053822814;
        } else {
          result[0] += 2.6442676259341997;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 194))) {
        if (LIKELY(false || (data[3].qvalue <= 140))) {
          result[0] += -0.1322526748337809;
        } else {
          result[0] += -0.9165228440058644;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 198))) {
          result[0] += 0.015399111888784775;
        } else {
          result[0] += -0.03293383786820958;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 50))) {
      if (UNLIKELY(false || (data[3].qvalue <= 206))) {
        result[0] += -1.74668638865153;
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 214))) {
          result[0] += 0.41404376746989746;
        } else {
          result[0] += -0.04320938041815753;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 208))) {
        if (LIKELY(false || (data[0].qvalue <= 72))) {
          result[0] += 0.15385322801835455;
        } else {
          result[0] += 0.8175728843952048;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.4756074244340375;
        } else {
          result[0] += -0.1983103516972138;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 56))) {
    if (LIKELY(false || (data[0].qvalue <= 24))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 26))) {
          result[0] += -0.08484652763250765;
        } else {
          result[0] += -0.028152113688230847;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 6))) {
          result[0] += -0.4321762504373764;
        } else {
          result[0] += -0.2651707672412985;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 58))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -1.0555429015159608;
        } else {
          result[0] += -0.7582275236447653;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += -2.9167088172746745;
        } else {
          result[0] += -1.4978527782972042;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 234))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[2].qvalue <= 38))) {
          result[0] += 0.02016281760562264;
        } else {
          result[0] += 0.5381136571935687;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.4649008843759111;
        } else {
          result[0] += 0.06919020340221031;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.15702109026459932;
        } else {
          result[0] += 2.1580531330984467;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.21686901261145552;
        } else {
          result[0] += -0.1001601769605728;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 42))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      if (LIKELY(false || (data[1].qvalue <= 8))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.19641092388804365;
        } else {
          result[0] += -0.055276022085065206;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 8))) {
          result[0] += -0.3392746616594763;
        } else {
          result[0] += -0.7843794295296335;
        }
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 44))) {
        result[0] += -2.238581113219261;
      } else {
        result[0] += -1.3538284465441337;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 200))) {
      if (LIKELY(false || (data[1].qvalue <= 62))) {
        if (LIKELY(false || (data[3].qvalue <= 188))) {
          result[0] += 0.017158471900924056;
        } else {
          result[0] += 1.4687521383872566;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 192))) {
          result[0] += -0.48506636615045484;
        } else {
          result[0] += -0.047229123064139726;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += -0.003343629263232336;
        } else {
          result[0] += -3.737171672379098;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.24658213105746754;
        } else {
          result[0] += 0.7366530590168956;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 60))) {
    if (LIKELY(false || (data[0].qvalue <= 24))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 18))) {
          result[0] += -0.07999229789885372;
        } else {
          result[0] += -0.024486695046509094;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 6))) {
          result[0] += -0.3591096819916043;
        } else {
          result[0] += -0.19716339864139187;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 58))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.8995184892360121;
        } else {
          result[0] += -0.6091088518301646;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += -2.423594094981318;
        } else {
          result[0] += -1.2236526361795572;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 236))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[2].qvalue <= 38))) {
          result[0] += 0.017590391024004715;
        } else {
          result[0] += 0.48376995639076664;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 192))) {
          result[0] += -0.4336376838327545;
        } else {
          result[0] += 0.04774652419020946;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.08141260086235247;
        } else {
          result[0] += 1.8715363113490904;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.2612265991556262;
        } else {
          result[0] += -0.11460675569674739;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 178))) {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[3].qvalue <= 134))) {
        if (LIKELY(false || (data[3].qvalue <= 128))) {
          result[0] += -0.010014087252989079;
        } else {
          result[0] += 0.44165882919228355;
        }
      } else {
        result[0] += 2.767749158126721;
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 30))) {
        if (LIKELY(false || (data[3].qvalue <= 140))) {
          result[0] += -1.265884517270139;
        } else {
          result[0] += -0.326555396705971;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += 0.17874279127753523;
        } else {
          result[0] += -0.10461815372106785;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (LIKELY(false || (data[1].qvalue <= 40))) {
        if (UNLIKELY(false || (data[3].qvalue <= 182))) {
          result[0] += -0.0291664181175232;
        } else {
          result[0] += 0.0321031669500698;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 180))) {
          result[0] += 1.2655710841109187;
        } else {
          result[0] += 3.8452706585121152;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 200))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.07346629730705066;
        } else {
          result[0] += -0.2790375844518317;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.030938073075268954;
        } else {
          result[0] += 0.2584663489831829;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 8))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      if (LIKELY(false || (data[1].qvalue <= 8))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.1897708646960375;
        } else {
          result[0] += -0.10591389661914655;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 8))) {
          result[0] += -0.27429189760805434;
        } else {
          result[0] += -0.6438250486029867;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 22))) {
        result[0] += -1.9822659349441527;
      } else {
        result[0] += -1.1856181401364942;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 176))) {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += 0.004593246195533139;
        } else {
          result[0] += 2.4929798120346622;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.9223952289248065;
        } else {
          result[0] += -0.044695212630605684;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 186))) {
        if (LIKELY(false || (data[1].qvalue <= 42))) {
          result[0] += 0.0022744264617153534;
        } else {
          result[0] += 0.9887394270605384;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 194))) {
          result[0] += -0.05980165741450547;
        } else {
          result[0] += 0.06763695160928662;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 58))) {
    if (LIKELY(false || (data[0].qvalue <= 46))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.23353499906984243;
        } else {
          result[0] += -0.030567457307068176;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 6))) {
          result[0] += -0.3026360332604611;
        } else {
          result[0] += -0.15352749305440105;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 58))) {
        if (UNLIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.9208640919129055;
        } else {
          result[0] += -0.4870238716204962;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += -2.028915634984555;
        } else {
          result[0] += -0.9910290916149432;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 230))) {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.012556621103070601;
        } else {
          result[0] += 0.27622010195921237;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 0.4583737829879677;
        } else {
          result[0] += -0.1957218213752224;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 68))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.13572258094742976;
        } else {
          result[0] += 1.654014792332844;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.10699664077497309;
        } else {
          result[0] += -0.26287929402538607;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 34))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      if (LIKELY(false || (data[1].qvalue <= 0))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.21031418917361988;
        } else {
          result[0] += -0.05299115840815619;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 8))) {
          result[0] += -0.21890068136134136;
        } else {
          result[0] += -0.5242896185723835;
        }
      }
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 22))) {
        result[0] += -1.6681880847613018;
      } else {
        result[0] += -0.9499344626594992;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 230))) {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += 0.003964550672321244;
        } else {
          result[0] += 0.248609910955958;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 80))) {
          result[0] += -0.17615619642542008;
        } else {
          result[0] += 0.41270129405980494;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.1074364755511174;
        } else {
          result[0] += -3.3754073861750165;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 56))) {
          result[0] += 0.5143961430823207;
        } else {
          result[0] += -0.23675266527233682;
        }
      }
    }
  }
  if (LIKELY(false || (data[0].qvalue <= 10))) {
    if (UNLIKELY(false || (data[2].qvalue <= 0))) {
      if (LIKELY(false || (data[1].qvalue <= 4))) {
        if (LIKELY(false || (data[0].qvalue <= 6))) {
          result[0] += -0.02091550687275058;
        } else {
          result[0] += 0.05471821127697738;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 8))) {
          result[0] += -0.06265624329964264;
        } else {
          result[0] += -0.003474992384641048;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 4))) {
        if (LIKELY(false || (data[0].qvalue <= 2))) {
          result[0] += -0.05813859439852901;
        } else {
          result[0] += 1.685043735887323;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 2))) {
          result[0] += 0.005356865798300517;
        } else {
          result[0] += -0.009951643674742359;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[0].qvalue <= 54))) {
        if (LIKELY(false || (data[1].qvalue <= 26))) {
          result[0] += 0.04421518555621426;
        } else {
          result[0] += 0.3861890671974087;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 2.733613769617948;
        } else {
          result[0] += 5.786070743288313;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 26))) {
        if (LIKELY(false || (data[1].qvalue <= 50))) {
          result[0] += -0.482797800051755;
        } else {
          result[0] += 2.5325565222740174;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.09547398872773903;
        } else {
          result[0] += -0.046573119312707335;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 4))) {
    if (LIKELY(false || (data[2].qvalue <= 28))) {
      if (LIKELY(false || (data[1].qvalue <= 10))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.08210078603934584;
        } else {
          result[0] += -0.11656505933452627;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 8))) {
          result[0] += -0.20987820191852383;
        } else {
          result[0] += -0.5384319635554019;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 44))) {
        result[0] += -1.9617384305207626;
      } else {
        result[0] += -1.2668292170304518;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 178))) {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += 0.003420268764872335;
        } else {
          result[0] += 2.213071489783301;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += -0.7776470923789272;
        } else {
          result[0] += -0.04327273641201862;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 186))) {
        if (LIKELY(false || (data[1].qvalue <= 40))) {
          result[0] += 0.00861831306648558;
        } else {
          result[0] += 1.848056525397011;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 196))) {
          result[0] += -0.04828241624484346;
        } else {
          result[0] += 0.05563476812699414;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 64))) {
    if (LIKELY(false || (data[0].qvalue <= 58))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 26))) {
          result[0] += -0.05078687692889694;
        } else {
          result[0] += -0.014356568909767928;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.17534628583710493;
        } else {
          result[0] += -0.5621430105402849;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 64))) {
        result[0] += -1.7740939281297767;
      } else {
        result[0] += -1.145018720260033;
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[0].qvalue <= 54))) {
        if (LIKELY(false || (data[3].qvalue <= 130))) {
          result[0] += 0.018993981486318878;
        } else {
          result[0] += 0.5365927638406978;
        }
      } else {
        result[0] += 15.05643708864848;
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 152))) {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          result[0] += -0.4728617369458554;
        } else {
          result[0] += 0.20752761050703802;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 38))) {
          result[0] += 0.09305968936956194;
        } else {
          result[0] += -0.009524064343686378;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 52))) {
    if (LIKELY(false || (data[0].qvalue <= 58))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.18727818329014223;
        } else {
          result[0] += -0.025890677188174106;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.15753058727404748;
        } else {
          result[0] += -0.5067434101778527;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 64))) {
        result[0] += -1.604397966136103;
      } else {
        result[0] += -1.0349207672706018;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 234))) {
      if (LIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += 0.020542942769893005;
        } else {
          result[0] += 13.61352872212728;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 152))) {
          result[0] += -0.20312529336650453;
        } else {
          result[0] += 0.019356250676449485;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.1529675914478733;
        } else {
          result[0] += -3.046054223456034;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.4578596813081651;
        } else {
          result[0] += -0.21317670897106453;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 100))) {
    if (LIKELY(false || (data[0].qvalue <= 58))) {
      if (LIKELY(false || (data[0].qvalue <= 12))) {
        if (LIKELY(false || (data[3].qvalue <= 88))) {
          result[0] += -0.009515774573242132;
        } else {
          result[0] += 0.10722986880689861;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.08947096205225108;
        } else {
          result[0] += -0.4843778472947222;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 64))) {
        result[0] += -1.9798399291992188;
      } else {
        result[0] += -1.155246785481771;
      }
    }
  } else {
    if (UNLIKELY(false || (data[2].qvalue <= 12))) {
      if (LIKELY(false || (data[3].qvalue <= 132))) {
        if (LIKELY(false || (data[3].qvalue <= 126))) {
          result[0] += 0.058783065329708964;
        } else {
          result[0] += 0.20552873062508797;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += 0.5012681102076195;
        } else {
          result[0] += 1.1655076967659646;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 152))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += -0.4052823818071666;
        } else {
          result[0] += 0.2663182250508323;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.09131312159878124;
        } else {
          result[0] += -0.023826012683033533;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 14))) {
    if (LIKELY(false || (data[0].qvalue <= 58))) {
      if (LIKELY(false || (data[0].qvalue <= 22))) {
        if (LIKELY(false || (data[1].qvalue <= 16))) {
          result[0] += -0.05587497206644211;
        } else {
          result[0] += -0.18871587509128254;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.471164876208749;
        } else {
          result[0] += -0.2899516754150391;
        }
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 64))) {
        result[0] += -1.2615578046052351;
      } else {
        result[0] += -0.824327716093797;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 230))) {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (LIKELY(false || (data[0].qvalue <= 54))) {
          result[0] += -0.0018346942873173101;
        } else {
          result[0] += 0.16034261671985583;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 0.3582006235217019;
        } else {
          result[0] += -0.15805343752114553;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.09120323986998596;
        } else {
          result[0] += -2.746553548719825;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += 0.020237036833509074;
        } else {
          result[0] += 0.5211988226836327;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 178))) {
    if (LIKELY(false || (data[2].qvalue <= 12))) {
      if (LIKELY(false || (data[3].qvalue <= 130))) {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += -0.005991282302809805;
        } else {
          result[0] += 0.21808256969336925;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += 0.3648159058387132;
        } else {
          result[0] += 1.0501265993764848;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 14))) {
        if (LIKELY(false || (data[3].qvalue <= 140))) {
          result[0] += -0.7560950076917835;
        } else {
          result[0] += -0.011609365583746693;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 114))) {
          result[0] += -0.9929890502662195;
        } else {
          result[0] += -0.018430796040420066;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (UNLIKELY(false || (data[0].qvalue <= 38))) {
        if (LIKELY(false || (data[3].qvalue <= 182))) {
          result[0] += 1.4941535253935205;
        } else {
          result[0] += 5.01069572504829;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 40))) {
          result[0] += -1.6721194921221052;
        } else {
          result[0] += 0.06600910509196743;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 194))) {
        if (LIKELY(false || (data[2].qvalue <= 50))) {
          result[0] += 0.06599082621417188;
        } else {
          result[0] += -0.5629778590534413;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 50))) {
          result[0] += -0.031091763815894752;
        } else {
          result[0] += 0.12480377527902276;
        }
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 38))) {
    if (LIKELY(false || (data[2].qvalue <= 36))) {
      if (LIKELY(false || (data[0].qvalue <= 14))) {
        if (UNLIKELY(false || (data[3].qvalue <= 2))) {
          result[0] += 0.1757544212263416;
        } else {
          result[0] += -0.03321362620945462;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.1209793900016442;
        } else {
          result[0] += -0.3390527655087508;
        }
      }
    } else {
      result[0] += -0.9544804904460907;
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 228))) {
      if (LIKELY(false || (data[2].qvalue <= 48))) {
        if (LIKELY(false || (data[3].qvalue <= 188))) {
          result[0] += 0.00850193251229063;
        } else {
          result[0] += 1.253315189034531;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 192))) {
          result[0] += -0.3207565361013706;
        } else {
          result[0] += 0.01040093706091699;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 0.07622775564010142;
        } else {
          result[0] += 0.6434668296100172;
        }
      } else {
        result[0] += -0.25382570830590884;
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 66))) {
    if (LIKELY(false || (data[2].qvalue <= 36))) {
      if (UNLIKELY(false || (data[3].qvalue <= 6))) {
        if (UNLIKELY(false || (data[3].qvalue <= 0))) {
          result[0] += -0.15891414485012728;
        } else {
          result[0] += -0.06083332526536694;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 30))) {
          result[0] += -0.025560713454775653;
        } else {
          result[0] += -0.009423279209306063;
        }
      }
    } else {
      result[0] += -0.8614186177253723;
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[2].qvalue <= 12))) {
        if (LIKELY(false || (data[3].qvalue <= 130))) {
          result[0] += 0.014946627847899083;
        } else {
          result[0] += 0.3928370868714558;
        }
      } else {
        result[0] += 12.309830587704978;
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 146))) {
        if (UNLIKELY(false || (data[2].qvalue <= 40))) {
          result[0] += -0.7278951537054682;
        } else {
          result[0] += 0.10042924111104164;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.048389262179927645;
        } else {
          result[0] += -0.024802333325858735;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 228))) {
    if (LIKELY(false || (data[3].qvalue <= 218))) {
      if (LIKELY(false || (data[3].qvalue <= 208))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.0026677754941615505;
        } else {
          result[0] += -0.11795346042813584;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 214))) {
          result[0] += 0.25660651477274304;
        } else {
          result[0] += 0.12738714081176464;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 80))) {
        if (UNLIKELY(false || (data[3].qvalue <= 222))) {
          result[0] += -0.36182422209697296;
        } else {
          result[0] += -0.09068035614479159;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 226))) {
          result[0] += 0.29626333166012725;
        } else {
          result[0] += 0.501580802754658;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 76))) {
      if (LIKELY(false || (data[3].qvalue <= 240))) {
        if (LIKELY(false || (data[1].qvalue <= 74))) {
          result[0] += 0.04606551220989991;
        } else {
          result[0] += 1.6768793160307642;
        }
      } else {
        result[0] += -2.4957903161863;
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 56))) {
        if (LIKELY(false || (data[3].qvalue <= 240))) {
          result[0] += -0.001995007368938002;
        } else {
          result[0] += 0.8600447828586285;
        }
      } else {
        result[0] += -0.22613400126344585;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 100))) {
    if (LIKELY(false || (data[2].qvalue <= 28))) {
      if (LIKELY(false || (data[1].qvalue <= 10))) {
        if (LIKELY(false || (data[3].qvalue <= 88))) {
          result[0] += -0.007032593132406662;
        } else {
          result[0] += 0.10789191115072033;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 94))) {
          result[0] += -0.14776386905895944;
        } else {
          result[0] += -0.04740764971575276;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 44))) {
        result[0] += -1.5670684486389161;
      } else {
        result[0] += -0.7503883746818261;
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 10))) {
      result[0] += 2.769067636305286;
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 28))) {
        if (LIKELY(false || (data[3].qvalue <= 134))) {
          result[0] += 0.07602934068056531;
        } else {
          result[0] += 2.8183202544858474;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 146))) {
          result[0] += -0.16233532230561315;
        } else {
          result[0] += 0.01202817781399166;
        }
      }
    }
  }
  if (LIKELY(false || (data[0].qvalue <= 54))) {
    if (LIKELY(false || (data[0].qvalue <= 50))) {
      if (LIKELY(false || (data[0].qvalue <= 48))) {
        if (LIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.0029886117550836865;
        } else {
          result[0] += -0.06408425115195845;
        }
      } else {
        result[0] += 0.26602660123013633;
      }
    } else {
      result[0] += -0.07929562686095666;
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 30))) {
      if (LIKELY(false || (data[2].qvalue <= 46))) {
        if (UNLIKELY(false || (data[2].qvalue <= 36))) {
          result[0] += 3.7119815171616426;
        } else {
          result[0] += 0.547443312065942;
        }
      } else {
        result[0] += 4.166473594393049;
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 48))) {
        if (UNLIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += 0.0017977573714891022;
        } else {
          result[0] += 0.23845299048938143;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += -0.037150860814943704;
        } else {
          result[0] += 0.14564257745376;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 178))) {
    if (LIKELY(false || (data[0].qvalue <= 60))) {
      if (LIKELY(false || (data[0].qvalue <= 50))) {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += -0.007183108829239199;
        } else {
          result[0] += 0.2394491795303417;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 140))) {
          result[0] += -0.6344022675495424;
        } else {
          result[0] += 0.007260844712462359;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 142))) {
        if (UNLIKELY(false || (data[3].qvalue <= 138))) {
          result[0] += -0.08417562885148172;
        } else {
          result[0] += 0.5906773558105556;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 0.18764238988337378;
        } else {
          result[0] += -2.326712134777498;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (UNLIKELY(false || (data[0].qvalue <= 38))) {
        if (LIKELY(false || (data[3].qvalue <= 182))) {
          result[0] += 1.339431215953403;
        } else {
          result[0] += 4.521662556143368;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 52))) {
          result[0] += 0.7292394052527168;
        } else {
          result[0] += -0.012391112078889212;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 194))) {
        if (LIKELY(false || (data[2].qvalue <= 50))) {
          result[0] += 0.05246317581619821;
        } else {
          result[0] += -0.4745475472223324;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 50))) {
          result[0] += -0.024886056556697605;
        } else {
          result[0] += 0.10675635333641575;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 178))) {
    if (LIKELY(false || (data[1].qvalue <= 46))) {
      if (LIKELY(false || (data[1].qvalue <= 44))) {
        if (LIKELY(false || (data[3].qvalue <= 170))) {
          result[0] += -0.0035541639556744805;
        } else {
          result[0] += 0.23776616044647245;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 22))) {
          result[0] += 1.6540061310768128;
        } else {
          result[0] += -0.007316129265448475;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 24))) {
        if (UNLIKELY(false || (data[3].qvalue <= 156))) {
          result[0] += -0.9207533250588412;
        } else {
          result[0] += -0.19604021761441262;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 26))) {
          result[0] += 0.39069615612658914;
        } else {
          result[0] += -0.07078300532699774;
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[3].qvalue <= 186))) {
      if (LIKELY(false || (data[1].qvalue <= 42))) {
        if (UNLIKELY(false || (data[3].qvalue <= 182))) {
          result[0] += -0.0537103470292483;
        } else {
          result[0] += 0.01762700340556235;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 180))) {
          result[0] += 0.6253090474974919;
        } else {
          result[0] += 2.807307553162299;
        }
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 202))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          result[0] += 0.04721793457525833;
        } else {
          result[0] += -0.17271753814495214;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += -0.01901583571747938;
        } else {
          result[0] += 0.1447237594141037;
        }
      }
    }
  }
  if (LIKELY(false || (data[0].qvalue <= 78))) {
    if (LIKELY(false || (data[0].qvalue <= 76))) {
      if (LIKELY(false || (data[0].qvalue <= 54))) {
        if (LIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += 0.0021335393865604085;
        } else {
          result[0] += -0.07177648191161112;
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 30))) {
          result[0] += 2.634354959784485;
        } else {
          result[0] += 0.004247555141275988;
        }
      }
    } else {
      result[0] += 0.45583923039077984;
    }
  } else {
    result[0] += -0.24432320779772257;
  }
  if (UNLIKELY(false || (data[3].qvalue <= 6))) {
    if (LIKELY(false || (data[0].qvalue <= 58))) {
      if (LIKELY(false || (data[0].qvalue <= 46))) {
        if (LIKELY(false || (data[1].qvalue <= 18))) {
          result[0] += -0.053137056565348964;
        } else {
          result[0] += -0.1615091542830301;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 50))) {
          result[0] += -0.48585747241973887;
        } else {
          result[0] += -0.2905982590516409;
        }
      }
    } else {
      result[0] += -1.0460150627295177;
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 228))) {
      if (LIKELY(false || (data[3].qvalue <= 218))) {
        if (LIKELY(false || (data[3].qvalue <= 208))) {
          result[0] += -0.0007595050498410175;
        } else {
          result[0] += 0.17149519474492791;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 68))) {
          result[0] += 0.2696969214216211;
        } else {
          result[0] += -0.148602349719807;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          result[0] += 0.041070141212163;
        } else {
          result[0] += 0.27300094236384387;
        }
      } else {
        result[0] += -0.220040733112148;
      }
    }
  }
  if (UNLIKELY(false || (data[3].qvalue <= 66))) {
    if (LIKELY(false || (data[0].qvalue <= 54))) {
      if (UNLIKELY(false || (data[3].qvalue <= 8))) {
        if (LIKELY(false || (data[0].qvalue <= 24))) {
          result[0] += -0.04786216375242139;
        } else {
          result[0] += -0.2854615034882365;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 44))) {
          result[0] += -0.0157676019115889;
        } else {
          result[0] += -0.00516823734649734;
        }
      }
    } else {
      result[0] += -0.8536656070967852;
    }
  } else {
    if (LIKELY(false || (data[1].qvalue <= 28))) {
      if (LIKELY(false || (data[0].qvalue <= 54))) {
        if (LIKELY(false || (data[3].qvalue <= 130))) {
          result[0] += 0.01178608871677298;
        } else {
          result[0] += 0.31573188820888726;
        }
      } else {
        result[0] += 10.428866252899171;
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 146))) {
        if (LIKELY(false || (data[0].qvalue <= 60))) {
          result[0] += -0.39214777400557815;
        } else {
          result[0] += 0.15464362462318024;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.05795968864116782;
        } else {
          result[0] += -0.018277034463468993;
        }
      }
    }
  }

  // Apply base_scores
  result[0] += 0;

  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void cpufj_predictor::postprocess(double* result)
{
  // Do nothing
}

// Feature names array
const char* cpufj_predictor::feature_names[cpufj_predictor::NUM_FEATURES] = {
  "n_vars", "n_cstrs", "total_nnz", "mem_total_mb"};
