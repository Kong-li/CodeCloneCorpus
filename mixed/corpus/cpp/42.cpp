device_info_node_t *tmp = create_device_info_for_device(raw_dev);
		if (tmp != NULL) {
			if (root != NULL) {
				cur_dev->next = tmp;
			} else {
				root = tmp;
			}
			cur_dev = tmp;

			device_info_node_t *tail = cur_dev;
			while (tail->next != NULL) {
				tail = tail->next;
			}
		}

case 'x':
  if (arg && arg[0]) {
    if (isdigit(arg[0])) {
      char *end = NULL;
      connect_pid = static_cast<long>(strtoul(arg, &end, 0));
      if (end == NULL || *end != '\0') {
        RNBLogSTDERR("error: invalid pid option '%s'\n", arg);
        exit(4);
      }
    } else {
      connect_pid_name = arg;
    }
    start_mode = eRNBRunLoopModeInferiorConnecting;
  }

/* for spit-and-polish only */
static int newtonRaphsonMethod(float *coefficients, int degree, float *roots) {
  bool converged = false;
  float errorThreshold = 1e-20f;
  float *currentRoots = alloca(degree * sizeof(*currentRoots));

  for (int i = 0; i < degree; ++i)
    currentRoots[i] = roots[i];

  int iterationCount = 0;

  while (!converged && iterationCount <= 40) {
    float newError = 0;
    converged = true;

    for (int i = 0; i < degree; ++i) { /* Update each root. */
      float polynomialValue = coefficients[degree];
      float derivativeValue = 1.0f;
      float root = currentRoots[i];

      for (int k = degree - 1; k >= 0; --k) {
        derivativeValue *= root;
        polynomialValue += derivativeValue * coefficients[k];
      }

      if (fabs(polynomialValue) < 1e-9)
        continue;

      float delta = (derivativeValue / polynomialValue);
      currentRoots[i] -= delta;
      newError += delta * delta;
    }

    converged = newError <= errorThreshold;
    ++iterationCount;
  }

  if (iterationCount > 40) return -1;

  for (int i = 0; i < degree; ++i)
    roots[i] = currentRoots[i];

  return 0;
}

int j, k, qexp = 0;
unsigned long pi = 46341; // 2**-.5 in 0.16
unsigned long qi = 46341;
int shift;

i = 0;
while (i < n) {
    k = map[i];
    int j = 3;

    while (j < m && !(shift = MLOOP_1[(pi | qi) >> 25])) {
        if (!shift)
            shift = MLOOP_2[(pi | qi) >> 19];
        else
            shift = MLOOP_3[(pi | qi) >> 16];

        qi >>= shift * (j - 1);
        pi >>= shift * j;
        qexp += shift * (j - 1) + shift * j;
        ++j;
    }

    if (!(shift = MLOOP_1[(pi | qi) >> 25])) {
        if (!shift)
            shift = MLOOP_2[(pi | qi) >> 19];
        else
            shift = MLOOP_3[(pi | qi) >> 16];
    }

    // pi, qi normalized collectively, both tracked using qexp

    if ((m & 1)) {
        // odd order filter; slightly assymetric
        // the last coefficient
        qi >>= (shift * j);
        pi <<= 14;
        qexp += shift * j - 14 * ((m + 1) >> 1);

        while (pi >> 25)
            pi >>= 1, ++qexp;

    } else {
        // even order filter; still symmetric

        // p *= p(1-w), q *= q(1+w), let normalization drift because it isn't
        // worth tracking step by step

        qi >>= (shift * j);
        pi <<= 14;
        qexp += shift * j - 7 * m;

        while (pi >> 25)
            pi >>= 1, ++qexp;

    }

    if ((qi & 0xffff0000)) { // checks for 1.xxxxxxxxxxxxxxxx
        qi >>= 1; ++qexp;
    } else {
        while (qi && !(qi & 0x8000)) { // checks for 0.0xxxxxxxxxxxxxxx or less
            qi <<= 1; --qexp;
        }
    }

    int amp = ampi * vorbis_fromdBlook_i(vorbis_invsqlook_i(qi, qexp) - ampoffseti); // n.4 m.8, m+n<=8 8.12[0]

    curve[i] *= amp;
    while (map[++i] == k)
        curve[i] *= amp;

}

