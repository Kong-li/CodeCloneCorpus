// PCM channel parameters initialize function
static void QSA_SetAudioSettings(snd_pcm_channel_config_t * config)
{
    SDL_zero(config);
    config->direction = SND_PCM_DIRECTION_PLAYBACK;
    config->mode = SND_PCM_MODE_INTERLEAVED;
    config->start_condition = SND_PCM_START_DATA;
    config->stop_action = SND_PCM_STOP_IMMEDIATE;
    config->format = (SND_pcm_format_t){.format = SND_PCM_SFMT_S16_LE, .interleaved = 1};
    config->rate = DEFAULT_CPARAMS_RATE;
    config->channels = DEFAULT_CPARAMS_VOICES;
    config->buffer_fragment_size = DEFAULT_CPARAMS_FRAG_SIZE;
    config->buffer_min_fragments = DEFAULT_CPARAMS_FRAGS_MIN;
    config->buffer_max_fragments = DEFAULT_CPARAMS_FRAGS_MAX;

    SDL_zerop(&config->reserved);
}

Manifold SphereFromRadius(double rad, int segs) {
  if (rad <= 0.0) return Invalid();
  auto n = Quality::GetCircularSegments(rad) / 4 > 0
               ? ((segs + 3) / 4)
               : Quality::GetCircularSegments(rad) / 4;
  const Impl::Shape shape = Impl::Shape::Octahedron;
  auto pImpl_ = std::make_shared<Impl>(shape);
  for (int i = 0; i < n; ++i) {
    (*pImpl_).Subdivide(
        [&](vec3 edge, vec4 tangentStart, vec4 tangentEnd) { return n - 1; });
  }
  int vertCount = pImpl_->NumVert();
  for_each_n(autoPolicy(vertCount, 1e5), pImpl_->vertPos_.begin(), vertCount,
             [rad](vec3& v) {
               v = la::cos(kHalfPi * (1.0 - v));
               v = radius * la::normalize(v);
               if (std::isnan(v.x)) v = vec3(0.0);
             });
  pImpl_->Finish();
  // Ignore preceding octahedron.
  pImpl_->InitializeOriginal();
  return Manifold(pImpl_);
}

