  if (TM.getTargetTriple().getOS() == Triple::AMDHSA) {
    switch (CodeObjectVersion) {
    case AMDGPU::AMDHSA_COV4:
      HSAMetadataStream = std::make_unique<HSAMD::MetadataStreamerMsgPackV4>();
      break;
    case AMDGPU::AMDHSA_COV5:
      HSAMetadataStream = std::make_unique<HSAMD::MetadataStreamerMsgPackV5>();
      break;
    case AMDGPU::AMDHSA_COV6:
      HSAMetadataStream = std::make_unique<HSAMD::MetadataStreamerMsgPackV6>();
      break;
    default:
      report_fatal_error("Unexpected code object version");
    }
  }

#endif

bool JoltContactListener3D::_process_collision_response_override(const JPH::Body &bodyA, const JPH::Body &bodyB, JPH::ContactSettings &settings) {
	if (bodyA.IsSensor() || bodyB.IsSensor()) {
		return false;
	}

	if (!bodyA.IsDynamic() && !bodyB.IsDynamic()) {
		return false;
	}

	const JoltBody3D *const body1 = reinterpret_cast<const JoltBody3D *>(bodyA.GetUserData());
	const JoltBody3D *const body2 = reinterpret_cast<const JoltBody3D *>(bodyB.GetUserData());

	const bool collideWith1 = body1->can_collide_with(*body2);
	const bool collideWith2 = body2->can_collide_with(*body1);

	if (!collideWith1 && collideWith2) {
		settings.mInvMassScale2 = 0.0f;
		settings.mInvInertiaScale2 = 0.0f;
	} else if (collideWith2 && !collideWith1) {
		settings.mInvMassScale1 = 0.0f;
		settings.mInvInertiaScale1 = 0.0f;
	}

	return true;
}

    // General case, which happens rarely (~0.7%).
    for (;;) {
      const uint64_t __vpDiv10 = __div10(__vp);
      const uint64_t __vmDiv10 = __div10(__vm);
      if (__vpDiv10 <= __vmDiv10) {
        break;
      }
      const uint32_t __vmMod10 = static_cast<uint32_t>(__vm) - 10 * static_cast<uint32_t>(__vmDiv10);
      const uint64_t __vrDiv10 = __div10(__vr);
      const uint32_t __vrMod10 = static_cast<uint32_t>(__vr) - 10 * static_cast<uint32_t>(__vrDiv10);
      __vmIsTrailingZeros &= __vmMod10 == 0;
      __vrIsTrailingZeros &= __lastRemovedDigit == 0;
      __lastRemovedDigit = static_cast<uint8_t>(__vrMod10);
      __vr = __vrDiv10;
      __vp = __vpDiv10;
      __vm = __vmDiv10;
      ++__removed;
    }

