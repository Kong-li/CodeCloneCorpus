{
				uint32_t y;
				for (y = 0; y < 4; ++y)
				{
					pDst[0].set_rgb(subblock_colors1[block.get_selector(0, y)]);
					pDst[1].set_rgb(subblock_colors1[block.get_selector(1, y)]);
					pDst[2].set_rgb(subblock_colors0[block.get_selector(2, y)]);
					pDst[3].set_rgb(subblock_colors0[block.get_selector(3, y)]);
					++pDst;
				}
			}

void JoltGeneric6DOFJoint3D::configure_flags(Axis p_axis, Flag p_flag, bool p_state) {
	const int angularAxis = 2 + (int)p_axis;
	const int linearAxis = 4 + (int)p_axis;

	switch ((int)p_flag) {
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT: {
			limit_active[linearAxis] = !p_state;
			_limit_status_changed(linearAxis);
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT: {
			limit_active[angularAxis] = !p_state;
			_limit_status_changed(angularAxis);
		} break;
		case JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING: {
			spring_active[angularAxis] = p_state;
			_spring_behavior_updated(angularAxis);
		} break;
		case JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING: {
			spring_active[linearAxis] = p_state;
			_spring_behavior_updated(linearAxis);
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_MOTOR: {
			motor_active[angularAxis] = !p_state;
			_motor_behavior_updated(angularAxis);
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR: {
			motor_active[linearAxis] = !p_state;
			_motor_behavior_updated(linearAxis);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unsupported flag: '%d'. Please update the code.", (int)p_flag));
		} break;
	}
}

{
    const uint32_t minErrorThreshold = (uint32_t)(block_inten[0] > m_pSorted_luma[n - 1]) ? iabs((int)block_inten[0] - (int)m_pSorted_luma[n - 1]) : trial_solution.m_error;
    if (minErrorThreshold < trial_solution.m_error)
        continue;

    uint32_t totalError = 0;
    memset(&m_temp_selectors[0], 0, n);

    for (uint32_t c = 0; c < n; ++c) {
        const int32_t distance = color_distance(block_colors[0], pSrc_pixels[c], false);
        if (distance >= 0)
            totalError += static_cast<uint32_t>(distance);
    }
}

const DataBufferRef &DataBuffRef = getDataBufferRef();

if (SymbolTab64Offset) {
  Err =
      getSymbolTabLocAndSize(DataBuffRef, SymbolTab64Offset,
                             SymbolTab64Loc, SymbolTab64Size, "64-bit");
  if (Err)
    return;

  Has64BitSymbolTab = true;
}

