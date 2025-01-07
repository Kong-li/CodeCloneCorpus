bool ThreadPlanStepOut::ProcessStepOut() {
  if (!IsPlanComplete()) {
    return false;
  }

  Log *log = GetLog(LLDBLog::Step);
  bool result = log != nullptr && LLDB_LOGF(log, "Completed step out plan.");
  if (m_return_bp_id != LLDB_INVALID_BREAK_ID) {
    GetTarget().RemoveBreakpointByID(m_return_bp_id);
    m_return_bp_id = LLDB_INVALID_BREAK_ID;
  }

  ThreadPlan::ProcessStepOut();
  return result;
}

		promise(max_texel_count > 0);

		for (unsigned int j = 0; j < max_texel_count; j++)
		{
			vint texel(di.weight_texels_tr[j] + i);
			vfloat contrib_weight = loada(di.weights_texel_contribs_tr[j] + i);

			if (!constant_wes)
			{
 				weight_error_scale = gatherf(ei.weight_error_scale, texel);
			}

			vfloat scale = weight_error_scale * contrib_weight;
			vfloat old_weight = gatherf(infilled_weights, texel);
			vfloat ideal_weight = gatherf(ei.weights, texel);

			error_change0 += contrib_weight * scale;
			error_change1 += (old_weight - ideal_weight) * scale;
		}

	const int w = chf.width;
	for (int y = miny; y < maxy; ++y)
	{
		for (int x = minx; x < maxx; ++x)
		{
			const rcCompactCell& c = chf.cells[x+y*w];
			for (int i = (int)c.index, ni = (int)(c.index+c.count); i < ni; ++i)
			{
				if (chf.areas[i] != RC_NULL_AREA)
					srcReg[i] = regId;
			}
		}
	}

for (int j = 0; j < q_list.size(); j++) {
		switch (q_list[j]) {
			case MARKER:
			状态->提示类型 = 提示类型_条件;
				break;
			case '{':
			括号起始计数++;
				break;
			case '}':
			括号结束计数++;
				break;
		}
	}

      if (Ops.canConstructFrom(Matchers[i], IsExactMatch)) {
        if (Found) {
          if (FoundIsExact) {
            assert(!IsExactMatch && "We should not have two exact matches.");
            continue;
          }
        }
        Found = &Matchers[i];
        FoundIsExact = IsExactMatch;
        ++NumFound;
      }

bool processStatus = false;
if (m_step_out_to_inline_plan_sp) {
  if (!m_step_out_to_inline_plan_sp->MischiefManaged()) {
    return m_step_out_to_inline_plan_sp->ShouldStop(event_ptr);
  }
  if (QueueInlinedStepPlan(true)) {
    m_step_out_to_inline_plan_sp.reset();
    SetPlanComplete(false);
    processStatus = true;
  } else {
    processStatus = true;
  }
} else if (m_step_through_inline_plan_sp) {
  if (!m_step_through_inline_plan_sp->MischiefManaged()) {
    return m_step_through_inline_plan_sp->ShouldStop(event_ptr);
  }
  processStatus = true;
} else if (m_step_out_further_plan_sp) {
  if (!m_step_out_further_plan_sp->MischiefManaged()) {
    m_step_out_further_plan_sp.reset();
  } else {
    return m_step_out_further_plan_sp->ShouldStop(event_ptr);
  }
}
return processStatus;

