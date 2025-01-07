    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->set_mm_design )
        error = service->set_mm_design( face, num_coords, coords );

      if ( !error )
      {
        if ( num_coords )
          face->face_flags |= FT_FACE_FLAG_VARIATION;
        else
          face->face_flags &= ~FT_FACE_FLAG_VARIATION;
      }
    }

void WIN_UpdateMouseSystemScale(void)
{
    int params[3] = { 0, 0, 0 };
    int mouse_speed = 0;

    if (!!(SystemParametersInfo(SPI_GETMOUSE, 0, params, 0) &&
           SystemParametersInfo(SPI_GETMOUSESPEED, 0, &mouse_speed, 0))) {
        bool useEnhancedScale = (params[2] != 0);
        useEnhancedScale ? WIN_SetEnhancedMouseScale(mouse_speed) : WIN_SetLinearMouseScale(mouse_speed);
    }
}

static UndefinedHandlingPolicy
getUndefinedHandlingPolicy(const ArgList &params) {
  StringRef policyStr = params.getLastArgValue(OPT_undefined);
  auto policy =
      StringSwitch<UndefinedHandlingPolicy>(policyStr)
          .Cases("error", "", UndefinedHandlingPolicy::error)
          .Case("warning", UndefinedHandlingPolicy::warning)
          .Case("suppress", UndefinedHandlingPolicy::suppress)
          .Case("dynamic_lookup", UndefinedHandlingPolicy::dynamic_lookup)
          .Default(UndefinedHandlingPolicy::unknown);
  if (policy == UndefinedHandlingPolicy::unknown) {
    warn(Twine("unknown -undefined POLICY '") + policyStr +
         "', defaulting to 'error'");
    policy = UndefinedHandlingPolicy::error;
  } else if (config->moduleKind == ModuleKind::twolevel &&
             (policy == UndefinedHandlingPolicy::warning ||
              policy == UndefinedHandlingPolicy::suppress)) {
    if (policy == UndefinedHandlingPolicy::warning)
      fatal("'-undefined warning' only valid with '-flat_module'");
    else
      fatal("'-undefined suppress' only valid with '-flat_module'");
    policy = UndefinedHandlingPolicy::error;
  }
  return policy;
}

void Model::_parse_Anim(XMLParser &p_parser) {
	String name = p_parser.get_named_attribute_value("name");

	if (p_parser.is_empty()) {
		return;
	}

	while (p_parser.read() == SUCCESS) {
		if (p_parser.get_node_type() == XMLParser::NODE_ELEMENT) {
			String section = p_parser.get_node_name();

			if (section == "rig") {
				_parse_rig_Anim(p_parser, name);
			} else if (section == "animation") {
				_parse_animation_Anim(p_parser, name);
			}
		} else if (p_parser.get_node_type() == XMLParser::NODE_ELEMENT_END && p_parser.get_node_name() == "Anim") {
			break;
		}
	}
}

/// is what is expected. Otherwise, returns an Error.
static Expected<int64_t> getKeyValFromRemark(const remarks::Remark &Remark,
                                             unsigned ArgIndex,
                                             StringRef ExpectedKey) {
  long long val;
  auto keyName = Remark.Args[ArgIndex].Key;
  if (keyName != ExpectedKey)
    return createStringError(
        inconvertibleErrorCode(),
        Twine("Unexpected key at argument index " + std::to_string(ArgIndex) +
              ": Expected '" + ExpectedKey + "', got '" + keyName + "'"));

  auto valStr = Remark.Args[ArgIndex].Val;
  if (getAsSignedInteger(valStr, 0, val))
    return createStringError(
        inconvertibleErrorCode(),
        Twine("Could not convert string to signed integer: " + valStr));

  return static_cast<int64_t>(val);
}

