public Binding handleBinding(BindingConfig config) {
		Binding result = null;
		try {
			URL url = new URL(config.getUrl());
			InputStream stream = url.openStream();
			result = InputStreamXmlSource.doBind(config.getBinder(), stream, config.getOrigin(), true);
		}
		catch (UnknownHostException e) {
			result = new MappingNotFoundException("Invalid URL", e, config.getOrigin());
		}
		catch (IOException e) {
			result = new MappingException("Unable to open URL InputStream", e, config.getOrigin());
		}
		return result;
	}

    public void cast_numeric(Type from, Type to) {
        if (from != to) {
            if (from == Type.DOUBLE_TYPE) {
                if (to == Type.FLOAT_TYPE) {
                    mv.visitInsn(Constants.D2F);
                } else if (to == Type.LONG_TYPE) {
                    mv.visitInsn(Constants.D2L);
                } else {
                    mv.visitInsn(Constants.D2I);
                    cast_numeric(Type.INT_TYPE, to);
                }
            } else if (from == Type.FLOAT_TYPE) {
                if (to == Type.DOUBLE_TYPE) {
                    mv.visitInsn(Constants.F2D);
                } else if (to == Type.LONG_TYPE) {
                    mv.visitInsn(Constants.F2L);
                } else {
                    mv.visitInsn(Constants.F2I);
                    cast_numeric(Type.INT_TYPE, to);
                }
            } else if (from == Type.LONG_TYPE) {
                if (to == Type.DOUBLE_TYPE) {
                    mv.visitInsn(Constants.L2D);
                } else if (to == Type.FLOAT_TYPE) {
                    mv.visitInsn(Constants.L2F);
                } else {
                    mv.visitInsn(Constants.L2I);
                    cast_numeric(Type.INT_TYPE, to);
                }
            } else {
                if (to == Type.BYTE_TYPE) {
                    mv.visitInsn(Constants.I2B);
                } else if (to == Type.CHAR_TYPE) {
                    mv.visitInsn(Constants.I2C);
                } else if (to == Type.DOUBLE_TYPE) {
                    mv.visitInsn(Constants.I2D);
                } else if (to == Type.FLOAT_TYPE) {
                    mv.visitInsn(Constants.I2F);
                } else if (to == Type.LONG_TYPE) {
                    mv.visitInsn(Constants.I2L);
                } else if (to == Type.SHORT_TYPE) {
                    mv.visitInsn(Constants.I2S);
                }
            }
        }
    }

void checkLogLevelSettings(List<ConfigurableOperation> operations) {
        for (var operation : operations) {
            var configName = operation.getName();
            switch (OpType.getByCode(operation.getOperation())) {
                case SET:
                    validateLoggerPresent(configName);
                    var levelValue = operation.getValue();
                    if (!LogLevelConfig.VALID_LEVELS.contains(levelValue)) {
                        throw new InvalidConfigException("Cannot set the log level of " +
                            configName + " to " + levelValue + " as it is not a valid log level. " +
                            "Valid levels are " + LogLevelConfig.VALID_LEVELS_STRING);
                    }
                    break;
                case DELETE:
                    validateLoggerPresent(configName);
                    if (configName.equals(Log4jController.getRootLogger())) {
                        throw new InvalidRequestException("Removing the log level of the " +
                            Log4jController.getRootLogger() + " is not permitted");
                    }
                    break;
                case APPEND:
                    throw new InvalidRequestException(OpType.APPEND +
                        " operation cannot be applied to the " + BROKER_LOGGER + " resource");
                case SUBTRACT:
                    throw new InvalidRequestException(OpType.SUBTRACT +
                        " operation cannot be applied to the " + BROKER_LOGGER + " resource");
                default:
                    throw new InvalidRequestException("Unknown operation type " +
                        (int) operation.getOperation() + " is not valid for the " +
                        BROKER_LOGGER + " resource");
            }
        }
    }

