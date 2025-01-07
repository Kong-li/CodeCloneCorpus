    public void setVariables(final Map<String, Object> variables) {
        if (variables == null || variables.isEmpty()) {
            return;
        }
        // First perform reserved word check on every variable name to be inserted
        for (final String name : variables.keySet()) {
            if (SESSION_VARIABLE_NAME.equals(name) ||
                    PARAM_VARIABLE_NAME.equals(name) ||
                    APPLICATION_VARIABLE_NAME.equals(name)) {
                throw new IllegalArgumentException(
                        "Cannot set variable called '" + name + "' into web variables map: such name is a reserved word");
            }
        }
        this.exchangeAttributeMap.setVariables(variables);
    }

