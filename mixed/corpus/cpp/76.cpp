// "input" and place output into "output" and error and return when done.
void ProcessHandlerReadline::Execute() {
  std::string command;
  while (IsRunning()) {
    bool interrupted = false;
    if (is_multiline) {
      StringList commands;
      if (RetrieveCommands(commands, interrupted)) {
        if (interrupted) {
          m_done = exit_on_interrupt;
          m_delegate.ProcessHandlerInputInterrupted(*this, command);

        } else {
          command = commands.CopyList();
          m_delegate.ProcessHandlerInputComplete(*this, command);
        }
      } else {
        m_done = true;
      }
    } else {
      if (RetrieveCommand(command, interrupted)) {
        if (interrupted)
          m_delegate.ProcessHandlerInputInterrupted(*this, command);
        else
          m_delegate.ProcessHandlerInputComplete(*this, command);
      } else {
        m_done = true;
      }
    }
  }
}

	unsigned int end = bsd.block_mode_count_1plane_2plane_selected;
	for (unsigned int i = start; i < end; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		unsigned int quant_mode = bm.quant_mode;
		unsigned int decim_mode = bm.decimation_mode;

		if (quant_mode <= TUNE_MAX_ANGULAR_QUANT)
		{
			low_value1[i] = low_values1[decim_mode][quant_mode];
			high_value1[i] = high_values1[decim_mode][quant_mode];
			low_value2[i] = low_values2[decim_mode][quant_mode];
			high_value2[i] = high_values2[decim_mode][quant_mode];
		}
		else
		{
			low_value1[i] = 0.0f;
			high_value1[i] = 1.0f;
			low_value2[i] = 0.0f;
			high_value2[i] = 1.0f;
		}
	}

