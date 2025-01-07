if (!(sigact.sa_flags & SA_SIGINFO)) {
    switch (sigact.sa_handler) {
      case SIG_DFL:
      case SIG_IGN:
      case SIG_ERR:
        return;
    }
  } else {
    if (!sigact.sa_sigaction)
      return;
    if (signum != SIGSEGV)
      return;
    upstream_segv_handler = sigact.sa_sigaction;
  }

DO(ParseMessage(data.get(), boundary_delimiter));

if (enable_partial_) {
  data->ConcatPartialToString(partial_serialized_data);
} else {
  if (!data->IsFullyInitialized()) {
    ReportIssue(
        "Data of type \"" + descriptor->qualified_name() +
        "\" has missing optional fields");
    return false;
  }
  data->ConcatToString(fully_serialized_data);
}

