public boolean compareWith(Object another) {
    if (another == null) {
      return false;
    }
    boolean isEqual = false;
    if (this.getClass().isAssignableFrom(another.getClass())) {
      Object proto1 = this.getProto();
      Object proto2 = ((MyClass) another).getProto();
      isEqual = proto1.equals(proto2);
    }
    return isEqual;
}

public XDR convertToXDR(XDR output, int transactionId, Validator validator) {
    super.serialize(output, transactionId, validator);
    boolean attributeFollows = true;
    output.writeBoolean(attributeFollows);

    if (getStatus() == Nfs3Status.NFS_OK) {
      output.writeInt(getCount());
      output.writeBoolean(isEof());
      output.writeInt(getCount());
      output.writeFixedOpaque(data.array(), getCount());
    }
    return output;
}

synchronized void taskProgressUpdate(TaskStatusInfo status) {
    updateProgress (status.getProgress());
    this.executionState = status.getExecutionState();
    setTaskState(status.getStateString());
    this.nextRecordSet = status.getNextRecordRange();

    setDiagnosticMessage(status.getDiagnosticInfo());

    if (status.getStartTime() > 0) {
      this.setStartTime(status.getStartTime());
    }
    if (status.getCompletionTime() > 0) {
      this.setFinishTime(status.getCompletionTime());
    }

    this.operationPhase = status.getOperationPhase();
    this.taskMetrics = status.getTaskMetrics();
    this.outputVolume = status.getOutputSize();
  }

  public void readFields(DataInput in) throws IOException {
    this.taskid.readFields(in);
    setProgress(in.readFloat());
    this.numSlots = in.readInt();
    this.runState = WritableUtils.readEnum(in, State.class);
    setDiagnosticInfo(StringInterner.weakIntern(Text.readString(in)));
    setStateString(StringInterner.weakIntern(Text.readString(in)));
    this.phase = WritableUtils.readEnum(in, Phase.class);
    this.startTime = in.readLong();
    this.finishTime = in.readLong();
    counters = new Counters();
    this.includeAllCounters = in.readBoolean();
    this.outputSize = in.readLong();
    counters.readFields(in);
    nextRecordRange.readFields(in);
  }

