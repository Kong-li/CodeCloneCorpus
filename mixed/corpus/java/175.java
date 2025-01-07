public static Mapping doMap(Mapper mapper, File file, Source source, boolean autoRelease) {
		try {
			return mapper.map( file, source );
		}
		catch ( Exception e ) {
			throw new InvalidDataException( source, e );
		}
		finally {
			if ( autoRelease ) {
				try {
					file.delete();
				}
				catch ( IOException ignore ) {
					log.trace( "Failed to delete file" );
				}
			}
		}
	}

  public static void skipFully(DataInput in, int len) throws IOException {
    int total = 0;
    int cur = 0;

    while ((total<len) && ((cur = in.skipBytes(len-total)) > 0)) {
        total += cur;
    }

    if (total<len) {
      throw new IOException("Not able to skip " + len + " bytes, possibly " +
                            "due to end of input.");
    }
  }

  protected void dumpStateInternal(StringBuilder sb) {
    sb.append("{Name: " + getName() +
        ", Weight: " + weights +
        ", Policy: " + policy.getName() +
        ", FairShare: " + getFairShare() +
        ", SteadyFairShare: " + getSteadyFairShare() +
        ", MaxShare: " + getMaxShare() +
        ", MinShare: " + minShare +
        ", ResourceUsage: " + getResourceUsage() +
        ", Demand: " + getDemand() +
        ", MaxAMShare: " + maxAMShare +
        ", Runnable: " + getNumRunnableApps() +
        "}");

    for(FSQueue child : getChildQueues()) {
      sb.append(", ");
      child.dumpStateInternal(sb);
    }
  }

