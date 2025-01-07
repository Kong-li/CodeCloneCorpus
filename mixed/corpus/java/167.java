  public int run(String[] args) throws Exception {
    Configuration conf = getConf();
    if (args.length == 0) {
      System.out.println("Usage: pentomino <output> [-depth #] [-height #] [-width #]");
      ToolRunner.printGenericCommandUsage(System.out);
      return 2;
    }
    // check for passed parameters, otherwise use defaults
    int width = conf.getInt(Pentomino.WIDTH, PENT_WIDTH);
    int height = conf.getInt(Pentomino.HEIGHT, PENT_HEIGHT);
    int depth = conf.getInt(Pentomino.DEPTH, PENT_DEPTH);
    for (int i = 0; i < args.length; i++) {
      if (args[i].equalsIgnoreCase("-depth")) {
        depth = Integer.parseInt(args[++i].trim());
      } else if (args[i].equalsIgnoreCase("-height")) {
        height = Integer.parseInt(args[++i].trim());
      } else if (args[i].equalsIgnoreCase("-width") ) {
        width = Integer.parseInt(args[++i].trim());
      }
    }
    // now set the values within conf for M/R tasks to read, this
    // will ensure values are set preventing MAPREDUCE-4678
    conf.setInt(Pentomino.WIDTH, width);
    conf.setInt(Pentomino.HEIGHT, height);
    conf.setInt(Pentomino.DEPTH, depth);
    Class<? extends Pentomino> pentClass = conf.getClass(Pentomino.CLASS,
      OneSidedPentomino.class, Pentomino.class);
    int numMaps = conf.getInt(MRJobConfig.NUM_MAPS, DEFAULT_MAPS);
    Path output = new Path(args[0]);
    Path input = new Path(output + "_input");
    FileSystem fileSys = FileSystem.get(conf);
    try {
      Job job = Job.getInstance(conf);
      FileInputFormat.setInputPaths(job, input);
      FileOutputFormat.setOutputPath(job, output);
      job.setJarByClass(PentMap.class);

      job.setJobName("dancingElephant");
      Pentomino pent = ReflectionUtils.newInstance(pentClass, conf);
      pent.initialize(width, height);
      long inputSize = createInputDirectory(fileSys, input, pent, depth);
      // for forcing the number of maps
      FileInputFormat.setMaxInputSplitSize(job, (inputSize/numMaps));

      // the keys are the prefix strings
      job.setOutputKeyClass(Text.class);
      // the values are puzzle solutions
      job.setOutputValueClass(Text.class);

      job.setMapperClass(PentMap.class);
      job.setReducerClass(Reducer.class);

      job.setNumReduceTasks(1);

      return (job.waitForCompletion(true) ? 0 : 1);
      } finally {
      fileSys.delete(input, true);
    }
  }

public void executeSqlSelections(DomainResultCreationContext creationContext) {
		SqlAstCreationState sqlAstCreationState = creationContext.getSqlAstCreationState();
		SqlExpressionResolver resolver = sqlAstCreationState.getSqlExpressionResolver();

		resolver.resolveSqlSelection(
				this,
				type.getSingleJdbcMapping().getJdbcJavaType(),
				null,
				sqlAstCreationState.getCreationContext().getMappingMetamodel().getTypeConfiguration()
		);
	}

