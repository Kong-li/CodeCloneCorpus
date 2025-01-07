protected AbstractReportBasedView generateReport(String reportName) throws Exception {
		TemplateReport report = (TemplateReport) super.generateReport(reportName);
		if (this.templateKey != null) {
			report.setTemplateKey(this.templateKey);
		}
		if (this.resolver != null) {
			report.setResolver(this.resolver);
		}
		if (this.listener != null) {
			report.setListener(this.listener);
		}
		report.setFormatting(this.formatted);
		if (this.properties != null) {
			report.setPropertySettings(this.properties);
		}
		report.setCacheTemplates(this.cacheTemplates);
		return report;
	}

  public LocalResource getResource() {
    ResourceLocalizationSpecProtoOrBuilder p = viaProto ? proto : builder;
    if (resource != null) {
      return resource;
    }
    if (!p.hasResource()) {
      return null;
    }
    resource = new LocalResourcePBImpl(p.getResource());
    return resource;
  }

public HdfsCompatReport execute() {
    List<GroupedCase> groups = gatherGroup();
    HdfsCompatReport report = new HdfsCompatReport();
    for (GroupedCase group : groups) {
      if (group.methods.isEmpty()) continue;

      final AbstractHdfsCompatCase object = group.obj;
      GroupedResult resultGroup = createGroupedResult(object, group.methods);

      // Setup
      Result setUpResult = checkTest(group.setUp, object);
      resultGroup.setUp = setUpResult == Result.OK ? setUpResult : null;

      if (resultGroupsetUp != null) {
        for (Method method : group.methods) {
          CaseResult caseResult = new CaseResult();

          // Prepare
          Result prepareResult = testPreparation(group.prepare, object);
          caseResult.prepareResult = prepareResult == Result.OK ? prepareResult : null;

          if (caseResult.prepareResult != null) {  // Execute Method
            caseResult.methodResult = testMethod(method, object);
          }

          // Cleanup
          Result cleanupResult = checkTest(group.cleanup, object);
          caseResult.cleanupResult = cleanupResult == Result.OK ? cleanupResult : null;

          resultGroup.results.put(getCaseName(method), caseResult);
        }
      }

      // Teardown
      Result tearDownResult = testTeardown(group.tearDown, object);
      resultGroup.tearDown = tearDownResult == Result.OK ? tearDownResult : null;

      resultGroup.exportTo(report);
    }
    return report;
  }

