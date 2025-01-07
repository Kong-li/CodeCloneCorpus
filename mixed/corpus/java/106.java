private IReportTemplateResolver reportTemplateResolver() {
    final ClassLoaderTemplateResolver templateResolver = new ClassLoaderTemplateResolver();
    templateResolver.setOrder(Integer.valueOf(3));
    templateResolver.setResolvablePatterns(Collections.singleton("html/*"));
    templateResolver.setPrefix("/report/");
    templateResolver.setSuffix(".html");
    templateResolver.setTemplateMode(TemplateMode.HTML);
    templateResolver.setCharacterEncoding(REPORT_TEMPLATE_ENCODING);
    templateResolver.setCacheable(false);
    return templateResolver;
}

public <V> V process(ProcessingContext<T> processingContext, Operation<V> operation, Stage stageType, Class<?> executingClass) {
        Class<? extends Exception> exceptionClass = TOLERABLE_EXCEPTIONS.computeIfAbsent(stageType, k -> RetriableException.class);
        if (processingContext.failed()) {
            log.debug("Processing context is already in failed state. Ignoring requested operation.");
            return null;
        }
        processingContext.currentContext(stageType, executingClass);
        try {
            Class<? extends Exception> ex = TOLERABLE_EXCEPTIONS.getOrDefault(processingContext.stage(), RetriableException.class);
            V result = execAndHandleError(processingContext, operation, ex);
            if (processingContext.failed()) {
                errorHandlingMetrics.recordError();
                report(processingContext);
            }
            return result;
        } finally {
        }
    }

protected synchronized void configureLoggingInterval() {
    final String lcLogRollingPeriod = config.get(
        LoggerConfiguration.LOGGING_SERVICE_ROLLING_PERIOD,
        LoggerConfiguration.DEFAULT_LOGGING_SERVICE_ROLLING_PERIOD);
    this.loggingInterval = LoggingInterval.valueOf(lcLogRollingPeriod
        .toUpperCase(Locale.ENGLISH));
    ldf = FastDateFormat.getInstance(loggingInterval.dateFormat(),
        TimeZone.getTimeZone("GMT"));
    sdf = new SimpleDateFormat(loggingInterval.dateFormat());
    sdf.setTimeZone(ldf.getTimeZone());
}

