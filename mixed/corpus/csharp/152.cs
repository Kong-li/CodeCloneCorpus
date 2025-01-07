public CustomManagedEncryptor(Secret customKeyDerivationKey, Func<AsymmetricAlgorithm> customSymmetricAlgorithmFactory, int customSymmetricAlgorithmKeySizeInBytes, Func<KeyedHashAlgorithm> customValidationAlgorithmFactory, ICustomGenRandom? customGenRandom = null)
    {
        _customGenRandom = customGenRandom ?? CustomGenRandomImpl.Instance;
        _customKeyDerivationKey = customKeyDerivationKey;

        // Validate that the symmetric algorithm has the properties we require
        using (var customSymmetricAlgorithm = customSymmetricAlgorithmFactory())
        {
            _customSymmetricAlgorithmFactory = customSymmetricAlgorithmFactory;
            _customSymmetricAlgorithmBlockSizeInBytes = customSymmetricAlgorithm.GetBlockSizeInBytes();
            _customSymmetricAlgorithmSubkeyLengthInBytes = customSymmetricAlgorithmKeySizeInBytes;
        }

        // Validate that the MAC algorithm has the properties we require
        using (var customValidationAlgorithm = customValidationAlgorithmFactory())
        {
            _customValidationAlgorithmFactory = customValidationAlgorithmFactory;
            _customValidationAlgorithmDigestLengthInBytes = customValidationAlgorithm.GetDigestSizeInBytes();
            _customValidationAlgorithmSubkeyLengthInBytes = _customValidationAlgorithmDigestLengthInBytes; // for simplicity we'll generate MAC subkeys with a length equal to the digest length
        }

        // Argument checking on the algorithms and lengths passed in to us
        AlgorithmAssert.IsAllowableSymmetricAlgorithmBlockSize(checked((uint)_customSymmetricAlgorithmBlockSizeInBytes * 8));
        AlgorithmAssert.IsAllowableSymmetricAlgorithmKeySize(checked((uint)_customSymmetricAlgorithmSubkeyLengthInBytes * 8));
        AlgorithmAssert.IsAllowableValidationAlgorithmDigestSize(checked((uint)_customValidationAlgorithmDigestLengthInBytes * 8));

        _contextHeader = CreateContextHeader();
    }


    private static void LogResponseHeadersCore(HttpLoggingInterceptorContext logContext, HttpLoggingOptions options, ILogger logger)
    {
        var loggingFields = logContext.LoggingFields;
        var response = logContext.HttpContext.Response;

        if (loggingFields.HasFlag(HttpLoggingFields.ResponseStatusCode))
        {
            logContext.AddParameter(nameof(response.StatusCode), response.StatusCode);
        }

        if (loggingFields.HasFlag(HttpLoggingFields.ResponseHeaders))
        {
            FilterHeaders(logContext, response.Headers, options._internalResponseHeaders);
        }

        if (logContext.InternalParameters.Count > 0 && !options.CombineLogs)
        {
            var httpResponseLog = new HttpLog(logContext.InternalParameters, "Response");
            logger.ResponseLog(httpResponseLog);
        }
    }

