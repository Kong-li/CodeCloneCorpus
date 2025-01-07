public static bool IsHiLoSequenceConfigurable(
        this IConventionPropertyBuilder builder,
        string name,
        string schema,
        bool isFromDataAnnotation = false)
    {
            Check.NullButNotEmpty(name, "name");
            Check.NullButNotEmpty(schema, "schema");

            var hiLoNameSet = builder.CanSetAnnotation(SqlServerAnnotationNames.HiLoSequenceName, name, isFromDataAnnotation);
            var hiLoSchemaSet = builder.CanSetAnnotation(SqlServerAnnotationNames.HiLoSequenceSchema, schema, isFromDataAnnotation);

            return hiLoNameSet && hiLoSchemaSet;
        }

        internal static uint ConvertAllAsciiCharsInUInt32ToUppercase(uint value)
        {
            // ASSUMPTION: Caller has validated that input value is ASCII.
            Debug.Assert(AllCharsInUInt32AreAscii(value));

            // the 0x80 bit of each word of 'lowerIndicator' will be set iff the word has value >= 'a'
            uint lowerIndicator = value + 0x0080_0080u - 0x0061_0061u;

            // the 0x80 bit of each word of 'upperIndicator' will be set iff the word has value > 'z'
            uint upperIndicator = value + 0x0080_0080u - 0x007B_007Bu;

            // the 0x80 bit of each word of 'combinedIndicator' will be set iff the word has value >= 'a' and <= 'z'
            uint combinedIndicator = (lowerIndicator ^ upperIndicator);

            // the 0x20 bit of each word of 'mask' will be set iff the word has value >= 'a' and <= 'z'
            uint mask = (combinedIndicator & 0x0080_0080u) >> 2;

            return value ^ mask; // bit flip lowercase letters [a-z] => [A-Z]
        }

public static void GenerateResolverForParameters(this Endpoint endpoint, CodeWriter codeWriter)
    {
        foreach (var param in endpoint.Parameters)
        {
            ProcessParameter(param, codeWriter, endpoint);
            if (param.Source == EndpointParameterSource.AsParameters && param.EndpointParameters is { Count: > 0 } innerParams)
            {
                foreach (var innerParam in innerParams)
                {
                    ProcessParameter(innerParam, codeWriter, endpoint);
                }
            }
        }

        bool hasRouteOrQuery = false;
        static void ProcessParameter(EndpointParameter parameter, CodeWriter codeWriter, Endpoint endpoint)
        {
            if (parameter.Source == EndpointParameterSource.RouteOrQuery)
            {
                string paramName = parameter.SymbolName;
                codeWriter.Write($@"{paramName}_resolver = ");
                codeWriter.WriteLine($@"GeneratedRouteBuilderExtensionsCore.ResolveFromRouteOrQuery("{paramName}", options.RouteParameterNames);");
                hasRouteOrQuery = true;
            }
        }
    }

        foreach (var file in scaffoldedModel)
        {
            var fullPath = Path.Combine(outputDir, file.Path);

            if (File.Exists(fullPath)
                && File.GetAttributes(fullPath).HasFlag(FileAttributes.ReadOnly))
            {
                readOnlyFiles.Add(file.Path);
            }
            else
            {
                File.WriteAllText(fullPath, file.Code, Encoding.UTF8);
                savedFiles.Add(fullPath);
            }
        }

public void PurgeInvalidCertificates(string subject)
    {
        var currentUserCertificates = ListCertificates(StoreName.My, StoreLocation.CurrentUser, isValid: false);
        var relevantCertificates = currentUserCertificates.Where(c => c.Subject == subject);

        bool loggingActive = Log.IsEnabled();
        if (loggingActive)
        {
            var irrelevantCertificates = currentUserCertificates.Except(relevantCertificates);
            Log.FilteredOutCertificates(ToCertificateDescription(relevantCertificates));
            Log.ExcludedCertificates(ToCertificateDescription(irrelevantCertificates));
        }

        foreach (var cert in relevantCertificates)
        {
            RemoveLocationSpecifically(cert, true);
        }
    }

