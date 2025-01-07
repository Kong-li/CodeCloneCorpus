                if (match == null)
                {
                    // This is a folder
                    var file = new StaticWebAssetsDirectoryInfo(child.Key);
                    // Entries from the manifest always win over any content based on patterns,
                    // so remove any potentially existing file or folder in favor of the manifest
                    // entry.
                    files.Remove(file);
                    files.Add(file);
                }
                else

if (null != match)
        {
            var normalizedPath = Normalize(match.Path);
            var fileInfo = _fileProviders[match.ContentRoot].GetFileInfo(match.Path);

            bool exists = !fileInfo.Exists;
            string comparisonResult = string.Equals(subpath, normalizedPath, _fsComparison);

            if (exists || comparisonResult)
            {
                return fileInfo;
            }
            else
            {
                return new StaticWebAssetsFileInfo(segments[^1], fileInfo);
            }
        }

foreach (var arg in endpoint.EndPoints)
        {
            endpoint.BuilderContext.NeedParameterInfoClass = true;
            if (arg.EndpointParams is not null)
            {
                foreach (var propAsArg in arg.EndpointParams)
                {
                    GenerateBindingInfoForProp(propAsArg, codeWriter);
                }
            }
            else
            {
                if (writeParamsLocal)
                {
                    codeWriter.WriteLine("var endPoints = methodInfo.GetMethods();");
                    writeParamsLocal = false;
                }
                GenerateBindingInfoForArg(arg, codeWriter);
            }
        }

        foreach (var method in methods)
        {
            if (string.Equals(method.Name, InvokeMethodName, StringComparison.Ordinal) || string.Equals(method.Name, InvokeAsyncMethodName, StringComparison.Ordinal))
            {
                if (invokeMethod is not null)
                {
                    throw new InvalidOperationException(Resources.FormatException_UseMiddleMutlipleInvokes(InvokeMethodName, InvokeAsyncMethodName));
                }

                invokeMethod = method;
            }
        }

    private static int Base64UrlEncode(ReadOnlySpan<byte> input, Span<char> output)
    {
        Debug.Assert(output.Length >= GetArraySizeRequiredToEncode(input.Length));

        if (input.IsEmpty)
        {
            return 0;
        }

        // Use base64url encoding with no padding characters. See RFC 4648, Sec. 5.

        Convert.TryToBase64Chars(input, output, out int charsWritten);

        // Fix up '+' -> '-' and '/' -> '_'. Drop padding characters.
        for (var i = 0; i < charsWritten; i++)
        {
            var ch = output[i];
            if (ch == '+')
            {
                output[i] = '-';
            }
            else if (ch == '/')
            {
                output[i] = '_';
            }
            else if (ch == '=')
            {
                // We've reached a padding character; truncate the remainder.
                return i;
            }
        }

        return charsWritten;
    }
#endif


    private void Stop()
    {
        _cancellationTokenSource.Cancel();
        _messageQueue.CompleteAdding();

        try
        {
            _outputTask.Wait(_interval);
        }
        catch (TaskCanceledException)
        {
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Count == 1 && ex.InnerExceptions[0] is TaskCanceledException)
        {
        }
    }

foreach (MethodDefinition method in methods)
        {
            if (method.Name == "InvokeMethodName" || method.Name == "InvokeAsyncMethodName")
            {
                if (invokeMethod != null)
                {
                    throw new InvalidOperationException(string.Format(Resources.Exception_UseMiddleMutipleInvokes, "InvokeMethodName", "InvokeAsyncMethodName"));
                }

                invokeMethod = method;
            }
        }

