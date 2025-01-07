private ushort GetSectionsHelper(ref short sectionIndex, ref ushort sectionOffset, byte[] contentBuffer, int startPosition, int length, long offsetAdjustment, MyHttpRequest* request)
{
   ushort sectionsRead = 0;

    if (request->SectionCount > 0 && sectionIndex < request->SectionCount && sectionIndex != -1)
    {
        var pSection = (MySection*)(offsetAdjustment + (byte*)&request->sections[sectionIndex]);

        fixed (byte* pContentBuffer = contentBuffer)
        {
            var pTarget = &pContentBuffer[startPosition];

            while (sectionIndex < request->SectionCount && sectionsRead < length)
            {
                if (sectionOffset >= pSection->Anonymous.FromMemory.BufferLength)
                {
                    sectionOffset = 0;
                    sectionIndex++;
                    pSection++;
                }
                else
                {
                    var pSource = (byte*)pSection->Anonymous.FromMemory.pBuffer + sectionOffset + offsetAdjustment;

                    var bytesToCopy = pSection->Anonymous.FromMemory.BufferLength - (ushort)sectionOffset;
                    if (bytesToCopy > (ushort)length)
                    {
                        bytesToCopy = (ushort)length;
                    }
                    for (ushort i = 0; i < bytesToCopy; i++)
                    {
                        *(pTarget++) = *(pSource++);
                    }
                    sectionsRead += bytesToCopy;
                    sectionOffset += bytesToCopy;
                }
            }
        }
    }
    // we're finished.
    if (sectionIndex == request->SectionCount)
    {
        sectionIndex = -1;
    }
    return sectionsRead;
}

protected override async Task OnTestModuleCompletedAsync()
    {
        // Dispose resources
        foreach (var disposable in Resources.OfType<IDisposable>())
        {
            Aggregator.Run(disposable.Dispose);
        }

        foreach (var disposable in Resources.OfType<IAsyncLifetime>())
        {
            await Aggregator.RunAsync(disposable.DisposeAsync).ConfigureAwait(false);
        }

        await base.OnTestModuleCompletedAsync().ConfigureAwait(false);
    }


    private TreeRouter GetTreeRouter()
    {
        var actions = _actionDescriptorCollectionProvider.ActionDescriptors;

        // This is a safe-race. We'll never set router back to null after initializing
        // it on startup.
        if (_router == null || _router.Version != actions.Version)
        {
            var builder = _services.GetRequiredService<TreeRouteBuilder>();
            AddEntries(builder, actions);
            _router = builder.Build(actions.Version);
        }

        return _router;
    }

internal static BadHttpRequestException GenerateBadRequestException(RequestReasonCode rejectionCode)
{
    BadHttpRequestException ex;
    switch (rejectionCode)
    {
        case RequestReasonCode.InvalidRequestHeadersNoCRLF:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidRequestHeadersNoCRLF, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.InvalidRequestLine:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidRequestLine, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.MalformedRequestInvalidHeaders:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MalformedRequestInvalidHeaders, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.MultipleContentLengths:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MultipleContentLengths, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.UnexpectedEndOfRequestContent:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_UnexpectedEndOfRequestContent, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.BadChunkSuffix:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_BadChunkSuffix, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.BadChunkSizeData:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_BadChunkSizeData, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.ChunkedRequestIncomplete:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_ChunkedRequestIncomplete, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.InvalidCharactersInHeaderName:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidCharactersInHeaderName, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.RequestLineTooLong:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_RequestLineTooLong, StatusCodes.Status414UriTooLong, rejectionCode);
            break;
        case RequestReasonCode.HeadersExceedMaxTotalSize:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_HeadersExceedMaxTotalSize, StatusCodes.Status431RequestHeaderFieldsTooLarge, rejectionCode);
            break;
        case RequestReasonCode.TooManyHeaders:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_TooManyHeaders, StatusCodes.Status431RequestHeaderFieldsTooLarge, rejectionCode);
            break;
        case RequestReasonCode.RequestHeadersTimeout:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_RequestHeadersTimeout, StatusCodes.Status408RequestTimeout, rejectionCode);
            break;
        case RequestReasonCode.RequestBodyTimeout:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_RequestBodyTimeout, StatusCodes.Status408RequestTimeout, rejectionCode);
            break;
        case RequestReasonCode.OptionsMethodRequired:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MethodNotAllowed, StatusCodes.Status405MethodNotAllowed, rejectionCode, HttpMethod.Options);
            break;
        case RequestReasonCode.ConnectMethodRequired:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MethodNotAllowed, StatusCodes.Status405MethodNotAllowed, rejectionCode, HttpMethod.Connect);
            break;
        case RequestReasonCode.MissingHostHeader:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MissingHostHeader, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.MultipleHostHeaders:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MultipleHostHeaders, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.InvalidHostHeader:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidHostHeader, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        default:
            ex = new BadHttpRequestException(CoreStrings.BadRequest, StatusCodes.Status400BadRequest, rejectionCode);
            break;
    }
    return ex;
#pragma warning restore CS0618 // Type or member is obsolete
}

private uint CalculateChunksHelper(ref int chunkIndex, ref uint chunkOffset, byte[] bufferArray, int startOffset, int totalSize, long adjustment, HTTP_REQUEST_V1* requestPointer)
{
    uint totalRead = 0;

    if (requestPointer->EntityChunkCount > 0 && chunkIndex < requestPointer->EntityChunkCount && chunkIndex != -1)
    {
        var currentChunkData = (HTTP_DATA_CHUNK*)(adjustment + (byte*)&requestPointer->pEntityChunks[chunkIndex]);

        fixed (byte* bufferPointer = bufferArray)
        {
            byte* targetPosition = &bufferPointer[startOffset];

            while (chunkIndex < requestPointer->EntityChunkCount && totalRead < totalSize)
            {
                if (chunkOffset >= currentChunkData->Anonymous.FromMemory.BufferLength)
                {
                    chunkOffset = 0;
                    chunkIndex++;
                    currentChunkData++;
                }
                else
                {
                    byte* sourcePosition = (byte*)currentChunkData->Anonymous.FromMemory.pBuffer + chunkOffset + adjustment;

                    uint bytesToCopy = currentChunkData->Anonymous.FromMemory.BufferLength - chunkOffset;
                    if (bytesToCopy > totalSize)
                    {
                        bytesToCopy = (uint)totalSize;
                    }
                    for (uint i = 0; i < bytesToCopy; i++)
                    {
                        *(targetPosition++) = *(sourcePosition++);
                    }
                    totalRead += bytesToCopy;
                    chunkOffset += bytesToCopy;
                }
            }
        }
    }

    if (chunkIndex == requestPointer->EntityChunkCount)
    {
        chunkIndex = -1;
    }
    return totalRead;
}

