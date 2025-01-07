public async Task ExecuteActionAsync(Func<object> action)
    {
        var completionSource = new PhotinoSynchronizationTaskCompletionSource<Func<object>, object>(action);
        bool isExecuteSync = CanExecuteSynchronously((state) =>
        {
            var completion = (PhotinoSynchronizationTaskCompletionSource<Func<object>, object>)state;
            try
            {
                completion.Callback();
                completion.SetResult(null);
            }
            catch (OperationCanceledException)
            {
                completion.SetCanceled();
            }
            catch (Exception ex)
            {
                completion.SetException(ex);
            }
        }, completionSource);

        if (!isExecuteSync)
        {
            await ExecuteSynchronouslyIfPossible((state) =>
            {
                var completion = (PhotinoSynchronizationTaskCompletionSource<Func<object>, object>)state;
                try
                {
                    completion.Callback();
                    completion.SetResult(null);
                }
                catch (OperationCanceledException)
                {
                    completion.SetCanceled();
                }
                catch (Exception exception)
                {
                    completion.SetException(exception);
                }
            }, completionSource);
        }

        return completionSource.Task;
    }

    public bool TryDisable(HttpLoggingFields fields)
    {
        if (IsAnyEnabled(fields))
        {
            Disable(fields);
            return true;
        }

        return false;
    }

public static IEnumerable<StoreObjectIdentifier> FetchLinkedStoreObjects(
        this IReadOnlyProperty entityProperty,
        StoreObjectType objectType)
    {
        var owningType = entityProperty.OwningType;
        var owningStoreObject = StoreObjectIdentifier.Create(owningType, objectType);
        if (owningStoreObject != null
            && entityProperty.GetColumnName(owningStoreObject.Value) != null)
        {
            yield return owningStoreObject.Value;
        }

        if (objectType is StoreObjectType.Procedure or StoreObjectType.Query)
        {
            yield break;
        }

        foreach (var section in owningType.GetMappingSections(objectType))
        {
            if (entityProperty.GetColumnName(section.StoreObject) != null)
            {
                yield return section.StoreObject;
            }
        }

        if (owningType.GetMappingStrategy() == RelationalAnnotationNames.TphMappingStrategy)
        {
            yield break;
        }

        if (owningType is IReadOnlyEntityType entityTypeName)
        {
            foreach (var derivedType in entityTypeName.GetDerivedTypes())
            {
                var derivedStoreObject = StoreObjectIdentifier.Create(derivedType, objectType);
                if (derivedStoreObject != null
                    && entityProperty.GetColumnName(derivedStoreObject.Value) != null)
                {
                    yield return derivedStoreObject.Value;
                }
            }
        }
    }

public static DeserializedHubEvent ReadDeserializedHubEvent(ref MessagePackerReader reader)
    {
        var size = reader.ReadMapHeader();
        var events = new DeserializedMessage[size];
        for (var index = 0; index < size; index++)
        {
            var eventProtocol = reader.ReadString()!;
            var serializedData = reader.ReadBytes()?.ToArray() ?? Array.Empty<byte>();

            events[index] = new DeserializedMessage(eventProtocol, serializedData);
        }

        return new DeserializedHubEvent(events);
    }

public CbcAuthenticatedEncryptor(Secret derivationKey, BCryptAlgorithmHandle algoForSymmetric, uint keySizeOfSymmetric, BCryptAlgorithmHandle algoForHMAC, IBCryptGenRandom? randomGenerator = null)
{
    _randomGen = randomGenerator ?? BCryptGenRandomImpl.Instance;
    _ctrHmacProvider = SP800_108_CTR_HMACSHA512Util.CreateProvider(derivationKey);
    _symmetricAlgHandle = algoForSymmetric;
    _symBlockLen = _symmetricAlgHandle.GetCipherBlockLength();
    _symKeyLen = keySizeOfSymmetric;
    _hmacAlgHandle = algoForHMAC;
    _digestLen = _hmacAlgHandle.GetHashDigestLength();
    _hmacSubkeyLen = _digestLen; // for simplicity we'll generate HMAC subkeys with a length equal to the digest length

    // Argument checking on the algorithms and lengths passed in to us
    AlgorithmAssert.IsAllowableSymmetricAlgorithmBlockSize(checked(_symBlockLen * 8));
    AlgorithmAssert.IsAllowableSymmetricAlgorithmKeySize(checked(_symKeyLen * 8));
    AlgorithmAssert.IsAllowableValidationAlgorithmDigestSize(checked(_digestLen * 8));

    _contextHeader = CreateContextHeader();
}

