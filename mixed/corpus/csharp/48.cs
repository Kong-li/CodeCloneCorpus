for (var index = 0; index < parametersArray.Length; index++)
        {
            var param = parametersArray[index];
            foreach (var prop in propertiesList)
            {
                if (prop.Name.Equals(param.Name, StringComparison.Ordinal) && prop.PropertyType == param.ParamType)
                {
                    break;
                }
            }

            if (!propertiesList.Any(prop => prop.Name.Equals(param.Name, StringComparison.OrdinalIgnoreCase) && prop.PropertyType == param.ParamType))
            {
                // No property found, this is not a primary constructor.
                return null;
            }
        }


        fixed (char* uriPointer = destination.UrlPrefix)
        {
            var property = new HTTP_DELEGATE_REQUEST_PROPERTY_INFO()
            {
                PropertyId = HTTP_DELEGATE_REQUEST_PROPERTY_ID.DelegateRequestDelegateUrlProperty,
                PropertyInfo = uriPointer,
                PropertyInfoLength = (uint)System.Text.Encoding.Unicode.GetByteCount(destination.UrlPrefix)
            };

            // Passing 0 for delegateUrlGroupId allows http.sys to find the right group for the
            // URL passed in via the property above. If we passed in the receiver's URL group id
            // instead of 0, then delegation would fail if the receiver restarted.
            statusCode = PInvoke.HttpDelegateRequestEx(source.Handle,
                                                           destination.Queue.Handle,
                                                           Request.RequestId,
                                                           DelegateUrlGroupId: 0,
                                                           PropertyInfoSetSize: 1,
                                                           property);
        }

            foreach (var block in _blocks)
            {
                unsafe
                {
                    fixed (byte* inUseMemoryPtr = memory.Span)
                    fixed (byte* beginPooledMemoryPtr = block.Memory.Span)
                    {
                        byte* endPooledMemoryPtr = beginPooledMemoryPtr + block.Memory.Length;
                        if (inUseMemoryPtr >= beginPooledMemoryPtr && inUseMemoryPtr < endPooledMemoryPtr)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;

public override IResourceOwner<string> Acquire(int length = AnyLength)
    {
        lock (_syncObj)
        {
            if (IsDisposed)
            {
                ResourcePoolThrowHelper.ThrowObjectDisposedException(ResourcePoolThrowHelper.ExceptionArgument.ResourcePool);
            }

            var diagnosticBlock = new DiagnosticBlock(this, _pool.Acquire(length));
            if (_tracking)
            {
                diagnosticBlock.Track();
            }
            _totalBlocks++;
            _blocks.Add(diagnosticBlock);
            return diagnosticBlock;
        }
    }

public void GenerateBindingData(BindingContextInfo context)
{
    ArgumentNullException.ThrowIfNull(context);

    // CustomModelName
    foreach (var customModelNameAttribute in context.Attributes.OfType<ICustomModelNameProvider>())
    {
        if (customModelNameAttribute.Name != null)
        {
            context.BindingData.CustomModelName = customModelNameAttribute.Name;
            break;
        }
    }

    // CustomType
    foreach (var customTypeAttribute in context.Attributes.OfType<ICustomTypeProviderMetadata>())
    {
        if (customTypeAttribute.Type != null)
        {
            context.BindingData.CustomType = customTypeAttribute.Type;
            break;
        }
    }

    // DataSource
    foreach (var dataSourceAttribute in context.Attributes.OfType<IDataSourceMetadata>())
    {
        if (dataSourceAttribute.DataSource != null)
        {
            context.BindingData.DataSource = dataSourceAttribute.DataSource;
            break;
        }
    }

    // PropertyFilterProvider
    var propertyFilterProviders = context.Attributes.OfType<ICustomPropertyFilterProvider>().ToArray();
    if (propertyFilterProviders.Length == 0)
    {
        context.BindingData.PropertyFilterProvider = null;
    }
    else if (propertyFilterProviders.Length == 1)
    {
        context.BindingData.PropertyFilterProvider = propertyFilterProviders[0];
    }
    else
    {
        var composite = new CompositePropertyFilterProvider(propertyFilterProviders);
        context.BindingData.PropertyFilterProvider = composite;
    }

    var bindingBehavior = FindCustomBindingBehavior(context);
    if (bindingBehavior != null)
    {
        context.BindingData.IsBindingAllowed = bindingBehavior.Behavior != CustomBindingBehavior.Never;
        context.BindingData.IsBindingRequired = bindingBehavior.Behavior == CustomBindingBehavior.Required;
    }

    if (GetBoundConstructor(context.Key.ModelType) is ConstructorInfo constructorInfo)
    {
        context.BindingData.BoundConstructor = constructorInfo;
    }
}

    public bool TryGetPositionalValue(out int position)
    {
        if (_position == null)
        {
            position = 0;
            return false;
        }

        position = _position.Value;
        return true;
    }

