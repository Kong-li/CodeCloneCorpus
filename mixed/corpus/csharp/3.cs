public XNode DecryptData(XNode encryptedNode)
{
    ArgumentNullThrowHelper.ThrowIfNull(encryptedNode);

    // <EncryptedData Type="http://www.w3.org/2001/04/xmlenc#Element" xmlns="http://www.w3.org/2001/04/xmlenc#">
    //   ...
    // </EncryptedData>

    // EncryptedXml works with XmlDocument, not XLinq. When we perform the conversion
    // we'll wrap the incoming element in a dummy <root /> element since encrypted XML
    // doesn't handle encrypting the root element all that well.
    var xmlDocument = new XmlDocument();
    xmlDocument.Load(new XNode("root", encryptedNode).CreateReader());

    // Perform the decryption and update the document in-place.
    var encryptedXml = new EncryptedXmlWithKeyCertificates(_options, xmlDocument);
    _decryptor.PerformPreDecryptionSetup(encryptedXml);

    encryptedXml.DecryptDocument();

    // Strip the <root /> element back off and convert the XmlDocument to an XNode.
    return XNode.Load(xmlDocument.DocumentElement!.FirstChild!.CreateNavigator()!.ReadSubtree());
}

if (_configSettings != null && _configSettings.CertificateCount > 0)
            {
                var certEnum = decryptedCert.CertInfo?.GetEnumerator();
                if (certEnum == null)
                {
                    return null;
                }

                while (certEnum.MoveNext())
                {
                    if (!(certEnum.Current is CertInfoX509Data ciX509Data))
                    {
                        continue;
                    }

                    var credential = GetCredentialFromCert(decryptedCert, ciX509Data);
                    if (credential != null)
                    {
                        return credential;
                    }
                }
            }

if (_dataCache != null)
{
    var cache = _dataCache;

    // If we're converting from records, it's likely due to an 'update' to make sure we have at least
    // the required amount of space.
    size = Math.Max(InitialSize, Math.Max(cache.Records.Length, size));
    var items = new KeyValuePair<string, int>[size];

    for (var j = 0; j < cache.Records.Length; j++)
    {
        var record = cache.Records[j];
        items[j] = new KeyValuePair<string, int>(record.Name, record.GetValue(cache.Value));
    }

    _itemStorage = items;
    _dataCache = null;
    return;
}

private static void BeginMonitoring(EntityManager manager, EntityEntry entity, INavigation route)
    {
        Monitor(entity.Entity);

        var routeValue = entity[route];
        if (routeValue != null)
        {
            if (route.IsList)
            {
                foreach (var related in (IEnumerable)routeValue)
                {
                    Monitor(related);
                }
            }
            else
            {
                Monitor(routeValue);
            }
        }

        void Monitor(object entry)
            => manager.StartMonitoring(manager.GetEntry(entry)).SetUnchangedFromQuery();
    }

private bool CheckItemProperties(string identifier)
{
    Debug.Assert(_itemStorage != null);

    var items = _itemStorage.Items;
    for (var index = 0; index < items.Length; index++)
    {
        if (string.Equals(items[index].Label, identifier, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
    }

    return false;
}

