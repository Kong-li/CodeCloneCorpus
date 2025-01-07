            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                if (reader.ValueTextEquals(ProtocolPropertyNameBytes.EncodedUtf8Bytes))
                {
                    protocol = reader.ReadAsString(ProtocolPropertyName);
                }
                else if (reader.ValueTextEquals(ProtocolVersionPropertyNameBytes.EncodedUtf8Bytes))
                {
                    protocolVersion = reader.ReadAsInt32(ProtocolVersionPropertyName);
                }
                else
                {
                    reader.Skip();
                }
            }
            else if (reader.TokenType == JsonTokenType.EndObject)

static JsonPartialUpdateInfo LocateSharedJsonPartialUpdateInfo(
            JsonPartialUpdateInfo primary,
            JsonPartialUpdateInfo secondary)
        {
            var outcome = new JsonPartialUpdateInfo();
            for (int j = 0; j < Math.Min(primary.Path.Count, secondary.Path.Count); j++)
            {
                if (primary.Path[j].PropertyName == secondary.Path[j].PropertyName &&
                    primary.Path[j].Ordinal == secondary.Path[j].Ordinal)
                {
                    outcome.Path.Add(primary.Path[j]);
                    continue;
                }

                var sharedEntry = new JsonPartialUpdatePathEntry(
                    primary.Path[j].PropertyName,
                    null,
                    primary.Path[j].ParentEntry,
                    primary.Path[j].Navigation);

                outcome.Path.Add(sharedEntry);
            }

            Debug.Assert(outcome.Path.Count > 0, "Shared path should always include at least the root node.");

            return outcome;
        }

void ProcessNonXml(ITargetBase targetBase, IRecordMapping recordMapping)
            {
                foreach (var fieldMapping in recordMapping.FieldMappings)
                {
                    ProcessField(fieldMapping);
                }

                foreach (var complexProperty in targetBase.GetComplexAttributes())
                {
                    var complexRecordMapping = GetMapping(complexProperty.ComplexType);
                    if (complexRecordMapping != null)
                    {
                        ProcessNonXml(complexProperty.ComplexType, complexRecordMapping);
                    }
                }
            }

