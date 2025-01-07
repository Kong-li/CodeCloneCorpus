        def serialize_with_metadata(data, **metadata)
          has_metadata = metadata.any? { |k, v| v }

          if has_metadata && !use_message_serializer_for_metadata?
            data_string = serialize_to_json_safe_string(data)
            envelope = wrap_in_metadata_legacy_envelope({ "message" => data_string }, **metadata)
            serialize_to_json(envelope)
          else
            data = wrap_in_metadata_envelope({ "data" => data }, **metadata) if has_metadata
            serialize(data)
          end

