        def raw_config
          if uri.opaque
            query_hash.merge(
              adapter: @adapter,
              database: uri.opaque
            )
          else
            query_hash.reverse_merge(
              adapter: @adapter,
              username: uri.user,
              password: uri.password,
              port: uri.port,
              database: database_from_path,
              host: uri.hostname
            )
          end

      def compressed(compress_threshold)
        return self if compressed?

        case @value
        when nil, true, false, Numeric
          uncompressed_size = 0
        when String
          uncompressed_size = @value.bytesize
        else
          serialized = Marshal.dump(@value)
          uncompressed_size = serialized.bytesize
        end

        def database_from_path
          if @adapter == "sqlite3"
            # 'sqlite3:/foo' is absolute, because that makes sense. The
            # corresponding relative version, 'sqlite3:foo', is handled
            # elsewhere, as an "opaque".

            uri.path
          else
            # Only SQLite uses a filename as the "database" name; for
            # anything else, a leading slash would be silly.

            uri.path.delete_prefix("/")
          end

