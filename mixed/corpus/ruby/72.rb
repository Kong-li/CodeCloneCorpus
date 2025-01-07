      def initialize(config, *)
        config = config.dup

        # Trilogy ignores `socket` if `host is set. We want the opposite to allow
        # configuring UNIX domain sockets via `DATABASE_URL`.
        config.delete(:host) if config[:socket]

        # Set FOUND_ROWS capability on the connection so UPDATE queries returns number of rows
        # matched rather than number of rows updated.
        config[:found_rows] = true

        if config[:prepared_statements]
          raise ArgumentError, "Trilogy currently doesn't support prepared statements. Remove `prepared_statements: true` from your database configuration."
        end

