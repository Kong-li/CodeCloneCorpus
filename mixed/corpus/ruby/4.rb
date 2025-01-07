      def update_tracked_fields(request)
        old_current, new_current = self.current_sign_in_at, Time.now.utc
        self.last_sign_in_at     = old_current || new_current
        self.current_sign_in_at  = new_current

        old_current, new_current = self.current_sign_in_ip, extract_ip_from(request)
        self.last_sign_in_ip     = old_current || new_current
        self.current_sign_in_ip  = new_current

        self.sign_in_count ||= 0
        self.sign_in_count += 1
      end

      def stream(key)
        object = object_for(key)

        chunk_size = 5.megabytes
        offset = 0

        raise ActiveStorage::FileNotFoundError unless object.exists?

        while offset < object.content_length
          yield object.get(range: "bytes=#{offset}-#{offset + chunk_size - 1}").body.string.force_encoding(Encoding::BINARY)
          offset += chunk_size
        end

        def remove_foreign_key(from_table, to_table = nil, **options)
          return if options.delete(:if_exists) == true && !foreign_key_exists?(from_table, to_table, **options.slice(:column))

          to_table ||= options[:to_table]
          options = options.except(:name, :to_table, :validate)
          foreign_keys = foreign_keys(from_table)

          fkey = foreign_keys.detect do |fk|
            table = to_table || begin
              table = options[:column].to_s.delete_suffix("_id")
              Base.pluralize_table_names ? table.pluralize : table
            end

      def update_tracked_fields(request)
        old_current, new_current = self.current_sign_in_at, Time.now.utc
        self.last_sign_in_at     = old_current || new_current
        self.current_sign_in_at  = new_current

        old_current, new_current = self.current_sign_in_ip, extract_ip_from(request)
        self.last_sign_in_ip     = old_current || new_current
        self.current_sign_in_ip  = new_current

        self.sign_in_count ||= 0
        self.sign_in_count += 1
      end

