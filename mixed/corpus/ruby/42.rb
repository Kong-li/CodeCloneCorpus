        def prepare_command_options
          args = {
            host:      "--host",
            port:      "--port",
            socket:    "--socket",
            username:  "--user",
            password:  "--password",
            encoding:  "--default-character-set",
            sslca:     "--ssl-ca",
            sslcert:   "--ssl-cert",
            sslcapath: "--ssl-capath",
            sslcipher: "--ssl-cipher",
            sslkey:    "--ssl-key",
            ssl_mode:  "--ssl-mode"
          }.filter_map { |opt, arg| "#{arg}=#{configuration_hash[opt]}" if configuration_hash[opt] }

          args
        end

      def wrap_inline_attachments(message)
        # If we have both types of attachment, wrap all the inline attachments
        # in multipart/related, but not the actual attachments
        if message.attachments.detect(&:inline?) && message.attachments.detect { |a| !a.inline? }
          related = Mail::Part.new
          related.content_type = "multipart/related"
          mixed = [ related ]

          message.parts.each do |p|
            if p.attachment? && !p.inline?
              mixed << p
            else
              related.add_part(p)
            end

        def delete_entry(key, **options)
          if File.exist?(key)
            begin
              File.delete(key)
              delete_empty_directories(File.dirname(key))
              true
            rescue
              # Just in case the error was caused by another process deleting the file first.
              raise if File.exist?(key)
              false
            end

    def overlap?(other)
      raise TypeError unless other.is_a? Range

      self_begin = self.begin
      other_end = other.end
      other_excl = other.exclude_end?

      return false if _empty_range?(self_begin, other_end, other_excl)

      other_begin = other.begin
      self_end = self.end
      self_excl = self.exclude_end?

      return false if _empty_range?(other_begin, self_end, self_excl)
      return true if self_begin == other_begin

      return false if _empty_range?(self_begin, self_end, self_excl)
      return false if _empty_range?(other_begin, other_end, other_excl)

      true
    end

      def method_missing(name, ...)
        # We can't know whether some method was defined or not because
        # multiple thread might be concurrently be in this code path.
        # So the first one would define the methods and the others would
        # appear to already have them.
        self.class.define_attribute_methods

        # So in all cases we must behave as if the method was just defined.
        method = begin
          self.class.public_instance_method(name)
        rescue NameError
          nil
        end

