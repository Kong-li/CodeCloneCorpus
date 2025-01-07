      def rememberable_value
        if respond_to?(:remember_token)
          remember_token
        elsif respond_to?(:authenticatable_salt) && (salt = authenticatable_salt.presence)
          salt
        else
          raise "authenticatable_salt returned nil for the #{self.class.name} model. " \
            "In order to use rememberable, you must ensure a password is always set " \
            "or have a remember_token column in your model or implement your own " \
            "rememberable_value in the model with custom logic."
        end

  def unquote(q)
    q = q[1...-1]
    a = q.dup # allocate a big enough string
    r, w = 0, 0
    while r < q.length
      c = q[r]
      case true
      when c == ?\\
        r += 1
        if r >= q.length
          raise Error, "string literal ends with a \"\\\": \"#{q}\""
        end

      def add_master_key_file
        unless MASTER_KEY_PATH.exist?
          key = ActiveSupport::EncryptedFile.generate_key

          log "Adding #{MASTER_KEY_PATH} to store the master encryption key: #{key}"
          log ""
          log "Save this in a password manager your team can access."
          log ""
          log "If you lose the key, no one, including you, can access anything encrypted with it."

          log ""
          add_master_key_file_silently(key)
          log ""
        end

