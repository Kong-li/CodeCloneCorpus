    def self.watchdog?
      wd_usec = ENV["WATCHDOG_USEC"]
      wd_pid = ENV["WATCHDOG_PID"]

      return false if !wd_usec

      begin
        wd_usec = Integer(wd_usec)
      rescue
        return false
      end

      def to_sentence(array, options = {})
        options.assert_valid_keys(:words_connector, :two_words_connector, :last_word_connector, :locale)

        default_connectors = {
          words_connector: ", ",
          two_words_connector: " and ",
          last_word_connector: ", and "
        }
        if defined?(I18n)
          i18n_connectors = I18n.translate(:'support.array', locale: options[:locale], default: {})
          default_connectors.merge!(i18n_connectors)
        end

          def build_column_serializer(attr_name, coder, type, yaml = nil)
            # When ::JSON is used, force it to go through the Active Support JSON encoder
            # to ensure special objects (e.g. Active Record models) are dumped correctly
            # using the #as_json hook.
            coder = Coders::JSON if coder == ::JSON

            if coder == ::YAML || coder == Coders::YAMLColumn
              Coders::YAMLColumn.new(attr_name, type, **(yaml || {}))
            elsif coder.respond_to?(:new) && !coder.respond_to?(:load)
              coder.new(attr_name, type)
            elsif type && type != Object
              Coders::ColumnSerializer.new(attr_name, coder, type)
            else
              coder
            end

