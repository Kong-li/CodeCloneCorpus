    def becomes(klass)
      became = klass.allocate

      became.send(:initialize) do |becoming|
        @attributes.reverse_merge!(becoming.instance_variable_get(:@attributes))
        becoming.instance_variable_set(:@attributes, @attributes)
        becoming.instance_variable_set(:@mutations_from_database, @mutations_from_database ||= nil)
        becoming.instance_variable_set(:@new_record, new_record?)
        becoming.instance_variable_set(:@previously_new_record, previously_new_record?)
        becoming.instance_variable_set(:@destroyed, destroyed?)
        becoming.errors.copy!(errors)
      end

      def add_source(source, options = {}, &block)
        log :source, source

        in_root do
          if block
            append_file_with_newline "Gemfile", "\nsource #{quote(source)} do", force: true
            with_indentation(&block)
            append_file_with_newline "Gemfile", "end", force: true
          else
            prepend_file "Gemfile", "source #{quote(source)}\n", verbose: false
          end

      def _create_record(attribute_names = self.attribute_names)
        attribute_names = attributes_for_create(attribute_names)

        self.class.with_connection do |connection|
          returning_columns = self.class._returning_columns_for_insert(connection)

          returning_values = self.class._insert_record(
            connection,
            attributes_with_values(attribute_names),
            returning_columns
          )

          returning_columns.zip(returning_values).each do |column, value|
            _write_attribute(column, type_for_attribute(column).deserialize(value)) if !_read_attribute(column)
          end if returning_values
        end

