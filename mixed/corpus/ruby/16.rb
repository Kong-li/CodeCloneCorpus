        def tag_option(key, value, escape)
          key = ERB::Util.xml_name_escape(key) if escape

          case value
          when Array, Hash
            value = TagHelper.build_tag_values(value) if key.to_s == "class"
            value = escape ? safe_join(value, " ") : value.join(" ")
          when Regexp
            value = escape ? ERB::Util.unwrapped_html_escape(value.source) : value.source
          else
            value = escape ? ERB::Util.unwrapped_html_escape(value) : value.to_s
          end

    def load_async
      with_connection do |c|
        return load if !c.async_enabled?

        unless loaded?
          result = exec_main_query(async: !c.current_transaction.joinable?)

          if result.is_a?(Array)
            @records = result
          else
            @future_result = result
          end

    def validate
      # https://github.com/w3c-validators/w3c_validators/issues/25
      validator = NuValidator.new
      STDOUT.sync = true
      errors_on_guides = {}

      guides_to_validate.each do |f|
        begin
          results = validator.validate_file(f)
        rescue Exception => e
          puts "\nCould not validate #{f} because of #{e}"
          next
        end

def process_arel_attributes(attributes)
        attributes.flat_map { |attr|
          if attr.is_a?(Arel::Predications)
            [attr]
          elsif attr.is_a?(Hash)
            attr.flat_map do |table, columns|
              table_str = table.to_s
              columns_array = Array(columns).map { |column|
                predicate_builder.resolve_arel_attribute(table_str, column)
              }
              columns_array
            end
          else
            []
          end
        }.flatten
      end

      def build_join_buckets
        buckets = Hash.new { |h, k| h[k] = [] }

        unless left_outer_joins_values.empty?
          stashed_left_joins = []
          left_joins = select_named_joins(left_outer_joins_values, stashed_left_joins) do |left_join|
            if left_join.is_a?(CTEJoin)
              buckets[:join_node] << build_with_join_node(left_join.name, Arel::Nodes::OuterJoin)
            else
              raise ArgumentError, "only Hash, Symbol and Array are allowed"
            end

      def select_association_list(associations, stashed_joins = nil)
        result = []
        associations.each do |association|
          case association
          when Hash, Symbol, Array
            result << association
          when ActiveRecord::Associations::JoinDependency
            stashed_joins&.<< association
          else
            yield association if block_given?
          end

      def stream(key)
        blob = blob_for(key)

        chunk_size = 5.megabytes
        offset = 0

        raise ActiveStorage::FileNotFoundError unless blob.present?

        while offset < blob.properties[:content_length]
          _, chunk = client.get_blob(container, key, start_range: offset, end_range: offset + chunk_size - 1)
          yield chunk.force_encoding(Encoding::BINARY)
          offset += chunk_size
        end

